"""
experiments/scrape_inaturalist.py
=================================
Pull research-grade iNaturalist photo observations for CHIRP's 20
Stanford-campus species (or a subset thereof).

Why photos and not videos
-------------------------
iNaturalist's data model only stores **photos** and **sounds** —
there is no video media type, and the documented ``media_type=video``
query parameter is a no-op. We confirmed this against the live API.
So this script downloads photos; CHIRP's pipeline treats each photo
as a 1-frame "video" via ``num_frames=1`` (the picture-only branch).

Filtering
---------
By default we restrict to:

- ``quality_grade=research``  — community-verified IDs (≥2 agreeing IDers)
- ``place_id=14``              — California (covers Stanford campus)
- ``license=cc0,cc-by,cc-by-nc`` — research-use safe; commercial use
                                   still needs per-photo check for CC-BY-NC

You can narrow ``place_id`` further (e.g. Santa Clara County = 962)
or widen the licence set via CLI.

Output layout
-------------
::

    <out_root>/
    ├── index.csv                   # cols: path, label, species, source, obs_id, license
    └── <slug>/                     # one dir per species
        └── obs_<id>_photo_<i>.jpg

This matches what ``CHIRPVideoDataset`` reads, so the resulting
``index.csv`` plugs straight into the datamodule.

Usage
-----
::

    # Quick probe — just count photos, don't download
    python experiments/scrape_inaturalist.py --probe-only --max-per-species 50

    # Full pull of the 8 gap species at Stanford
    python experiments/scrape_inaturalist.py \\
        --species-set gap \\
        --max-per-species 300 \\
        --out data/inaturalist

    # All 20 species (slow — ~6000 photos)
    python experiments/scrape_inaturalist.py \\
        --species-set all \\
        --max-per-species 300 \\
        --out data/inaturalist
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from pipelines.video_dataset import SPECIES  # noqa: E402

logger = logging.getLogger(__name__)

API_BASE  = "https://api.inaturalist.org/v1"
USER_AGENT = "chirp-research-bot/0.1 (github.com/medha156/CHIRP)"

# place_ids (https://www.inaturalist.org/places)
PLACE_CALIFORNIA       = 14
PLACE_SANTA_CLARA_CO   = 962      # narrower if you want Stanford-area only

# Default split: 5 species are in VB100 (videos), 7 are in Birds-525 (photos),
# 8 must come from iNaturalist photos.
SPECIES_SETS: dict[str, list[str]] = {
    # 8 species that aren't in VB100 or Birds-525 → MUST come from iNaturalist
    "gap": [
        "American Crow",
        "Bushtit",
        "California Scrub-Jay",
        "Chestnut-backed Chickadee",
        "Cooper's Hawk",
        "Lesser Goldfinch",
        "Oak Titmouse",
        "Yellow-rumped Warbler",
    ],
    # All 20 — useful if you want iNat-only training without the other corpora
    "all": SPECIES,
}


# ---------------------------------------------------------------------------
# Tiny HTTP helper (stdlib only)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, *, retries: int = 3, backoff: float = 1.5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:                             # rate-limited
                wait = backoff ** attempt * 2
                logger.warning("429 from iNat; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} retries")


def _http_get_bytes(url: str, dest: Path, *, retries: int = 3) -> bool:
    """Stream a binary download to ``dest``. Returns True on success."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r, dest.open("wb") as f:
                while chunk := r.read(64 * 1024):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.debug("retry %d for %s: %s", attempt + 1, url, e)
            time.sleep(1.5 ** attempt)
    return False


# ---------------------------------------------------------------------------
# Species → taxon_id lookup
# ---------------------------------------------------------------------------

def lookup_taxon_id(common_name: str) -> int:
    """Return iNaturalist's numeric taxon_id for a bird's common name."""
    q = urllib.parse.urlencode({
        "q":    common_name,
        "rank": "species",
        "per_page": 5,
    })
    data = _http_get_json(f"{API_BASE}/taxa?{q}")
    # Prefer a result whose common name matches exactly.
    for r in data["results"]:
        if (r.get("preferred_common_name") or "").lower() == common_name.lower():
            return int(r["id"])
    # Fall back to the top result (only if Aves).
    for r in data["results"]:
        # ancestor_ids contains taxon path; 3 = class Aves on iNaturalist
        if 3 in (r.get("ancestor_ids") or []):
            return int(r["id"])
    raise LookupError(f"No taxon_id found for {common_name!r}")


# ---------------------------------------------------------------------------
# Observation iteration
# ---------------------------------------------------------------------------

def iter_observations(
    taxon_id: int,
    *,
    place_id: int = PLACE_CALIFORNIA,
    license_set: str = "cc0,cc-by,cc-by-nc",
    quality_grade: str = "research",
    max_results: int = 300,
    per_page: int = 100,
) -> Iterator[dict]:
    """Yield observation dicts up to ``max_results``.

    Uses ``id_below`` cursor pagination because iNat caps offset-based paging
    at 10 000 — fine for the volumes we want anyway.
    """
    n_yielded = 0
    id_below: int | None = None
    while n_yielded < max_results:
        params = {
            "taxon_id":      taxon_id,
            "place_id":      place_id,
            "quality_grade": quality_grade,
            "photo_license": license_set,
            "photos":        "true",
            "per_page":      min(per_page, max_results - n_yielded),
            "order":         "desc",
            "order_by":      "created_at",
        }
        if id_below is not None:
            params["id_below"] = id_below
        url = f"{API_BASE}/observations?{urllib.parse.urlencode(params)}"
        data = _http_get_json(url)
        results = data.get("results") or []
        if not results:
            return
        for r in results:
            yield r
            n_yielded += 1
            if n_yielded >= max_results:
                return
        id_below = results[-1]["id"]                      # cursor for next page
        time.sleep(0.3)                                   # be polite


# ---------------------------------------------------------------------------
# Per-photo download
# ---------------------------------------------------------------------------

def _photo_url(photo: dict, size: str = "medium") -> str | None:
    """Construct the URL for the requested photo size.

    The API returns a ``url`` that ends in ``/square.jpg``; we swap the
    last path component for the desired size. Available sizes:
    square (75), small (240), medium (500), large (1024), original.
    """
    base = photo.get("url")
    if not base:
        return None
    # Replace size segment: .../12345/square.jpg → .../12345/<size>.jpg
    parts = base.rsplit("/", 1)
    if len(parts) != 2:
        return base
    fname = parts[1]
    if "." not in fname:
        return base
    ext = fname.rsplit(".", 1)[-1]
    return f"{parts[0]}/{size}.{ext}"


def _safe_slug(name: str) -> str:
    return (
        name.lower()
            .replace("'", "")
            .replace(" ", "_")
            .replace("-", "_")
    )


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def scrape_species(
    common_name: str,
    chirp_idx: int,
    out_root: Path,
    *,
    place_id: int,
    license_set: str,
    max_per_species: int,
    photo_size: str,
    probe_only: bool,
) -> list[dict]:
    """Pull up to ``max_per_species`` photos for one species.

    Returns the list of rows to add to index.csv.
    """
    slug = _safe_slug(common_name)
    species_dir = out_root / slug
    if not probe_only:
        species_dir.mkdir(parents=True, exist_ok=True)

    try:
        taxon_id = lookup_taxon_id(common_name)
    except LookupError as e:
        logger.warning("  %s: %s", common_name, e)
        return []

    rows: list[dict] = []
    n_seen = n_downloaded = 0
    for obs in iter_observations(
        taxon_id,
        place_id=place_id,
        license_set=license_set,
        max_results=max_per_species,
    ):
        photos = obs.get("photos") or []
        for i, photo in enumerate(photos):
            url = _photo_url(photo, size=photo_size)
            if not url:
                continue
            fname = f"obs_{obs['id']}_photo_{i}.jpg"
            rel = f"{slug}/{fname}"
            dest = species_dir / fname

            n_seen += 1
            if probe_only:
                continue                                  # don't download

            if not dest.exists():
                ok = _http_get_bytes(url, dest)
                if not ok:
                    continue
                n_downloaded += 1
                time.sleep(0.05)                          # gentle throttle
            rows.append({
                "path":    rel,
                "label":   chirp_idx,
                "species": common_name,
                "source":  "inaturalist",
                "obs_id":  obs["id"],
                "license": photo.get("license_code") or "unknown",
            })

    if probe_only:
        logger.info("  %s [%d]: %d candidate photos found (no download)",
                    common_name, chirp_idx, n_seen)
    else:
        logger.info("  %s [%d]: %d photos in index (downloaded %d new)",
                    common_name, chirp_idx, len(rows), n_downloaded)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species-set", choices=list(SPECIES_SETS),
                        default="gap",
                        help="Which species to pull (default: 8-species gap set).")
    parser.add_argument("--out", default="data/inaturalist",
                        help="Output directory root.")
    parser.add_argument("--max-per-species", type=int, default=300)
    parser.add_argument("--place-id", type=int, default=PLACE_CALIFORNIA,
                        help="iNat place_id. 14=California (default), 962=Santa Clara Co.")
    parser.add_argument("--license-set", default="cc0,cc-by,cc-by-nc",
                        help="Comma-separated CC licence codes to allow.")
    parser.add_argument("--photo-size",
                        choices=["small", "medium", "large", "original"],
                        default="medium",
                        help="Which iNat photo size to download (default: medium ~500 px).")
    parser.add_argument("--probe-only", action="store_true",
                        help="Just count available photos per species, no download.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_root = Path(args.out)
    species_list = SPECIES_SETS[args.species_set]
    logger.info("Pulling %d species into %s (place_id=%d, max/species=%d, size=%s)",
                len(species_list), out_root, args.place_id,
                args.max_per_species, args.photo_size)

    if not args.probe_only:
        out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for name in species_list:
        try:
            idx = SPECIES.index(name)
        except ValueError:
            logger.warning("Skipping %r — not in CHIRP SPECIES list", name)
            continue
        rows = scrape_species(
            name, idx, out_root,
            place_id=args.place_id,
            license_set=args.license_set,
            max_per_species=args.max_per_species,
            photo_size=args.photo_size,
            probe_only=args.probe_only,
        )
        all_rows.extend(rows)

    # ---- write index.csv -------------------------------------------------
    if not args.probe_only:
        csv_path = out_root / "index.csv"
        with csv_path.open("w", newline="") as f:
            fields = ["path", "label", "species", "source", "obs_id", "license"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        logger.info("Wrote %s with %d photo rows", csv_path, len(all_rows))

        # Per-species summary
        from collections import Counter
        per_species = Counter(r["species"] for r in all_rows)
        print(f"\n{'='*60}\nPer-species counts (n={len(all_rows)} photos)\n{'='*60}")
        for sp in species_list:
            n = per_species.get(sp, 0)
            warn = "" if n >= 50 else "  ⚠ low"
            print(f"  {sp:30s}  {n:4d}{warn}")
    else:
        print("\n(probe-only — no files downloaded)")


if __name__ == "__main__":
    main()
