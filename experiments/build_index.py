"""
experiments/build_index.py
==========================
Merge the three data sources into a single CHIRP-compatible
``data/<dataset>/index.csv`` (or a unified ``merged/index.csv``).

Sources and how they're handled
-------------------------------
1. **VB100** (Zenodo) — *real videos*, 5/20 Stanford species covered.
   Layout produced by ``vb100_video_NN.zip`` extraction:

       data/raw/vb100/extracted/<Species_Name>/<clip>.mp4

   We rename folders to the CHIRP canonical species string and copy
   (or symlink) clips into ``data/vb100/clips/<slug>/...``.

2. **Birds-525** (HuggingFace) — *photos*, 8/20 Stanford species covered.
   Loaded via the ``datasets`` library. Each photo is written as a JPG
   under ``data/birds525/clips/<slug>/img_<i>.jpg``.

3. **iNaturalist** (already downloaded by ``scrape_inaturalist.py``) —
   *photos*, the 8 species that aren't in either other corpus.
   We just consume the index.csv that scraper produced.

Output
------
By default we write **one ``index.csv`` per source** (so each behaves
like a separate CHIRP dataset the datamodule can load alongside the
others), plus an optional ``--unified`` mode that concatenates everything
into a single ``data/merged/index.csv`` for simpler training.

Each row carries the standard CHIRP columns plus a ``source`` and
``modality`` ({photo, video}) tag so downstream code can branch if it
wants to (e.g. set num_frames=1 for photos).

Usage
-----
::

    # After running scrape_inaturalist.py and downloading + unzipping VB100:
    python experiments/build_index.py \\
        --vb100-extracted data/raw/vb100/extracted \\
        --birds525-out    data/birds525 \\
        --inat-index      data/inaturalist/index.csv \\
        --unified-out     data/merged/index.csv

    # Skip B525 (e.g. if you don't have hf credentials)
    python experiments/build_index.py --skip-birds525 ...
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import logging
import shutil
from pathlib import Path

import pandas as pd

from pipelines.video_dataset import NUM_CLASSES, SPECIES  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source → CHIRP species mappings
# ---------------------------------------------------------------------------

# VB100 uses underscored names like "Acorn_Woodpecker". Convert via _norm.
# Birds-525 uses ALLCAPS like "ANNAS HUMMINGBIRD". Same _norm trick works.
def _norm(s: str) -> str:
    return (s.lower()
             .replace("'", "")
             .replace("-", " ")
             .replace("_", " ")
             .strip())


SPECIES_LOOKUP = {_norm(name): idx for idx, name in enumerate(SPECIES)}


def chirp_index_for(source_name: str) -> int | None:
    """Map a source-side species string → CHIRP class idx, or None if unknown."""
    return SPECIES_LOOKUP.get(_norm(source_name))


def _slug(species_name: str) -> str:
    return _norm(species_name).replace(" ", "_")


# ---------------------------------------------------------------------------
# VB100 → CHIRP
# ---------------------------------------------------------------------------

def build_vb100_index(
    extracted_root: Path,
    out_root: Path,
    *,
    copy_files: bool = False,
) -> list[dict]:
    """Walk VB100's extracted layout and produce CHIRP-compatible rows.

    VB100 extracts as one folder per species (named like
    ``Acorn_Woodpecker``) containing one or more video files.
    Only species that map to a CHIRP class are kept.

    Parameters
    ----------
    extracted_root:
        Path to the directory that contains the per-species folders.
    out_root:
        Output dataset root (``data/vb100``).
    copy_files:
        When False (default) we register the *original* extracted paths
        in index.csv and don't copy anything (saves ~10 GB of duplicate
        storage). When True we copy each clip into
        ``<out_root>/clips/<slug>/...`` to make the data self-contained.
    """
    if not extracted_root.exists():
        logger.warning("VB100 extracted dir not found: %s", extracted_root)
        return []

    clips_dir = out_root / "clips"
    if copy_files:
        clips_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    species_dirs = sorted(p for p in extracted_root.iterdir() if p.is_dir())
    for sp_dir in species_dirs:
        chirp_idx = chirp_index_for(sp_dir.name)
        if chirp_idx is None:
            continue                                       # not a CHIRP species
        chirp_name = SPECIES[chirp_idx]
        slug = _slug(chirp_name)

        for clip in sorted(sp_dir.iterdir()):
            if clip.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                continue

            if copy_files:
                (clips_dir / slug).mkdir(parents=True, exist_ok=True)
                dest = clips_dir / slug / clip.name
                if not dest.exists():
                    shutil.copy2(clip, dest)
                rel = f"clips/{slug}/{clip.name}"
            else:
                # Register absolute path so the datamodule's root_dir can be empty
                rel = str(clip.resolve())

            rows.append({
                "path":     rel,
                "label":    chirp_idx,
                "species":  chirp_name,
                "source":   "vb100",
                "modality": "video",
                "license":  "CC-BY-NC-SA-4.0",
            })

    logger.info("VB100 → %d video rows across %d Stanford species",
                len(rows), len({r['label'] for r in rows}))
    return rows


# ---------------------------------------------------------------------------
# Birds-525 → CHIRP
# ---------------------------------------------------------------------------

def build_birds525_index(
    out_root: Path,
    *,
    max_per_species: int = 200,
    hf_dataset: str = "yashikota/birds-525-species-image-classification",
) -> list[dict]:
    """Download Birds-525 photos from HuggingFace and emit CHIRP rows.

    Only the 8 Stanford species that exist in Birds-525 are kept.
    Images are written as JPGs into ``<out_root>/clips/<slug>/img_<i>.jpg``
    so the CHIRPVideoDataset can decode them as 1-frame "videos".

    Returns the list of rows; also writes ``<out_root>/index.csv``.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("`datasets` library not installed. "
                     "Run: poetry add --group dev datasets")
        return []

    logger.info("Loading Birds-525 from HuggingFace (streaming)…")
    ds = load_dataset(hf_dataset, split="train", streaming=True)
    label_names = ds.features["label"].names if hasattr(ds, "features") else None
    if label_names is None:
        # Streaming datasets sometimes hide features; fetch them differently
        from datasets import load_dataset_builder
        info = load_dataset_builder(hf_dataset).info
        label_names = info.features["label"].names

    # B525_idx → CHIRP_idx (only the matching species)
    b525_to_chirp: dict[int, int] = {}
    for b_idx, b_name in enumerate(label_names):
        ch = chirp_index_for(b_name)
        if ch is not None:
            b525_to_chirp[b_idx] = ch
    logger.info("  matched %d Birds-525 species → CHIRP classes",
                len(b525_to_chirp))

    clips_dir = out_root / "clips"
    rows: list[dict] = []
    per_species_count: dict[int, int] = {}

    # Stream and write only the matching examples.
    for ex in ds:
        b_label = ex["label"]
        if b_label not in b525_to_chirp:
            continue
        chirp_idx = b525_to_chirp[b_label]
        if per_species_count.get(chirp_idx, 0) >= max_per_species:
            continue

        chirp_name = SPECIES[chirp_idx]
        slug = _slug(chirp_name)
        sp_dir = clips_dir / slug
        sp_dir.mkdir(parents=True, exist_ok=True)

        idx = per_species_count.get(chirp_idx, 0)
        fname = f"img_{idx:04d}.jpg"
        dest = sp_dir / fname
        if not dest.exists():
            ex["image"].convert("RGB").save(dest, "JPEG", quality=90)

        rows.append({
            # absolute path so the merged index doesn't need a root_dir
            "path":     str(dest.resolve()),
            "label":    chirp_idx,
            "species":  chirp_name,
            "source":   "birds525",
            "modality": "photo",
            "license":  "CC0",
        })
        per_species_count[chirp_idx] = idx + 1

        # Stop once we have enough of every matched class
        if all(per_species_count.get(c, 0) >= max_per_species
               for c in b525_to_chirp.values()):
            break

    logger.info("Birds-525 → %d photo rows across %d Stanford species",
                len(rows), len({r['label'] for r in rows}))
    return rows


# ---------------------------------------------------------------------------
# iNaturalist (already downloaded — just re-tag rows)
# ---------------------------------------------------------------------------

def build_inat_index(inat_csv: Path, inat_root: Path) -> list[dict]:
    """Take the iNaturalist scraper's CSV and standardise the row schema."""
    if not inat_csv.exists():
        logger.warning("iNaturalist index not found: %s", inat_csv)
        return []
    df = pd.read_csv(inat_csv)
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append({
            "path":     str((inat_root / r["path"]).resolve()),
            "label":    int(r["label"]),
            "species":  r["species"],
            "source":   "inaturalist",
            "modality": "photo",
            "license":  r.get("license", "cc-by-nc"),
        })
    logger.info("iNaturalist → %d photo rows across %d Stanford species",
                len(rows), len({r['label'] for r in rows}))
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

CHIRP_INDEX_FIELDS = ["path", "label", "species", "source", "modality", "license"]


def write_index(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHIRP_INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_coverage_summary(rows: list[dict]) -> None:
    from collections import defaultdict
    per_class: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        per_class[int(r["label"])].append((r["source"], r["modality"]))

    print("\n" + "=" * 78)
    print(f"{'class':>5}  {'species':<28} {'#total':>7} {'video':>7} {'photo':>7}  sources")
    print("=" * 78)
    total_video = total_photo = total_classes = 0
    for cls in range(NUM_CLASSES):
        items = per_class.get(cls, [])
        if items:
            total_classes += 1
        n_video = sum(1 for _s, m in items if m == "video")
        n_photo = sum(1 for _s, m in items if m == "photo")
        total_video += n_video
        total_photo += n_photo
        sources = ",".join(sorted({s for s, _m in items})) if items else "—"
        flag = "" if items else "  ⚠ MISSING"
        print(f"  {cls:>3}  {SPECIES[cls]:<28} {len(items):>7} "
              f"{n_video:>7} {n_photo:>7}  {sources}{flag}")
    print("=" * 78)
    print(f"  Coverage: {total_classes}/{NUM_CLASSES} classes have ≥1 sample")
    print(f"  Totals:   {total_video} video + {total_photo} photo = "
          f"{total_video + total_photo} samples")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vb100-extracted", default="data/raw/vb100/extracted",
                        help="Directory containing per-species VB100 subfolders.")
    parser.add_argument("--vb100-out",       default="data/vb100",
                        help="Where to write the VB100 index.csv (clips registered in-place by default).")
    parser.add_argument("--vb100-copy-files", action="store_true",
                        help="Copy VB100 mp4s into vb100-out/clips/ instead of registering original paths.")
    parser.add_argument("--skip-vb100", action="store_true")

    parser.add_argument("--birds525-out", default="data/birds525")
    parser.add_argument("--birds525-max-per-species", type=int, default=200)
    parser.add_argument("--skip-birds525", action="store_true")

    parser.add_argument("--inat-index", default="data/inaturalist/index.csv")
    parser.add_argument("--inat-root",  default="data/inaturalist")
    parser.add_argument("--skip-inat", action="store_true")

    parser.add_argument("--unified-out", default="data/merged/index.csv",
                        help="Where to write the concatenated all-sources index.csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    all_rows: list[dict] = []

    # ---- VB100 ----------------------------------------------------
    if not args.skip_vb100:
        vb_rows = build_vb100_index(
            Path(args.vb100_extracted), Path(args.vb100_out),
            copy_files=args.vb100_copy_files,
        )
        if vb_rows:
            write_index(vb_rows, Path(args.vb100_out) / "index.csv")
        all_rows.extend(vb_rows)

    # ---- Birds-525 ------------------------------------------------
    if not args.skip_birds525:
        b_rows = build_birds525_index(
            Path(args.birds525_out),
            max_per_species=args.birds525_max_per_species,
        )
        if b_rows:
            write_index(b_rows, Path(args.birds525_out) / "index.csv")
        all_rows.extend(b_rows)

    # ---- iNaturalist ----------------------------------------------
    if not args.skip_inat:
        i_rows = build_inat_index(Path(args.inat_index), Path(args.inat_root))
        all_rows.extend(i_rows)

    # ---- unified --------------------------------------------------
    if all_rows:
        write_index(all_rows, Path(args.unified_out))

    print_coverage_summary(all_rows)


if __name__ == "__main__":
    main()
