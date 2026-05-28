"""
experiments/generate_synthetic_data.py
======================================
Generate a small synthetic CHIRP-shaped video dataset for end-to-end
pipeline integration tests.

Each "species" gets a distinct, learnable visual fingerprint:
- A unique base hue
- A unique moving-shape pattern (circle / rect / triangle, varying radius
  and trajectory)

The signal is intentionally easy so a model with very few training epochs
can show non-trivial above-chance behaviour — this verifies the pipeline
plumbing (data → preprocess → model → loss → checkpoint) without
requiring real bird videos.

Layout produced
---------------
    <out>/
    ├── fbd_sv_2024/
    │   ├── index.csv          # cols: path,label,species
    │   └── clips/<species>/clip_000.mp4 ... clip_NNN.mp4
    └── vb100/
        ├── index.csv
        └── clips/<species>/clip_000.mp4 ...

Usage
-----
::

    python experiments/generate_synthetic_data.py \\
        --out data/synthetic --clips-per-class 8 --duration 2.0 --fps 8 --size 96
"""

from __future__ import annotations

# Direct-script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipelines.video_dataset import NUM_CLASSES, SPECIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-class visual fingerprint
# ---------------------------------------------------------------------------

def _class_fingerprint(class_idx: int) -> dict:
    """Return a deterministic visual fingerprint for a class.

    Each class gets:
      - A base BGR colour (well-separated in HSV space).
      - A shape (circle / rect / triangle) cycling deterministically.
      - A motion pattern (linear / circular / vertical / horizontal).
      - A speed multiplier.
    """
    # HSV colour evenly spaced around the wheel.
    hue   = int(180 * class_idx / NUM_CLASSES)
    sat   = 200
    val   = 220
    hsv   = np.uint8([[[hue, sat, val]]])
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

    shapes  = ["circle", "rect", "triangle"]
    motions = ["linear", "circular", "vertical", "horizontal"]
    return {
        "color":   bgr,
        "shape":   shapes[class_idx % len(shapes)],
        "motion":  motions[(class_idx // len(shapes)) % len(motions)],
        "speed":   0.6 + 0.05 * class_idx,
    }


def _render_frame(
    t: float,                    # normalised time in [0, 1]
    size: int,
    fingerprint: dict,
    seed: int,
) -> np.ndarray:
    """Render one frame. Adds light noise + position jitter for variety."""
    rng = np.random.default_rng(seed + int(t * 10_000))
    img = np.full((size, size, 3), 30, dtype=np.uint8)       # dark grey BG
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)     # tiny noise

    cx_base, cy_base = size // 2, size // 2
    motion = fingerprint["motion"]
    speed  = fingerprint["speed"]

    # Compute position based on motion pattern.
    angle  = 2 * np.pi * t * speed
    radius = size // 4
    if motion == "linear":
        cx = int(size * 0.2 + (size * 0.6) * t)
        cy = cy_base
    elif motion == "circular":
        cx = cx_base + int(radius * np.cos(angle))
        cy = cy_base + int(radius * np.sin(angle))
    elif motion == "vertical":
        cy = int(size * 0.2 + (size * 0.6) * t)
        cx = cx_base
    else:                                                     # horizontal oscillation
        cx = cx_base + int(radius * np.cos(angle))
        cy = cy_base

    # Per-clip position jitter so two clips of the same class differ.
    cx += int(rng.integers(-3, 4))
    cy += int(rng.integers(-3, 4))

    color = fingerprint["color"]
    shape = fingerprint["shape"]
    r = size // 8

    if shape == "circle":
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
    elif shape == "rect":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness=-1)
    else:                                                     # triangle
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], color)

    return img


def _write_clip(path: Path, class_idx: int, seed: int, *,
                size: int, fps: int, duration: float) -> None:
    fp = _class_fingerprint(class_idx)
    n_frames = max(2, int(round(fps * duration)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open writer for {path}")
    try:
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)                      # normalised [0,1]
            frame = _render_frame(t, size, fp, seed=seed)
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------

def write_dataset(
    out_root: Path,
    dataset_name: str,
    clips_per_class: int,
    *,
    size: int, fps: int, duration: float, seed_base: int,
) -> Path:
    """Write a synthetic CHIRP dataset directory and return its index.csv path."""
    root = out_root / dataset_name
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in range(NUM_CLASSES):
        slug = SPECIES[cls].lower().replace(" ", "_").replace("'", "").replace("-", "_")
        (clips_dir / slug).mkdir(parents=True, exist_ok=True)
        for i in range(clips_per_class):
            rel = Path("clips") / slug / f"clip_{i:03d}.mp4"
            path = root / rel
            seed = seed_base + cls * 100 + i
            _write_clip(path, cls, seed,
                        size=size, fps=fps, duration=duration)
            rows.append({"path": str(rel), "label": cls, "species": SPECIES[cls]})

    index = pd.DataFrame(rows)
    csv_path = root / "index.csv"
    index.to_csv(csv_path, index=False)
    logger.info("Wrote %s — %d clips at %d×%d, %d fps × %.1fs",
                csv_path, len(rows), size, size, fps, duration)
    return csv_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/synthetic",
                        help="Root directory to write the synthetic data root.")
    parser.add_argument("--clips-per-class", type=int, default=8,
                        help="How many clips per species per dataset.")
    parser.add_argument("--size", type=int, default=96,
                        help="Square frame size (px). 96 keeps disk + decode cheap.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Seconds per clip.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    csv1 = write_dataset(out_root, "fbd_sv_2024",
                         clips_per_class=args.clips_per_class,
                         size=args.size, fps=args.fps, duration=args.duration,
                         seed_base=0)
    csv2 = write_dataset(out_root, "vb100",
                         clips_per_class=max(1, args.clips_per_class // 2),
                         size=args.size, fps=args.fps, duration=args.duration,
                         seed_base=5000)

    total_clips = NUM_CLASSES * (args.clips_per_class + max(1, args.clips_per_class // 2))
    print(f"\nGenerated {total_clips} synthetic clips under {out_root}/")
    print(f"  - {csv1}")
    print(f"  - {csv2}")


if __name__ == "__main__":
    main()
