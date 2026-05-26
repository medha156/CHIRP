# CHIRP — Bird Species Classification

**C**lassification of **H**avitat **I**nhabitants via **R**apid video **P**rocessing

## Project Overview

CHIRP classifies bird species from short video clips (2–10 seconds) using a dual-model ensemble:

- **Video Swin-T** — captures temporal motion patterns across frames (wing beats, flight paths, foraging behavior)
- **EfficientNet-B3** — extracts fine-grained visual features from key frames (plumage, beak shape, size)

The outputs are fused via learned attention weighting for a final 20-class prediction.

## Target Classes (20 species — Stanford campus)

Species chosen from the Bay Area oak-woodland + urban avifauna reliably
observed across the Main Quad, Arboretum, the Dish, and San Francisquito
Creek. Alphabetised so class indices are predictable from the name alone.

| # | Species | # | Species |
|---|---------|---|---------|
| 0 | Acorn Woodpecker | 10 | Cooper's Hawk |
| 1 | American Crow | 11 | Dark-eyed Junco |
| 2 | American Robin | 12 | House Finch |
| 3 | Anna's Hummingbird | 13 | Lesser Goldfinch |
| 4 | Black Phoebe | 14 | Mourning Dove |
| 5 | Brewer's Blackbird | 15 | Northern Mockingbird |
| 6 | Bushtit | 16 | Oak Titmouse |
| 7 | California Scrub-Jay | 17 | Red-tailed Hawk |
| 8 | California Towhee | 18 | White-crowned Sparrow |
| 9 | Chestnut-backed Chickadee | 19 | Yellow-rumped Warbler |

## Directory Layout

```
chirp/
├── data/               # Raw and processed video clips + CSVs
│   ├── raw/            # Original .mp4 clips, organized by species
│   ├── processed/      # Resized/trimmed clips + extracted frames
│   └── splits/         # train.csv, val.csv, test.csv
├── models/             # Model definitions and checkpoints
│   ├── video_swin.py   # Video Swin-T wrapper
│   ├── efficientnet.py # EfficientNet-B3 wrapper
│   └── ensemble.py     # Attention fusion head
├── pipelines/          # Training and evaluation pipelines
│   ├── train.py        # Main training loop
│   ├── evaluate.py     # Per-class metrics + confusion matrix
│   └── infer.py        # Single-clip inference script
├── utils/              # Shared helpers
│   ├── dataset.py      # VideoDataset, frame sampling
│   ├── transforms.py   # Video augmentation pipeline
│   ├── metrics.py      # Accuracy, macro-F1, top-k
│   └── config.py       # Hydra/YAML config loader
├── notebooks/          # Exploratory analysis and visualization
├── outputs/            # Run artifacts (logs, checkpoints, plots)
├── src/chirp/          # Installable package (Poetry source root)
├── tests/              # pytest unit tests
├── pyproject.toml
└── CLAUDE.md
```

## Architecture

```
Video clip (T frames, 224×224)
        │
        ├──► Video Swin-T ──► temporal features [B, 768]
        │         (patch: 2×4×4, window: 8×7×7)
        │
        └──► EfficientNet-B3 ──► frame features [B, 1536]
                  (applied to 4 uniformly sampled keyframes, pooled)
        │
        └──► Attention Fusion Head ──► 20-class logits
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input resolution | 224 × 224 |
| Clip length | 16 frames (uniform sample) |
| Batch size | 16 |
| Optimizer | AdamW, lr=1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Label-smoothed CrossEntropy (ε=0.1) |
| Epochs | 50 |
| Augmentation | RandAugment + temporal jitter + mixup |

## Quick Start

```bash
# Install dependencies
poetry install

# Prepare data splits
python pipelines/train.py --config configs/base.yaml

# Evaluate
python pipelines/evaluate.py --checkpoint outputs/best.pt

# Single clip inference
python pipelines/infer.py --video path/to/clip.mp4
```

## Development Notes

- Frame sampling is **uniform** by default; temporal jitter is applied during training only.
- Video Swin-T weights are initialized from Kinetics-400 pretrained checkpoint.
- EfficientNet-B3 weights are initialized from ImageNet pretrained checkpoint.
- Mixed precision (bf16) is enabled by default on CUDA devices.
- Use `notebooks/` for EDA and per-class error analysis; keep notebooks clean (clear outputs before committing).
- All experiment configs live in `configs/` (not tracked in git if they contain paths).
