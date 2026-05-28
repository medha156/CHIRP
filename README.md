# CHIRP

**Bird species classification from short video clips** using a dual-backbone ensemble of **Video Swin-T** (temporal features) and **EfficientNet-B3** (per-frame features), evaluated on 20 Stanford-campus species.

[![CI](https://github.com/medha156/CHIRP/actions/workflows/ci.yml/badge.svg)](https://github.com/medha156/CHIRP/actions/workflows/ci.yml)

---

## Project overview

```
Video clip (T=16 frames, 224×224 RGB, optionally + optical flow)
        │
        ├──► Video Swin-T ─────────────► [B, 768]   temporal features
        │     (Kinetics-400 pretrained,
        │      patch 2×4×4, window 8×7×7)
        │
        └──► EfficientNet-B3 ──────────► [B, 1536]  per-frame features
              (ImageNet pretrained,
               4 keyframes, mean/max/attention pool)

                          │
                          ▼
              FusionHead (MLP 2304 → 512 → 20)
                          │
                          ▼
                     softmax over 20 species
```

The 20 classes are common Stanford campus birds — California Scrub-Jay, Acorn Woodpecker, Anna's Hummingbird, etc. Full list in [CLAUDE.md](CLAUDE.md) and [`pipelines/video_dataset.py`](pipelines/video_dataset.py).

### What's inside

| Path | Purpose |
|---|---|
| [`pipelines/`](pipelines/) | Dataset loader, preprocessor, augmentations, optical-flow channels, datamodule |
| [`models/`](models/) | Swin-T wrapper, EfficientNet-B3 encoder, fusion head, classical baselines (KNN / RF / XGBoost) |
| [`training/`](training/) | TrainConfig dataclasses, Trainer (AdamW + cosine warmup + early stop + WandB), CLI entry point |
| [`eval/`](eval/) | Test-set metrics + confusion matrix, SHAP feature importance, GradCAM + attention rollout |
| [`experiments/`](experiments/) | Ablation sweeps (model variants) + picture-vs-video temporal sweep |
| [`configs/`](configs/) | Sample YAML configs |
| [`tests/`](tests/) | 75 CPU-only unit tests (no model weights, no real videos) |

---

## Install

Requires Python 3.10–3.13. Poetry handles the dependency lock:

```bash
# 1. Clone
git clone https://github.com/medha156/CHIRP.git
cd CHIRP

# 2. Install via Poetry (preferred)
poetry install --extras dev      # adds pytest, ruff, mypy

# Or via pip if you don't want Poetry
pip install -e ".[dev]"
```

### Optional: fast video decoding with decord

By default the loader uses `torchvision.io.read_video` (slow but works everywhere). For ~5× faster decoding on Linux x86_64 / CUDA:

```bash
pip install chirp[decord]
# or:
pip install decord>=0.6.0
```

The loader auto-detects decord and falls back to torchvision otherwise — no config change needed.

---

## Datasets

CHIRP is trained on two bird-video corpora:

### FBD-SV-2024 — Flying Bird Detection, Surveillance Video (2024)

- **Source**: arXiv 2409.00317 ([paper](https://arxiv.org/abs/2409.00317))
- Surveillance-camera clips of flying birds (small, fast-moving)
- Released by the authors under a research license — request access via the contact in the paper

### VB100 — Video-Bird 100-class dataset

- **Source**: Ge et al., CVPR 2017 Workshops ([paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w8/papers/Ge_Animal_Recognition_in_CVPR_2017_paper.pdf))
- 100 bird species, naturalistic video clips
- Available via the original authors — see paper for request instructions

### Expected on-disk layout

After downloading, organise as follows:

```
data/
├── fbd_sv_2024/
│   ├── index.csv           # required: columns path, label, species
│   └── clips/              # actual .mp4 / .avi files (path entries
│                           # in index.csv are relative to this dir)
└── vb100/
    ├── index.csv
    └── clips/
```

### Building `index.csv`

Each `index.csv` must have **at minimum**: `path`, `label`, `species`. Labels are integer class indices 0–19 matching the [Stanford taxonomy in `pipelines/video_dataset.py`](pipelines/video_dataset.py).

Example for FBD-SV-2024:

```csv
path,label,species
clips/anna_hummingbird/clip_0001.mp4,3,Anna's Hummingbird
clips/california_scrub_jay/clip_0042.mp4,7,California Scrub-Jay
clips/red_tailed_hawk/clip_0117.mp4,17,Red-tailed Hawk
```

Many datasets ship with their own taxonomy; you'll need to write a small mapping script from the source labels to the 20-class CHIRP labels.

---

## Quickstart

### 1. Verify install + run tests

```bash
poetry run pytest tests/ -q          # 75 unit tests, CPU only, ~25s
```

### 2. Preprocess (sanity check the data pipeline)

The data pipeline is on-the-fly — there's no separate `preprocess.py` step. The `CHIRPDataModule` reads `index.csv`, splits 70/15/15 stratified, and writes a class-distribution plot for review:

```bash
# Smoke test on synthetic CSVs (writes outputs/class_distribution.png)
poetry run python pipelines/datamodule.py

# Or smoke-test the dataset loader directly on a real CSV
poetry run python pipelines/video_dataset.py data/fbd_sv_2024/index.csv \
    --n-frames 16 --batch-size 4
```

### 3. Train

```bash
# Full Swin-T + EB3 fusion ensemble (default)
poetry run python training/train.py --config configs/fusion.yaml

# Override individual fields without editing the YAML
poetry run python training/train.py --config configs/fusion.yaml \
    --override optim.lr=5e-4 model.dropout=0.4 data.use_optical_flow=true
```

This writes:
- `outputs/runs/fusion_baseline/checkpoints/best.pt` — best checkpoint
- `outputs/runs/fusion_baseline/config.yaml` — config snapshot
- `outputs/runs/fusion_baseline/class_distribution.png`
- `outputs/results.csv` — one row appended per run

### 4. Evaluate

```bash
poetry run python eval/evaluate.py \
    --config configs/fusion.yaml \
    --checkpoint outputs/runs/fusion_baseline/checkpoints/best.pt
```

Outputs to `outputs/figures/`: 2-panel confusion matrix + per-class F1 chart, plus `outputs/metrics/metrics_<run>.json`.

### 5. Visualise (SHAP + attention)

```bash
# Feature importance for RF / XGBoost baselines
poetry run python eval/shap_analysis.py \
    --config configs/fusion.yaml --baseline rf

# GradCAM + attention rollout overlay on a single clip
poetry run python eval/attention_maps.py \
    --config configs/fusion.yaml \
    --checkpoint outputs/runs/swin/checkpoints/best.pt \
    --clip data/fbd_sv_2024/clips/california_scrub_jay/clip_001.mp4
```

### 6. Ablations + temporal sweep

```bash
# (a)–(d): swin-only, eb3-only, classical heads (RF/XGB), pooling sweep
poetry run python experiments/run_ablations.py --base-config configs/fusion.yaml

# Picture-vs-video: T ∈ {1, 2, 4, 8, 16} across EB3 / Swin / fusion
poetry run python experiments/run_temporal_sweep.py --base-config configs/fusion.yaml

# Validate sweep configs without training (CI-friendly)
poetry run python experiments/run_ablations.py \
    --base-config configs/fusion.yaml --dry-run
```

Both write consolidated CSVs (`outputs/ablation_results.csv`, `outputs/temporal_sweep_results.csv`) plus per-experiment subdirectories under `outputs/runs/`.

---

## Results

*Placeholder — fill in after running the full sweep on real data.*

### Main models (test split, 20 classes)

| Model | Frames (T) | Optical flow | Top-1 acc | Macro-F1 |
|---|---|---|---|---|
| EfficientNet-B3 only | 1 (picture) | — | _TBD_ | _TBD_ |
| EfficientNet-B3 only | 16 | — | _TBD_ | _TBD_ |
| Video Swin-T only | 16 | ❌ | _TBD_ | _TBD_ |
| Video Swin-T only | 16 | ✅ | _TBD_ | _TBD_ |
| **Swin-T + EB3 fusion** | **16** | **❌** | _TBD_ | _TBD_ |
| **Swin-T + EB3 fusion** | **16** | **✅** | _TBD_ | _TBD_ |

### Classical baselines on frozen EB3 embeddings

| Head | Val F1 | Test F1 |
|---|---|---|
| KNN (k=5, cosine) | _TBD_ | _TBD_ |
| Random Forest (500 trees) | _TBD_ | _TBD_ |
| XGBoost (400 trees, depth 6) | _TBD_ | _TBD_ |

### EfficientNet pooling ablation (Macro-F1)

| Pool | Val F1 |
|---|---|
| mean | _TBD_ |
| max | _TBD_ |
| attention (learned) | _TBD_ |

### Picture-vs-video lift

| Backbone | T=1 (picture) | T=16 (video) | Δ |
|---|---|---|---|
| EfficientNet-B3 | _TBD_ | _TBD_ | _TBD_ |
| Video Swin-T | — (n/a) | _TBD_ | _TBD_ |
| Fusion | — | _TBD_ | _TBD_ |

---

## Development

```bash
# Run tests (CPU, no weight downloads, ~25s)
poetry run pytest tests/ -q

# Lint
poetry run ruff check .

# Type check
poetry run mypy src/chirp/
```

### Reproducibility

- Every training run snapshots its full config to `<output_dir>/config.yaml`
- Stratified splits seeded via `cfg.seed` (default 42)
- Best checkpoint includes optimizer + scheduler state for resumable training
- `outputs/results.csv` is append-only — each run is one row with timestamp + all hyperparameters

---

## Citation

If you use this codebase, please cite both source datasets:

```bibtex
@inproceedings{ge2017vb100,
  title={Animal Recognition in Camera-Trap Imagery},
  author={Ge, Z. and Bewley, A. and others},
  booktitle={CVPR Workshops},
  year={2017}
}

@article{fbd_sv_2024,
  title={Flying Bird Detection in Surveillance Video},
  year={2024},
  note={arXiv:2409.00317}
}
```

---

## License

See [LICENSE](LICENSE) (TBD — currently unlicensed; defaults to "all rights reserved" until a license file is added).
