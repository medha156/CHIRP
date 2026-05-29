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

CHIRP uses **three free, openly-downloadable sources** to cover the 20 Stanford species — none requires a research-license email. The earlier project plan named FBD-SV-2024 and VB100 as primary sources; investigation showed FBD-SV-2024 doesn't have species labels and VB100 only covers 5 of our 20 species, so the actual mix is:

| Source | Type | Stanford species covered | License | Where |
|---|---|---|---|---|
| **VB100** | Videos | **5** (Acorn Woodpecker, Black Phoebe, California Towhee, Red-tailed Hawk, White-crowned Sparrow) | CC BY-NC-SA 4.0 | [Zenodo 60375](https://zenodo.org/record/60375) (9.7 GB) |
| **Birds-525** | Photos | **8** (American Robin, Anna's Hummingbird, Brewer's Blackbird, Dark-eyed Junco, House Finch, Mourning Dove, Northern Mockingbird, Red-tailed Hawk) | CC0 | [HuggingFace](https://huggingface.co/datasets/yashikota/birds-525-species-image-classification) (~2 GB) |
| **iNaturalist** | Photos | **8 gap species** (American Crow, Bushtit, California Scrub-Jay, Chestnut-backed Chickadee, Cooper's Hawk, Lesser Goldfinch, Oak Titmouse, Yellow-rumped Warbler) | CC0 / CC-BY / CC-BY-NC per photo | [api.inaturalist.org](https://api.inaturalist.org) |

iNaturalist only stores **photos** and **sounds** — no video media type exists in their data model. So 5 of our 20 classes get real videos (VB100); the other 15 get photos that the CHIRP pipeline treats as 1-frame "videos" via `num_frames=1`. This is honest: a 20-class classifier trained on mixed video + photo input.

### One-command pull

```bash
# 1. VB100 videos — 9.7 GB across 22 archives
mkdir -p data/raw/vb100/archives && cd data/raw/vb100/archives
for i in $(seq -w 01 22); do
    curl -L -O "https://zenodo.org/record/60375/files/vb100_video_${i}.zip"
done
for f in *.zip; do unzip -q "$f" -d ../extracted/; done
cd -

# 2. iNaturalist photos for the 8 gap species (Bay Area, research-grade, CC)
poetry run python experiments/scrape_inaturalist.py \
    --species-set gap --max-per-species 300 --out data/inaturalist

# 3. Birds-525 photos (downloaded inline by build_index.py)

# 4. Build the unified CHIRP index.csv
poetry run python experiments/build_index.py \
    --vb100-extracted data/raw/vb100/extracted \
    --birds525-out    data/birds525 \
    --inat-index      data/inaturalist/index.csv \
    --unified-out     data/merged/index.csv
```

After this you'll have:
- `data/vb100/index.csv` — 5 species × ~14 video clips
- `data/birds525/index.csv` — 8 species × ≤200 photos each
- `data/inaturalist/index.csv` — 8 species × ≤300 photos each
- `data/merged/index.csv` — concatenated all-sources index for training

The `index.csv` schema is `path, label, species, source, modality, license`. The `source` and `modality` columns let downstream code branch (e.g. set `num_frames=1` automatically when `modality==photo`).

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
