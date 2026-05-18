# F2FMatcher — Fiber-to-Fiber Matching Across Histological Stains

F2FMatcher matches individual muscle fibers (myofibers) across pairs of histological images from serial sections of the same muscle sample. It handles cross-stain matching (e.g., immunofluorescence vs histochemistry, different IHC markers) by combining a **VAE-based latent representation** of Cellpose flow fields, a **pairwise classifier** on latent embeddings, and a **geometry-aware iterative matching algorithm** that propagates matches using spatial consistency constraints.

![Matching example](video/comparison.gif)

*Fiber matching across serial muscle sections stained with different markers. Each section is segmented by Cellpose to delineate individual fibers; Cellpose flow fields are encoded into a VAE latent space, then fibers are matched across stains by a pairwise classifier whose predictions are refined through iterative geometry-constrained label propagation.*

## Pipeline Overview

```
CZI/PNG input
    │
    ▼
1. Image I/O — Export CZI to PNG, resize to reference pixel resolution
    │
    ▼
2. Cellpose Segmentation — Finetuned Cellpose models → masks + flow fields (x, y, cell probability)
    │
    ▼
3. ROI Cropping — 256×256 crops around each fiber centroid → .npz (flow_x, flow_y, roi_mask)
    │
    ▼
4. VAE Embedding — Convert flow fields to mag/angle, resize 128×128, encode → 256-dim latent vector
    │
    ▼
5. Pairwise Classifier — Score all cross-image ROI pairs → N₁ × N₂ logit matrix
    │
    ▼
6. Matching Algorithm — Cost = classifier × spatial signature → initial guess (triangle geo. filter) → iterative local propagation → Hungarian fill
    │
    ▼
7. Output — paired_labels.pkl + visualization PNGs (matched contours with pair IDs)
    │
    ▼
8. Quantification — Per-channel staining intensity in matched ROIs (membrane, cytoplasm, whole)
```

---

## Model Architecture

### VAE: SharedMultiHeadVAE

The VAE compresses Cellpose flow fields (128×128 crops) into 256-dimensional latent vectors.

**Encoder (VAEEncoder):**
```
Input: 3 × 128 × 128 (mag, angle, roi_mask)
    │
    ├─ Conv2d(3→64) → ReLU
    ├─ Conv2d(64→64) → ReLU → MaxPool(2×2)
    ├─ Conv2d(64→128) → ReLU
    ├─ Conv2d(128→128) → ReLU → MaxPool(2×2)
    ├─ Conv2d(128→256) → ReLU
    ├─ Conv2d(256→256) → ReLU → AdaptiveAvgPool(4×4)
    │
    ├─ fc_mu: Linear(4096 → 256)
    └─ fc_logvar: Linear(4096 → 256)
```

**Latent space:** 256-dim, trained with reparameterization trick.

**Decoder (Shared + multi-head):**
```
Latent z (256-dim)
    │
 Linear(256 → 512×8×8) → reshape (512, 8, 8)
    │
 Shared decoder (transposed conv):
    ConvTranspose2d(512→256, 4, stride=2) → ReLU  # 16×16
    Conv2d(256→256, 3) → ReLU
    ConvTranspose2d(256→128, 4, stride=2) → ReLU  # 32×32
    Conv2d(128→128, 3) → ReLU
    ConvTranspose2d(128→64, 4, stride=2) → ReLU   # 64×64
    Conv2d(64→64, 3) → ReLU
    ConvTranspose2d(64→32, 4, stride=2) → ReLU    # 128×128
    │
 Three output heads (each: Conv2d(32→16) → ReLU → Conv2d(16→1)):
    ├── head_fx:   reconstructs flow_x
    ├── head_fy:   reconstructs flow_y
    └── head_mask: reconstructs roi_mask
```

**Training losses:**
- **Reconstruction loss:** MSE on flow_x, flow_y, roi_mask (sum)
- **KL divergence:** weighted by β_KL (default 0.001)
- **Latent consistency:** MSE between paired augmented views (β_consistency = 1.0)

### Pairwise Classifier (PairClassifier)

A binary classifier that determines whether two ROI embeddings originate from the same fiber.

```
Embedding₁ (256-dim) ─┐
                       ├─ concat → Linear(512→128) → ReLU → Dropout(0.5)
Embedding₂ (256-dim) ─┘           → Linear(128→64) → ReLU → Dropout(0.5)
                                  → Linear(64→1) → Sigmoid → score [0, 1]
```

**Training:** BCELoss, Adam (lr=1e-4), early stopping on validation F1 (patience=10). Negative pairs are sampled at 4× the rate of positive pairs (`negative_fold=4`). Best checkpoint achieves val F1 ≈ 0.944.

---

## Matching Strategy

The matching algorithm (`match_fibers` in `src/f2fmatcher/matching/matcher.py`) proceeds in four stages:

### Stage 1: Cost Matrix Computation

Two complementary costs are computed for every cross-image pair of ROIs:

| Cost component | Description |
|---|---|
| **Classifier logits** `S[i,j]` | PairClassifier score: how likely ROI i (img1) and ROI j (img2) are the same fiber |
| **Spatial signature similarity** `W[i,j]` | Geometric niche: k-nearest-neighbor distance vectors (k=3,5,7) are compared via Wasserstein distance across scales, then geometrically averaged |

**Combined cost:** `cost[i,j] = S[i,j] × W[i,j]`

### Stage 2: Initial Guess with Geometric Filtering

1. Select the top N (default N=80) highest-scoring pairs from the cost matrix (greedy, no duplicates)
2. For every combination of `n_pair_selected` (default 4) among those N:
   - Check **triangle geometry**: for all triplets within the combination, compare side lengths and angles of the triangles formed by the centroids of images 1 and 2
   - Require `cost_sides < 30` AND `cost_angles < 0.15` for every triplet
3. Only pairs appearing in at least one valid combination are kept as seeds

This ensures the initial set is geometrically consistent under local affine transformations between the two serial sections.

### Stage 3: Iterative Local Propagation

Starting from the validated seed pairs, the algorithm iteratively expands:

```
For each matched pair (A₁, A₂):
    1. Find all ROIs within distance_neighbors_ref (200 px) of A₁ in image 1
       and of A₂ in image 2
    2. Among those neighbors, consider unannotated cross-image pairs
    3. For the highest-scoring candidate pair (B₁, B₂):
        a. Find the 3 nearest already-matched ROIs to B₁ in image 1
        b. Match them to their paired ROIs in image 2
        c. Check triangle geometry: (B₁, neighbor_i, neighbor_j) vs (B₂, matched_i, matched_j)
        d. Accept if geometric distortion is below thresholds
    4. Update the prediction matrix and repeat until <0.25% new pairs per step
```

**Convergence** is reached when fewer than 0.25% of total ROIs are added in one step.

### Stage 4: Fill Unannotated ROIs

For remaining unmatched ROIs:
1. Estimate an **affine transform** from all matched centroids (img₁ → img₂)
2. Transform each unmatched ROI centroid from image 1 into image 2 coordinates
3. For each, find candidate ROIs in image 2 within `max_distance_affine` (150 px) with classifier score > `min_cls_logit`
4. Filter candidates by triangle geometry against nearest matched neighbors
5. Add the best valid match

### Post-processing

- **Deduplication:** Remove any ROI matched to multiple partners
- **Validation filtering:** Re-check every match against its 3 nearest matched neighbors using triangle geometry

---

## Installation

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate fibermatcher

# 2. Install the package in development mode
cd /DATA/F2FMatcher
pip install -e .

# 3. Model checkpoints are included in models/
#    - VAE: models/LatentVAE2_256_128.pth
#    - Classifier: models/fibermatcher_cls_2.pth
#    - Cellpose models (external, set path in config):
#      /media/DATABRUT/DB_DDC/serverGPU/CellPose_DDC/CP_model_zoo/models/
```

## Usage

### Single image pair

```bash
f2fmatcher run-pipeline \
    --img1 TAG01 --img2 TAG01 \
    --source1 /path/to/imgs1 --source2 /path/to/imgs2 \
    --cp-model-1 "CP_AV_Laminin_Dia_Qua_TA_AxioScan10X" \
    --cp-model-2 "CP_AV_TA_COX-SDH-NADH_AxioScan10X" \
    --param-img1 fluorescence --param-img2 brightfield \
    --obj1 40X --obj2 10X --channel1 0 --channel2 0 \
    --czi1 --czi2 \
    --output /path/to/output \
    --export-images
```

### Batch processing (Python param file, backward-compatible)

```bash
f2fmatcher run-pipeline --param-file /path/to/param_file.py
```

Where `param_file.py` defines: `source_1`, `source_2`, `czi_img1`, `channel_index_img1`, `CP_model_name_1`, `list_pair_images`, etc.

### Training

```bash
# Train VAE
f2fmatcher train-vae --config configs/default.yaml --checkpoint-dir ./checkpoints

# Train classifier
f2fmatcher train-classifier --config configs/default.yaml --checkpoint-path ./classifier.pth
```

## Configuration

All pipeline parameters are in `configs/default.yaml`. Key sections:

| Section | Key parameters |
|---|---|
| `dataset` | `crop_size: 256`, `resize: 128`, `n_augmentation: 50` |
| `vae` | `latent_dim: 256`, `checkpoint: models/LatentVAE2_256_128.pth` |
| `classifier` | `checkpoint: models/fibermatcher_cls_2.pth`, `batch_size: 256`, `lr: 0.0001` |
| `cellpose` | `model_path: (external, set in config)`, `flow_threshold: 0.4` |
| `matching` | `n_initial_guess: 80`, `distance_neighbors_ref: 200`, `min_cls_logit: 0.5` |

## Project Structure

```
/DATA/F2FMatcher/
├── src/f2fmatcher/
│   ├── config.py              # YAML config loader
│   ├── cli.py                 # CLI entry point
│   ├── io/                    # CZI reader, pixel sizes
│   ├── segmentation/          # Cellpose wrapper, augmentation
│   ├── vae/                   # VAE model, dataset, training, embedding
│   ├── classifier/            # PairClassifier, dataset, training
│   ├── matching/              # Spatial signatures, cost matrix, propagation
│   ├── analysis/              # Staining quantification
│   ├── visualization/         # Prediction plots
│   └── utils/                 # Seed, IO helpers
├── scripts/                   # Runnable CLIs
├── configs/                   # YAML config files
├── tests/                     # Unit tests
├── notebooks/                 # Annotation QA notebooks
└── pyproject.toml             # Package definition
```

## Data Format

| Stage | Format | Content |
|---|---|---|
| Raw input | `.czi` or `.png` | Full histological slide (Zeiss AxioScan) |
| Cellpose masks | `.pkl` | Tuple: (masks, flows, styles) |
| ROI crops | `.npz` | `flow_x`, `flow_y`, `cell_prob`, `roi_mask` (256×256) |
| VAE embeddings | `.npy` | 256-dim float32 vector per ROI |
| Matching output | `.pkl` | List of `(label_img1, label_img2)` matched pairs |

## Matching Performance (typical)

| Metric | Value |
|---|---|
| ROIs per image | 2,000 – 3,500 |
| Matched pairs per pair | 1,900 – 2,800 |
| Coverage | 50 – 95% |
| Propagation steps | 10 – 20 |
| Initial seeds | 12 – 80 selected from 80 guesses |
| Classifier val F1 | ≈ 0.944 |

## Video

![Matching animation](video/comparison.gif)

Full video: [comparison.mp4](video/comparison.mp4)

*The matching process in action. Cellpose segmentations are shown as fiber contours; pairs of fibers predicted to correspond across the two stains are linked by matching labels and colored overlays. The algorithm starts from a small set of geometrically consistent seed pairs and iteratively propagates matches to neighboring fibers, converging to cover most fibers in the section.*

## License

MIT License (see `LICENSE`).
