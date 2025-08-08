# Visual Anomaly Detection of Intake Module Prefilters Using Unsupervised Learning

This repository supports my bachelor’s thesis:  
**"Visual Anomaly Detection of Intake Module Prefilters Using Unsupervised Learning"**  
📄 Thesis link: [VUT Final Thesis Portal](https://www.vut.cz/en/students/final-thesis/detail/165885)

---

## Overview

A **lightweight, production-ready pipeline** for detecting visual defects such as scratches, burns, and missing geometry on automotive plastic pre-filters using **unsupervised deep learning**.

Two main approaches are implemented:

1. **Baseline Autoencoder with Attention** – RGB reconstruction.
2. **Deep Feature Reconstruction (DFR)** – reconstructs EfficientNet-B6 features for robust anomaly detection and localization.

Both models are trained **only on normal (OK) images** and evaluated on both **image-level classification (OK/NOK)** and **pixel-level segmentation**.

The best model achieves:
- **Image AUROC:** up to 0.92  
- **Pixel AUROC:** up to 0.89  
on in-house datasets collected under controlled lighting.

---

## Contents
- [Problem and Data](#problem-and-data)
- [Methods](#methods)
- [Results](#results)
- [Quickstart](#quickstart)
- [Reproduce Experiments](#reproduce-experiments)
- [Inference Demo](#inference-demo)
- [Dataset Structure](#dataset-structure)
- [Implementation Details](#implementation-details)
- [Roadmap](#roadmap)
- [License and Citation](#license-and-citation)
- [Visuals](#visuals)

---

## Problem and Data

Target part: automotive plastic **intake module pre-filter** with a smooth, glossy “top show surface” prone to visible scratches.

**Industrial constraints:**
- Very few defective samples available → unsupervised approach is ideal.

**Imaging setup:**
- 80×80×80 cm reflective lightbox with dual LED bars.
- Camera: DFK Z12GX236 (Sony IMX236LQ, 1920×1200, cropped to 1024×1024).
- 5 zoom levels per capture.

📸 **IMAGE HERE:** *Photo of the lightbox setup from thesis.*

**Data acquisition:**
- OK: ~11,250 images (450 parts × 5 zooms × 5 orientations).
- NOK: ~1,225 images (35 parts × 5 zooms × 7 orientations).

Two datasets:
- **Dataset-A:** Wide FOV (zoom_0, zoom_1).
- **Dataset-B:** Tight FOV (zoom_3 to zoom_5).

**Enhancement pipeline** (test set + 15% of train):
- CLAHE on L channel (LAB)
- Gamma correction (γ=0.8)
- Unsharp masking

📸 **IMAGE HERE:** *Example of enhancement effect (before/after).*

---

## Methods

### 1. Baseline Autoencoder with Attention
- Symmetric convolutional autoencoder with one mid-level skip.
- CBAM (channel + spatial attention) in decoder.
- Loss: 0.15 MSE + 0.85 SSIM.

**Decision function:** Mean of top-k pixel errors (k=1,500).  
**Outcome:** Good at background reconstruction, but sometimes flagged large normal areas as anomalous.

---

### 2. Deep Feature Reconstruction (DFR)
- Backbone: EfficientNet-B6 (ImageNet pretrained).
- Extract & aggregate features from 6 stages → pooled to 128×128 or 170×170.
- 1×1 convolutional bottleneck AE.
- Latent size via PCA (≈90% variance retained).

**Two variants:**
- **Model-128:** Better pixel-level AUROC (noise suppression).
- **Model-170:** Better image-level AUROC (detects small anomalies).

📸 **IMAGE HERE:** *Example heatmap + mask output for Dataset-A.*  
📸 **IMAGE HERE:** *Example heatmap + mask output for Dataset-B.*

---

## Results

| Model               | Dataset   | Image AUROC | Pixel AUROC |
|--------------------|-----------|-------------|-------------|
| Baseline AE        | A         | 0.61        | 0.85*       |
| Baseline AE        | B         | 0.63        | 0.58        |
| DFR Model-128      | A         | 0.86        | **0.89**    |
| DFR Model-128      | B         | 0.90        | 0.82        |
| DFR Model-170      | A         | 0.86        | 0.88        |
| **DFR Model-170**  | **B**     | **0.92**    | 0.79        |

> *Baseline AE pixel AUROC on Dataset-A is artificially inflated by background.

📈 **IMAGE HERE:** *ROC curve for DFR 170-B.*

---

## Quickstart

**Requirements:**
- Python 3.9+
- PyTorch + CUDA
- GPU ≥ 11GB VRAM recommended (DFR feature tensors are large)

---

## 🚀 Quickstart

### 1. Requirements
- Python **3.9+**
- [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support (recommended)
- GPU with **≥ 11 GB VRAM** for DFR models (feature tensors are large)

```bash
# Clone the repository
git clone <repo_url>
cd <repo>

# Install dependencies
pip install -r requirements.txt


data/
  train/
    ok/
  test/
    ok/
    nok/
    masks/
      nok/


python Baseline_AE_with_attention.py \
    --train_dir data/train/ok \
    --test_ok_dir data/test/ok \
    --test_nok_dir data/test/nok \
    --masks_dir data/test/masks/nok \
    --epochs 25 \
    --batch_size 16


python Feature_extraction_and_aggregation.py \
    --dataset Dataset-A \
    --output_dir features/

