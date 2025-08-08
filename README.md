# Visual Anomaly Detection of Intake Module Pre-filters

[![Thesis](https://img.shields.io/badge/thesis-available-blue)](https://www.vut.cz/en/students/final-thesis/detail/165885)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

> A production-ready pipeline for detecting visual defects on automotive plastic components using unsupervised deep learning

This repository supports my bachelor's thesis "Visual Anomaly Detection of Intake Module Pre-filters Using Unsupervised Deep Learning" from Brno University of Technology. The work presents an industrial-grade solution for automated quality inspection of automotive pre-filter components.

**🎯 Key Achievement:** AUROC up to 0.92 for image-level anomaly detection using only normal samples for training.

## 🔍 Overview

The project tackles a real industrial challenge: detecting subtle visual defects (scratches, burns, missing geometry) on glossy automotive plastic parts where defective samples are scarce. Two deep learning approaches are implemented and compared:

- **Baseline Autoencoder**: RGB reconstruction with attention mechanisms
- **Deep Feature Reconstruction (DFR)**: Feature-space reconstruction using EfficientNet-B6 backbone

Both models are trained exclusively on normal (OK) images and evaluated on both image-level classification (OK/NOK) and pixel-level anomaly localization.

## 📁 Contents

- [🔬 Problem & Data](#-problem--data)
- [🧠 Methods](#-methods)
- [📊 Results](#-results)
- [🛠️ Reproduce Experiments](#️-reproduce-experiments)
- [🎮 Inference Demo](#-inference-demo)
- [📂 Dataset Structure](#-dataset-structure)
- [⚙️ Implementation Details](#️-implementation-details)
- [🗺️ Roadmap](#️-roadmap)
- [📄 Citation](#-citation)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PyTorch with CUDA support (recommended)
- GPU with ≥11GB VRAM (for DFR models)

## 🔬 Problem & Data

### Target Component
- **Part**: Automotive plastic intake module pre-filter
- **Challenge**: Glossy "top show surface" susceptible to scratches and burns
- **Constraint**: Limited defective samples → unsupervised learning approach

<img width="1159" height="423" alt="image" src="https://github.com/user-attachments/assets/e5a7b42a-f38b-415d-940b-168e3122ce76" />


### Experimental Setup
- **Imaging Environment**: 80×80×80cm reflective lightbox with dual LED strips
- **Camera**: DFK Z12GX236 (Sony IMX236LQ sensor, 1920×1200 → 1024×1024 crop)
- **Lighting**: 5500K LEDs with bidirectional setup for optimal scratch detection

**📸 ADD IMAGE: Figure 18 - Experimental setup for image acquisition**

### Dataset Statistics
| Split | Parts | Images per Part | Total Images |
|-------|-------|----------------|--------------|
| OK (Train/Test) | 450 | 5 zooms × 5 orientations = 25 | ~11,250 |
| NOK (Test only) | 35 | 5 zooms × 7 orientations = 35 | ~1,225 |

### Data Processing Pipeline
- **Enhancement** (test set + 15% train):
  - CLAHE on L channel (LAB color space)
  - Gamma correction (γ=0.8)
  - Unsharp masking for edge enhancement

**📸 ADD IMAGE: Figure 23 - Enhancement effect comparison (before/after)**

- **Augmentation** (training):
  - Photometric: gamma [0.7,1.25], brightness [0.8,1.05], contrast [0.8,1.2]
  - Geometric: H/V flips (40%), rotations 90°/180°/270° (15%)
  - Noise: Gaussian noise (3 levels, 30%), Gaussian blur (20%)

## 🧠 Methods

### 1️⃣ Baseline Autoencoder with Attention
Input (512×512×3) → Encoder → Bottleneck → Decoder → Output (512×512×3)
↓ ↑
Skip Connection + CBAM Attention


- **Architecture**: Symmetric conv autoencoder with mid-level skip connection
- **Attention**: CBAM (Channel + Spatial) in decoder stages  
- **Loss**: Composite 0.15×MSE + 0.85×SSIM
- **Training**: AdamW, 25 epochs, batch size 16
- **Result**: Poor generalization, useful as negative control

### 2️⃣ Deep Feature Reconstruction (DFR) ⭐
Input → EfficientNet-B6 → Multi-scale Features → Feature Autoencoder → Anomaly Score
(6 layers) (832 channels) (1×1 convolutions)


- **Backbone**: EfficientNet-B6 (ImageNet pretrained)
- **Features**: Extract from 6 network stages, upsample and concatenate (832 channels)
- **Autoencoder**: Lightweight 1×1 conv bottleneck (BN + ReLU)
- **Latent Dimension**: PCA-estimated to retain 90% variance (~210-260)
- **Training**: MSE loss, Adam(1e-3), 150 epochs, batch size 4

**Two DFR Variants:**
- **Model-128**: 128×128 feature maps → Better pixel-level AUROC (noise suppression)
- **Model-170**: 170×170 feature maps → Better image-level AUROC (fine detail sensitivity)

## 📊 Results

### Performance Comparison

| Model | Dataset | Image AUROC | Pixel AUROC | Accuracy |
|-------|---------|-------------|-------------|----------|
| Baseline AE | A | 0.61 | 0.85* | 59% |
| Baseline AE | B | 0.63 | 0.58 | 57% |
| **DFR-128** | A | **0.86** | **0.89** | 77% |
| **DFR-128** | B | **0.90** | **0.82** | 81% |
| **DFR-170** | A | **0.86** | **0.88** | 77% |
| **DFR-170** | B | **0.92** | **0.79** | 84% |

*Artificially inflated by background reconstruction

**📸 ADD IMAGE: Figure 31 - ROC curve of best model (170-B)**

### Qualitative Results

**📸 ADD IMAGE: Figure 32 - DFR outputs for Dataset-A (original, heatmap, ground truth, prediction)**
**📸 ADD IMAGE: Figure 33 - DFR outputs for Dataset-B (original, heatmap, ground truth, prediction)**

**🏆 Production Choice**: DFR Model-170 on Dataset-B achieves 0.92 image-level AUROC, optimal for industrial OK/NOK classification.

## 🛠️ Reproduce Experiments

### 1. Prepare Dataset
Structure your data according to the [Dataset Structure](#-dataset-structure) section below.

### 2. Train Baseline Autoencoder
Configure paths in the script
python Baseline_AE_with_attention.py


### 3. Train DFR Model
Step 1: Verify feature extraction
python Feature_extraction_and_aggregation.py

Step 2: Estimate latent dimension
python PCA_Latent_dimension_estimation.py

Step 3: Train the model (set c_l from PCA results)
python Feature_AE_final.py


### 4. Evaluate Models
The scripts include evaluation code with AUROC computation, confusion matrices, and pixel-level metrics.

## 🎮 Inference Demo

Run real-time inference with the best model:
python Final_implementation_of_Real-time_inference.py


**Outputs:**
- OK/NOK classification with confidence
- Anomaly heatmap visualization  
- Thresholded binary mask overlay

**📸 ADD IMAGE: Figure 34 - Final implementation (OK part example)**
**📸 ADD IMAGE: Figure 35 - Final implementation (NOK part with overlay)**

⚠️ **Note**: 20px border cropping applied to suppress padding artifacts - keep critical regions away from image edges.

## 📂 Dataset Structure

data/
├── train/
│ └── ok/ # Normal images only (unsupervised)
├── test/
│ ├── ok/ # Normal test images
│ └── nok/ # Anomalous test images
└── masks/
└── nok/ # Binary segmentation masks (same basenames as test/nok)

**Dataset Splits:**
- **Dataset-A**: zoom_0, zoom_1 (wide FOV, full part visible)
- **Dataset-B**: zoom_3, zoom_4, zoom_5 (tight FOV, central region focus)

**File Naming**: `zoom_X_timestamp.jpg` where X ∈ {0,1,2,3,4,5}

## ⚙️ Implementation Details

### Hardware Requirements
- **GPU**: NVIDIA GTX 1080 Ti (11GB VRAM) or equivalent
- **RAM**: 32GB recommended for large feature tensors
- **Storage**: ~15GB for full dataset

### Key Parameters
Baseline AE
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 25
BATCH_SIZE = 16
LOSS_WEIGHTS = [0.15, 0.85] # [MSE, SSIM]

DFR Model
LEARNING_RATE = 1e-3
EPOCHS = 150
BATCH_SIZE = 4
LATENT_DIM = # [Dataset-A, Dataset-B]
TOP_K = 20 # anomaly scoring


### Performance Tips
- Use mixed precision training for memory efficiency
- Crop 20px borders during evaluation to reduce artifacts
- Apply enhancement selectively (test + 15% train) to avoid overfitting

## 🗺️ Roadmap

- [ ] **Border Artifact Reduction**: Improve receptive field handling
- [ ] **Multi-view Fusion**: Combine 2-3 controlled poses instead of random orientations  
- [ ] **Edge Deployment**: ONNX/TensorRT export for industrial PCs
- [ ] **Lightweight Backbones**: Evaluate TinyAD/MobileNet alternatives
- [ ] **Comparative Study**: Integrate PatchCore, PaDiM, Student-Teacher methods

## 📄 Citation

If this work helps your research or industrial application, please cite:

@thesis{hruska2025visual,
title={Visual Anomaly Detection of Intake Module Pre-filters Using Unsupervised Deep Learning},
author={Hruška, Josef},
year={2025},
school={Brno University of Technology, Faculty of Mechanical Engineering},
url={https://www.vut.cz/en/students/final-thesis/detail/165885}
}


## 📋 Repository Files

Core scripts from thesis appendix:

| File | Description |
|------|-------------|
| `Augmentation.py` | Data augmentation pipeline |
| `Baseline_AE_with_attention.py` | Baseline autoencoder with CBAM |
| `Feature_AE_final.py` | DFR model (main contribution) |
| `Feature_extraction_and_aggregation.py` | EfficientNet-B6 feature extraction |
| `Final_implementation_of_Real-time_inference.py` | Production inference demo |
| `PCA_Latent_dimension_estimation.py` | Latent space dimensionality via PCA |
| `Test_enhancement.py` | Image enhancement pipeline |
| `SSIM_loss_class.py` | Structural similarity loss function |

## 🤝 Contributing

This repository primarily serves as a research artifact. For industrial applications or extensions, please reach out through the thesis supervisor or university contact.

---

**⚡ Built with PyTorch • 🏭 Industry-tested • 🎓 Academic research**

*Developed at Brno University of Technology in collaboration with MANN+HUMMEL*
