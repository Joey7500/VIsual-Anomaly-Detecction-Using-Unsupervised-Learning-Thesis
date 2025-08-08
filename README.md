# This reopsitory serves as a support to my thesis "Visual Anomaly Detection of Intake Module Prefilters Using Unsupervised Learning"

https://www.vut.cz/en/students/final-thesis/detail/165885

Further is a brief summary of the project

A lightweight, production‑oriented pipeline for detecting visual defects (scratches, burns, missing geometry) on automotive plastic pre‑filters using unsupervised deep learning. The project includes two approaches:

A baseline image autoencoder with attention (reconstruction in RGB).

A Deep Feature Reconstruction (DFR) model that reconstructs EfficientNet-B6 features for robust anomaly detection and localization.

Both models are trained on only normal (OK) images and evaluated on both image-level (OK/NOK) and pixel-level segmentation. The best model achieves AUROC up to 0.92 on image classification and up to 0.89 at pixel level on in‑house datasets collected under controlled lighting.

Note: This README summarizes the thesis and repository content and is structured for quick adoption.


Contents
What’s included

Problem and data

Methods

Results

Quickstart

Reproduce experiments

Inference demo

Dataset structure

Implementation details

Roadmap

License and citation

What’s included
Two unsupervised models:

Baseline Autoencoder with CBAM attention and skip connection.

Deep Feature Reconstruction Autoencoder with EfficientNet-B6 backbone.

End‑to‑end scripts:

Data augmentation, enhancement, PCA latent estimation.

Training and evaluation at image and pixel levels.

Real‑time inference demo.

Annotations:

Pixel‑level masks for anomalous (NOK) images, created with CVAT.

Files are documented in the thesis and mirrored here for reproducibility.

Problem and data
Target part: automotive plastic intake module pre‑filter (smooth, glossy “top show surface” susceptible to scratches).

Industrial constraint: few defective samples; unsupervised approach preferred.

Imaging setup:

80×80×80cm reflective lightbox with dual LED bars, top‑mounted camera.

Camera: DFK Z12GX236 (Sony IMX236LQ, 1920×1200, cropped to 1024×1024), variable focal length with 5 zooms per capture.

Acquisition:

OK parts: 450 parts × 5 zooms × 5 orientations ≈ 11,250 images.

NOK parts: 35 parts × 5 zooms × 7 orientations ≈ 1,225 images.

Two evaluation datasets:

Dataset‑A: wide FOV (zoom_0, zoom_1), whole part visible or slightly cropped.

Dataset‑B: tighter FOV (zoom_3 to zoom_5), focused on central glossy region.

Enhancement (for test set and 15% of train for robustness):

CLAHE on L channel (LAB).

Gamma correction (γ=0.8).

Unsharp masking.

Augmentation (train):

Photometric: gamma0.7,1.25, brightness0.8,1.05, contrast0.8,1.2, Gaussian noise (3 levels, p=0.3), Gaussian blur (k=3/4/5, p=0.2).

Geometric: H/V flips (p=0.4), rotations (90/180/270°, p=0.15).

Fallback: ensure at least minimal noise so every augmented image is unique.

Methods
1) Baseline Autoencoder with Attention (RGB reconstruction)
Symmetric conv autoencoder with one mid‑level skip.

Decoder stages include CBAM (channel+spatial attention).

Loss: composite 0.15 MSE + 0.85 SSIM; AdamW; 25 epochs; batch 16.

Decision function: mean of top‑k pixelwise reconstruction errors, k=1,500.

Result: struggled to generalize; good at reconstructing background; mislabeled large part area as anomalous under certain views. Useful as negative control.

2) Deep Feature Reconstruction (DFR)
Backbone: EfficientNet‑B6 (pretrained on ImageNet).

Extract multi‑scale features from 6 stages; upsample to 512×512; average pool to 170×170 (stride 3) or 128×128 (stride 4); concatenate along channels (832C).

Autoencoder: 1×1 conv bottleneck architecture (encoder/decoder; BN+ReLU), chosen for efficiency and stability; larger kernels or CBAM did not improve performance and increased memory.

Latent size: determined via PCA to preserve ≈90% variance (≈210 for Dataset‑A, ≈260 for Dataset‑B).

Loss: MSE; Adam(lr=1e‑3); 150 epochs; batch 4.

Decision function: crop 20px border of heatmap to suppress boundary artefacts; use mean of top‑k anomalies, k=20.

Two DFR variants:

Model‑128: feature map 128×128.

Model‑170: feature map 170×170.

Trade‑off observed:

Model‑170: better image‑level AUROC (more sensitive to small/subtle anomalies).

Model‑128: better pixel‑level AUROC (spatial smoothing suppresses noise).

Results
Image‑level (OK/NOK) and pixel‑level (AUROC on masks). Best configurations:

Baseline AE:

Dataset‑A: AUROC 0.61 (image), 0.85 (pixel; artificially inflated by background).

Dataset‑B: AUROC 0.63 (image), 0.58 (pixel).

DFR, Model‑128:

Dataset‑A: AUROC 0.86 (image), 0.89 (pixel).

Dataset‑B: AUROC 0.90 (image), 0.82 (pixel).

DFR, Model‑170:

Dataset‑A: AUROC 0.86 (image), 0.88 (pixel).

Dataset‑B: AUROC 0.92 (image), 0.79 (pixel).

Chosen for implementation: DFR Model‑170 on Dataset‑B (170‑B) for highest image‑level AUROC (0.92), aligning with industrial priority of OK/NOK classification. Note: ~10% crop at borders to remove padding artefacts; may miss border defects—operational mitigations suggested in Roadmap.

Quickstart
Prereqs:

Python 3.9+

PyTorch with CUDA (recommended)

Install dependencies from each script or consolidate into requirements.txt

Clone:

See repository link at the top of this page. If using this repo structure, place the scripts from the “Attachment” section (thesis appendix) into the repo root.

Recommended GPU: ≥11GB VRAM (DFR uses high‑dimensional feature tensors).

Reproduce experiments
Prepare datasets

Organize OK training, OK test, NOK test as in “Dataset structure” below.

Ensure 1024×1024 crops exist. Scripts will resize to 512×512 at runtime.

Train Baseline AE

Configure paths in Baseline_AE_with_attention.py.

Run training; weights saved automatically.

Train DFR

Run Feature_extraction_and_aggregation.py to verify EfficientNet-B6 hooks and aggregated tensors (128×128 and/or 170×170).

Estimate latent dim with PCA_Latent_dimension_estimation.py; set c_l accordingly in Feature_AE_final.py.

Train Feature_AE_final.py for Dataset‑A and Dataset‑B; log train/val loss (val includes NOK to confirm higher error on anomalies—no backprop).

Evaluate

Use the same testing splits and decision functions as in the scripts for AUROC, confusion matrices, and pixel AUROC.

Inference demo
Script: Final_implementation_of_Real-time_inference.py

Loads the best DFR model (170‑B) and runs real‑time inference from the camera feed in the lightbox setup.

Outputs:

OK/NOK decision with confidence.

Anomaly heatmap (upsampled), and thresholded mask overlay.

Note: Borders cropped internally to suppress padding artefacts; adjust framing to keep critical ROI away from image edges.

Dataset structure
Recommended on‑disk layout:

data/

train/

ok/ # only normal images used for training

test/

ok/

nok/

masks/

nok/ # binary masks aligned to test/nok images (same basename)

By zoom/FOV configuration, create Dataset‑A and Dataset‑B splits:

Dataset‑A: zoom_0, zoom_1

Dataset‑B: zoom_3, zoom_4, zoom_5

File naming convention used in acquisition:

zoom_0 … zoom_5 per orientation capture, with timestamped basenames.

Implementation details
Enhancement pipeline: CLAHE (LAB L‑channel) → gamma(0.8) → unsharp masking; applied to all test images and 15% of train to improve robustness.

Augmentations (train): photometric + geometric; avoid overexposure; rotations limited to multiples of 90° to avoid interpolation artefacts.

Baseline AE:

4× downsampling stages; 1 skip; CBAM in decoder; sigmoid output in.

Loss: 0.15 MSE + 0.85 SSIM; AdamW(1e‑3, weight_decay=1e‑5); 25 epochs; batch 16.

DFR:

EfficientNet‑B6 feature hooks (6 layers), upsample→pool→concat to [B,832,H’,W’].

Autoencoder with 1×1 convs (BN+ReLU) enc/dec; latent c_l from PCA (90% variance).

Loss: MSE; Adam(1e‑3); 150 epochs; batch 4; mixed precision advised if memory‑limited.

Scoring: crop 20px borders; top‑k mean (k=20) on residual heatmap.

Variants:

Model‑128: better pixel AUROC (noise suppression).

Model‑170: better image AUROC (sensitivity to small anomalies).

Hardware used: NVIDIA GTX 1080 Ti (11GB), 32GB RAM.

Roadmap
Reduce border loss:

Expand backbone receptive field handling to avoid border pooling artefacts.

Add safe margin in staging fixture to keep critical ROI away from edges.

Multi‑view fusion:

Fuse predictions across 2–3 controlled poses instead of many random orientations; improves coverage without heavy data diversity.

Edge deployment:

Export to ONNX/TensorRT; profile on industrial PC.

Evaluate TinyAD backbones for reduced footprint with STFPM‑like approach.

Alternative methods for comparison:

PatchCore/KNN memory bank, Padim, Student‑Teacher variants; integrate a common eval harness.

License and citation
Please follow the license in this repository.

If this work helps your research or deployment, please cite the thesis:

Hruška, J. Visual Anomaly Detection of Intake Module Pre‑filters Using Unsupervised Deep Learning. Brno University of Technology, Faculty of Mechanical Engineering, Institute of Automation and Computer Science, 2025.

Visuals
Below are representative visuals from the thesis to guide usage and expectations.

Experimental lightbox + camera setup (top mount; dual LED bars):

Enhancement effect (CLAHE + gamma + unsharp) on subtle scratch:

DFR outputs (heatmap and predicted mask) for Dataset‑A example:

DFR outputs (heatmap and predicted mask) for Dataset‑B example:

ROC curve (DFR 170‑B):

[images:

Images illustrate qualitative behavior; border cropping is applied during scoring to mitigate upsampling/padding artefacts.

Repository pointers
The thesis appendix lists the core scripts (place them under repo root or src/):

Augmentation.py — data augmentation code.

Baseline_AE_with_attention.py — baseline autoencoder.

Baseline_wo_attention.py — simplified baseline.

Feature_AE_final.py — DFR autoencoder (final).

Feature_AE_with_attention_and_skips.py — alternative DFR variant (not used).

Feature_extraction_and_aggregation.py — EfficientNet-B6 feature hooks + pooling/concat.

Final_implementation_of_Real-time_inference.py — real‑time demo.

PCA_Latent_dimension_estimation.py — latent size via PCA to retain ≈90% variance.

SSIM_loss_class.py — SSIM for baseline AE.

Test_enhancement.py — CLAHE/gamma/unsharp test enhancement.


