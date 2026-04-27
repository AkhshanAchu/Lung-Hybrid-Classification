# LungClassification

A hybrid deep learning pipeline for classifying chest X-rays into four categories: **COVID-19**, **Lung Opacity**, **Normal**, and **Viral Pneumonia**. The system combines attention-augmented segmentation (U-Net), pretrained CNN classifiers (ConvNeXt-Tiny and ResNet-50), and bio-inspired feature selection (Firefly Algorithm and PSO) into a multi-stage architecture.

**Dataset**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Architecture](#architecture)
  - [Segmentation: AttentionHybridUNet](#segmentation-attentionhybridunet)
  - [Classification: HybridAttentionClassifier](#classification-hybridattentionclassifier)
  - [Feature Selection: Firefly and PSO](#feature-selection-firefly-and-pso)
- [Entry Points](#entry-points)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Dependencies](#dependencies)

---

## Overview

The project addresses chest X-ray classification as a two-stage problem. First, a segmentation model isolates the lung region from each image. Second, a classification model trained on the concatenated original and masked image (6-channel input) performs multi-class prediction. A separate pipeline extracts deep features from the trained classifier and applies meta-heuristic feature selection (Firefly, PSO) to identify the most discriminative feature subset.

The core contribution is the `MultiScaleHybridAttention` module, which is used in three distinct roles across the pipeline: as skip-connection gates in the segmentation network, as mid-stage refinements inside the ConvNeXt and ResNet classifiers, and as a standalone feature refinement mechanism.

---

## Repository Structure

```
LungClassification-master/
│
├── main_hybrid_unet.py          # Train and compare segmentation models
├── main_hybrid_classifier.py    # Train and compare classification models
├── main_hybrid_firefly.py       # Feature selection comparison pipeline
├── mask_create.py               # Run segmentation to generate masked images
├── valid_classifier.py          # Validate a plain classifier
├── valid_hybrid_classifier.py   # Validate a hybrid attention classifier
├── valid_segment.py             # Validate the segmentation model
│
├── best_feature_mask.json       # Saved binary feature mask (JSON)
├── best_feature_mask.npy        # Saved binary feature mask (NumPy)
│
├── models/
│   ├── attention.py             # MultiScaleHybridAttention core module
│   ├── attention_classifier.py  # HybridAttentionClassifier, ResNetHybridClassifier
│   ├── attention_unet.py        # AttentionHybridUNet
│   ├── classifier.py            # MLPClassifier, BetterMLP
│   └── unet.py                  # Classic AttU_Net (baseline)
│
├── preprocess/
│   ├── preprocess.py            # Apply segmentation masks to raw images
│   └── feature.py               # Extract deep features from trained classifier
│
├── train/
│   ├── train_unet.py            # Segmentation training loop
│   ├── train_classifier.py      # Baseline classifier training loop
│   ├── train_classifier_hybrid.py  # Hybrid model training loop
│   ├── firefly.py               # Firefly Algorithm (SVM and MLP variants)
│   ├── feature_selection.py     # PSO and Variance Threshold selectors
│   └── loss.py                  # Custom loss functions
│
├── utils/
│   ├── dataloader.py            # Dataset classes for classification and segmentation
│   ├── data_create.py           # DataLoader factories
│   ├── metrics.py               # Compute and plot all evaluation metrics
│   ├── preprocess.py            # Normalisation utilities
│   └── tools.py                 # Miscellaneous helpers
│
└── inference_results/
    ├── COVID-97.png
    ├── Lung_Opacity-97.png
    ├── Normal-992.png
    └── Viral Pneumonia-91.png
```

---

## Pipeline

The system is divided into three sequential stages. Each stage has its own entry point script.

```
Stage 1: Segmentation
  Input: raw chest X-rays (3-channel RGB)
  Model: AttentionHybridUNet (or classic AttU_Net baseline)
  Output: binary lung masks saved to prd_label/ per class

        ↓

Stage 2: Classification
  Input: 6-channel tensor [original RGB | masked RGB]
  Model: HybridAttentionClassifier (ConvNeXt-Tiny + attention)
       : ResNetHybridClassifier    (ResNet-50 + attention)
  Output: 4-class softmax prediction + checkpoints + plots

        ↓

Stage 3: Feature Selection
  Input: deep features extracted from trained classifier backbone
  Selectors: Variance Threshold (baseline)
             Firefly Algorithm + SVM / MLP
             PSO + SVM / MLP
  Output: binary feature mask, per-method accuracy table
```

### Flow of Execution

**Step 1 — Prepare masks**

Run `mask_create.py` to apply a pretrained `AttU_Net` to the raw dataset. This generates `prd_label/` folders inside each class directory containing the masked images.

```bash
python mask_create.py
```

**Step 2 — Train segmentation (optional, to retrain from scratch)**

`main_hybrid_unet.py` trains both the new `AttentionHybridUNet` and the classic `AttU_Net` side by side, reporting IoU and Dice.

```bash
python main_hybrid_unet.py
```

**Step 3 — Train classifiers**

`main_hybrid_classifier.py` runs the full classification pipeline: loads the 6-channel dataset, trains `HybridAttentionClassifier` (ConvNeXt backbone) and `ResNetHybridClassifier` (ResNet-50 backbone), evaluates both, and saves checkpoints plus all diagnostic plots.

```bash
python main_hybrid_classifier.py
```

**Step 4 — Feature selection comparison**

`main_hybrid_firefly.py` loads the trained ConvNeXt classifier, extracts penultimate-layer features, and benchmarks five selection strategies: Variance Threshold, Firefly+SVM, Firefly+MLP, PSO+SVM, PSO+MLP. Produces a comparison bar chart and summary table.

```bash
python main_hybrid_firefly.py
```

**Step 5 — Validation**

Use `valid_hybrid_classifier.py` or `valid_classifier.py` to run inference on a held-out split and regenerate metrics without retraining.

```bash
python valid_hybrid_classifier.py
```

---

## Architecture

### Segmentation: AttentionHybridUNet

The segmentation backbone is a standard encoder-decoder U-Net modified in two ways. The bottleneck applies `MultiScaleHybridAttention` to refine the deepest feature map before decoding begins. Each skip connection is passed through a `HybridSkipGate` rather than the classic additive attention block, allowing the decoder query to modulate skip features using the full hybrid attention mechanism.

```
Input (3, H, W)
    │
Encoder
    Enc1 → 64 ch, H×W
    Enc2 → 128 ch, H/2×W/2
    Enc3 → 256 ch, H/4×W/4
    Enc4 → 512 ch, H/8×W/8
         ↓
    Bottleneck 1024 ch, H/16×W/16
         │
    MultiScaleHybridAttention  ←── bottleneck refinement
         ↓
Decoder (with HybridSkipGate at each level)
    Dec4 (+ gated Enc4 skip)
    Dec3 (+ gated Enc3 skip)
    Dec2 (+ gated Enc2 skip)
    Dec1 (+ gated Enc1 skip)
         ↓
    Conv 1×1 → output mask (1 ch)
```

The baseline model (`AttU_Net` in `models/unet.py`) uses standard attention gates and is trained in parallel for comparison.

---

### Classification: HybridAttentionClassifier

Each X-ray is paired with its lung mask. The original RGB image and the mask-applied version are concatenated channel-wise to form a 6-channel input. Two backbone variants are provided.

**HybridAttentionClassifier (ConvNeXt-Tiny backbone)**

The first convolutional layer of ConvNeXt-Tiny is patched to accept 6 input channels. `MultiScaleHybridAttention` is injected after stage 3 (192 channels, 32×32 spatial) and stage 5 (384 channels, 16×16 spatial). A two-layer MLP head with LayerNorm, GELU, and dropout follows global average pooling.

```
6-ch Input (6, 256, 256)
    │
ConvNeXt stem → stage1 → stage2 → stage3
                                      │
                             MultiScaleHybridAttention (192 ch)
                                      │
                               stage4 → stage5
                                           │
                                  MultiScaleHybridAttention (384 ch)
                                           │
                                    stage6 → stage7  (768 ch)
                                           │
                              AdaptiveAvgPool2d → Flatten
                                           │
                              LayerNorm → Linear(768, 256)
                              GELU → Dropout → Linear(256, 4)
                                           │
                                    4-class output
```

**ResNetHybridClassifier (ResNet-50 backbone)**

The same 6-channel input patch is applied to the first conv of ResNet-50. `MultiScaleHybridAttention` is applied to the output of `layer3` (1024 channels, 16×16 spatial). The MLP head uses ReLU and dropout before the final linear layer.

---

### MultiScaleHybridAttention

The attention module (`models/attention.py`) merges two complementary spatial attention strategies in parallel with learnable fusion weights, plus an optional Squeeze-and-Excitation (SE) gate.

`SelfBlockBranch` computes per-head Q/K/V projections using symmetric convolutional kernels, producing spatially-aware attention maps.

`CrissCrossBranch` uses asymmetric kernel combinations across Q, K, V (criss-cross layout), capturing cross-directional dependencies that symmetric kernels miss.

The two branch outputs are fused via a learnable scalar per branch, then optionally gated by a channel-SE block, and projected back to the original channel count through a 1×1 convolution.
<img width="232" height="150" alt="multiscale_hybrid_attention_architecture" src="https://github.com/user-attachments/assets/4ff5a4bb-3b41-47ff-a995-44ba5eb4a939" />

---

### Feature Selection: Firefly and PSO

After classification training, deep features are extracted from the backbone (global average pooled, 768-dimensional) using `preprocess/feature.py`. Three selector types operate on this feature space.

**Variance Threshold** — removes low-variance features and keeps the top 50% by variance. Fast statistical baseline, no iteration.

**Firefly Algorithm** — a swarm of binary-coded fireflies each representing a feature mask. Fireflies move toward brighter (higher-accuracy) peers via a sigmoid-mapped attraction update. Fitness is evaluated by training a linear SVM or a two-layer MLP on the selected feature subset. Implemented in `train/firefly.py`.

**Particle Swarm Optimisation (PSO)** — each particle holds a real-valued velocity vector. Positions are binarised via sigmoid to yield a feature mask. The swarm updates using inertia, cognitive (personal best), and social (global best) components. The same SVM or MLP evaluator is used. Implemented in `train/feature_selection.py`.

All selectors share the same `.run(X, y)` interface and return a binary mask plus the best observed accuracy, making them interchangeable.

---

## Entry Points

| Script | Purpose |
|---|---|
| `mask_create.py` | Generate lung masks from pretrained AttU_Net |
| `main_hybrid_unet.py` | Train and compare AttentionHybridUNet vs AttU_Net |
| `main_hybrid_classifier.py` | Train HybridConvNeXt and HybridResNet50, full eval |
| `main_hybrid_firefly.py` | Feature selection comparison (5 methods) |
| `valid_hybrid_classifier.py` | Inference + metrics on validation split |
| `valid_classifier.py` | Same for plain classifier |
| `valid_segment.py` | Segmentation metrics on validation split |

---

## Results

Sample inference outputs are stored in `inference_results/`. Each filename encodes the class and the model's confidence.

| File | Class | Confidence |
|---|---|---|
| `COVID-97.png` | COVID-19 | 97% |
| `Lung_Opacity-97.png` | Lung Opacity | 97% |
| `Normal-992.png` | Normal | 99.2% |
| `Viral Pneumonia-91.png` | Viral Pneumonia | 91% |

Evaluation metrics computed across all runs: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC. Diagnostic plots (confusion matrix, ROC curves, training curves) are saved to the configured `SAVE_DIR` during training.

---

## Setup and Usage

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/LungClassification.git
cd LungClassification
pip install -r requirements.txt
```

**2. Download the dataset**

Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle and extract it. The expected structure inside each class folder is:

```
COVID-19_Radiography_Dataset/
    COVID/
        images/
        masks/
    Lung_Opacity/
        images/
        masks/
    Normal/
        images/
        masks/
    Viral Pneumonia/
        images/
        masks/
```

**3. Configure dataset paths**

Edit the `DATASET_PATH` variable at the top of each entry point script to point to your local dataset directory.

**4. Run the pipeline**

```bash
# Step 1: generate masked images (requires a pretrained segmentation checkpoint)
python mask_create.py

# Step 2: train segmentation models
python main_hybrid_unet.py

# Step 3: train classification models
python main_hybrid_classifier.py

# Step 4: run feature selection comparison
python main_hybrid_firefly.py
```

**GPU**: All scripts auto-detect CUDA. Training on CPU is supported but will be slow. A GPU with at least 8 GB VRAM is recommended for the classification step with batch size 8.

---

## Dependencies

```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
Pillow
tqdm
```

Install with:

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn Pillow tqdm
```
