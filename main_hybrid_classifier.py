"""
main_hybrid_classifier.py
==========================
End-to-end pipeline:

Step 1  –  Train HybridAttentionClassifier (ConvNeXt + our attention)
Step 2  –  Train ResNetHybridClassifier    (ResNet-50  + our attention)
Step 3  –  Full evaluation of both models
           • Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
           • Confusion Matrix, ROC Curves
           • Training Curves

Usage:
    python main_hybrid_classifier.py
"""

import os
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.data_create import get_train_val_dataloaders
from models.attention_classifier import (
    HybridAttentionClassifier,
    ResNetHybridClassifier,
)
from train.train_classifier_hybrid import train_hybrid_model
from utils.metrics import (
    compute_metrics, print_metrics, save_all_plots
)

# ──────────────────────────────────────────────────────────────────
# CONFIG  –  edit these paths
# ──────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
SAVE_DIR     = "checkpoint_hybrid"
BATCH_SIZE   = 8
IMAGE_SIZE   = (256, 256)
NUM_CLASSES  = 4
NUM_EPOCHS   = 20
LR           = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES  = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]


def evaluate(model, val_loader, device, class_names, prefix, save_dir):
    """Run inference, collect probs, compute + save all metrics."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    m = compute_metrics(y_true, y_pred, y_prob,
                        class_names=class_names)
    print_metrics(m)
    save_all_plots(m, y_true, y_prob,
                   history=None,          # history passed separately
                   save_dir=save_dir,
                   prefix=prefix)
    return m


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")

    # ── Data ──────────────────────────────────────────────────────
    print("\nLoading data …")
    train_loader, val_loader = get_train_val_dataloaders(
        root_dir   = DATASET_PATH,
        batch_size = BATCH_SIZE,
        val_split  = 0.2,
        image_size = IMAGE_SIZE,
        num_workers= 0,
    )

    results = {}

    # ══════════════════════════════════════════════════════════════
    # MODEL A  –  HybridAttentionClassifier (ConvNeXt backbone)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(" MODEL A  –  HybridAttentionClassifier (ConvNeXt + Attention)")
    print("═"*60)

    model_a = HybridAttentionClassifier(
        num_classes    = NUM_CLASSES,
        input_channels = 6,
        img_size       = IMAGE_SIZE[0],
        num_heads      = 4,
        head_dim       = 32,
    )

    model_a, hist_a = train_hybrid_model(
        model       = model_a,
        train_loader= train_loader,
        val_loader  = val_loader,
        num_epochs  = NUM_EPOCHS,
        lr          = LR,
        device      = DEVICE,
        save_path   = f"{SAVE_DIR}/hybrid_convnext_best.pth",
        model_name  = "HybridConvNeXt",
    )

    # Load best weights before eval
    model_a.load_state_dict(
        torch.load(f"{SAVE_DIR}/hybrid_convnext_best.pth",
                   map_location=DEVICE))
    model_a.to(DEVICE)

    m_a = evaluate(model_a, val_loader, DEVICE,
                   CLASS_NAMES, "HybridConvNeXt", SAVE_DIR)

    # Patch history into plots
    from utils.metrics import plot_training_curves
    plot_training_curves(hist_a,
        save_path=f"{SAVE_DIR}/HybridConvNeXt_training_curves.png",
        title_prefix="HybridConvNeXt")

    results['HybridConvNeXt'] = m_a

    # ══════════════════════════════════════════════════════════════
    # MODEL B  –  ResNetHybridClassifier
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print(" MODEL B  –  ResNetHybridClassifier (ResNet-50 + Attention)")
    print("═"*60)

    model_b = ResNetHybridClassifier(
        num_classes    = NUM_CLASSES,
        input_channels = 6,
        img_size       = IMAGE_SIZE[0],
        num_heads      = 4,
        head_dim       = 32,
    )

    model_b, hist_b = train_hybrid_model(
        model       = model_b,
        train_loader= train_loader,
        val_loader  = val_loader,
        num_epochs  = NUM_EPOCHS,
        lr          = LR,
        device      = DEVICE,
        save_path   = f"{SAVE_DIR}/hybrid_resnet_best.pth",
        model_name  = "HybridResNet50",
    )

    model_b.load_state_dict(
        torch.load(f"{SAVE_DIR}/hybrid_resnet_best.pth",
                   map_location=DEVICE))
    model_b.to(DEVICE)

    m_b = evaluate(model_b, val_loader, DEVICE,
                   CLASS_NAMES, "HybridResNet50", SAVE_DIR)

    plot_training_curves(hist_b,
        save_path=f"{SAVE_DIR}/HybridResNet50_training_curves.png",
        title_prefix="HybridResNet50")

    results['HybridResNet50'] = m_b

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL COMPARISON SUMMARY")
    print("═"*60)
    header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "  N/A "
        print(f"{name:<22} {m['accuracy']:.4f} {m['precision']:.4f} "
              f"{m['recall_sensitivity']:.4f} {m['specificity']:.4f} "
              f"{m['f1_score']:.4f} {auc}")
    print(f"\nAll checkpoints and plots saved to '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
