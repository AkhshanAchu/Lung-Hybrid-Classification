"""
valid_hybrid_classifier.py
===========================
Validation / inference script for any trained classifier.
Computes and saves all required metrics:
  Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC
  Confusion Matrix, ROC Curves, Bar chart, Inference time histogram

Usage:
    python valid_hybrid_classifier.py
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils.data_create import get_train_val_dataloaders
from models.attention_classifier import HybridAttentionClassifier
from utils.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
    plot_training_curves,
    save_all_plots,
)

# ──────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset"
MODEL_PATH   = "checkpoint_hybrid/hybrid_convnext_best.pth"
HISTORY_PATH = "checkpoint_hybrid/hybrid_convnext_best_history.json"
SAVE_DIR     = "results_hybrid_validation"
BATCH_SIZE   = 8
IMAGE_SIZE   = (256, 256)
NUM_CLASSES  = 4
CLASS_NAMES  = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
# ──────────────────────────────────────────────────────────────────


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────────
    print("Loading validation data …")
    _, val_loader = get_train_val_dataloaders(
        root_dir   = DATASET_PATH,
        batch_size = BATCH_SIZE,
        val_split  = 0.2,
        image_size = IMAGE_SIZE,
        num_workers= 0,
    )

    # ── Load model ────────────────────────────────────────────────
    print("Loading model …")
    model = HybridAttentionClassifier(
        num_classes    = NUM_CLASSES,
        input_channels = 6,
        img_size       = IMAGE_SIZE[0],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    # ── Inference ─────────────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []
    inference_times = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)

            t0 = time.time()
            logits = model(images)
            t1 = time.time()

            per_img = (t1 - t0) / images.size(0) * 1000  # ms
            inference_times.extend([per_img] * images.size(0))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── Metrics ───────────────────────────────────────────────────
    m = compute_metrics(y_true, y_pred, y_prob, class_names=CLASS_NAMES)
    print_metrics(m)

    inf_times = np.array(inference_times)
    print(f"\n  Inference time (ms/image):  "
          f"avg={inf_times.mean():.2f}  "
          f"min={inf_times.min():.2f}  "
          f"max={inf_times.max():.2f}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_confusion_matrix(
        m['confusion_matrix'], CLASS_NAMES,
        save_path=f"{SAVE_DIR}/confusion_matrix.png",
        title="HybridAttentionClassifier – Confusion Matrix")

    plot_roc_curves(
        y_true, y_prob, CLASS_NAMES,
        save_path=f"{SAVE_DIR}/roc_curves.png",
        title="HybridAttentionClassifier – ROC Curves")

    # Training curves (if history file exists)
    if os.path.exists(HISTORY_PATH):
        import json
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        plot_training_curves(
            history,
            save_path=f"{SAVE_DIR}/training_curves.png",
            title_prefix="HybridConvNeXt")

    # Inference time distribution
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(inf_times, bins=30, color='steelblue', alpha=0.75)
    ax.set_xlabel("Inference Time per Image (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Time Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{SAVE_DIR}/inference_times.png", dpi=150)
    plt.close(fig)

    print(f"\n  All plots saved to '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
