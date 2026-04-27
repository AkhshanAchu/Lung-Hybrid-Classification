"""
metrics.py  –  Comprehensive Evaluation Suite
================================================
Covers every metric and plot requested:

Classification Metrics:
  Accuracy, Precision, Recall (Sensitivity), Specificity,
  F1-score, ROC-AUC

Visualizations:
  Confusion Matrix, ROC Curve (per-class OvR + macro avg)

Training Curves:
  Accuracy vs Epoch, Loss vs Epoch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize


# ──────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None,
                    class_names=None, average='weighted'):
    """
    Compute all required classification metrics.

    Args:
        y_true      : 1-D array of ground-truth integer labels
        y_pred      : 1-D array of predicted integer labels
        y_prob      : 2-D array [N, C] of class probabilities (for AUC)
        class_names : list of human-readable class names
        average     : sklearn averaging strategy for multi-class

    Returns:
        dict with all metric values + per-class breakdown
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes   = np.unique(y_true)
    n_classes = len(classes)

    if class_names is None:
        class_names = [str(c) for c in classes]

    # ── basic metrics ─────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    # ── per-class specificity ─────────────────────────────────────
    specificities = []
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    specificity_macro = float(np.mean(specificities))

    # ── ROC-AUC ───────────────────────────────────────────────────
    roc_auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr', average='macro')

    return {
        'accuracy':            acc,
        'precision':           prec,
        'recall_sensitivity':  rec,
        'specificity':         specificity_macro,
        'f1_score':            f1,
        'roc_auc':             roc_auc,
        'confusion_matrix':    cm,
        'per_class_specificity': specificities,
        'class_names':         class_names,
        'report':              classification_report(
                                   y_true, y_pred,
                                   target_names=class_names,
                                   zero_division=0),
    }


def print_metrics(m: dict):
    """Pretty-print a metrics dict from compute_metrics()."""
    print("\n" + "="*52)
    print("  CLASSIFICATION METRICS")
    print("="*52)
    print(f"  Accuracy              : {m['accuracy']:.4f}")
    print(f"  Precision (weighted)  : {m['precision']:.4f}")
    print(f"  Recall / Sensitivity  : {m['recall_sensitivity']:.4f}")
    print(f"  Specificity (macro)   : {m['specificity']:.4f}")
    print(f"  F1-score  (weighted)  : {m['f1_score']:.4f}")
    if m['roc_auc'] is not None:
        print(f"  ROC-AUC (macro OvR)   : {m['roc_auc']:.4f}")
    print("\n  Per-class specificity:")
    for name, sp in zip(m['class_names'], m['per_class_specificity']):
        print(f"    {name:20s}: {sp:.4f}")
    print("\n" + m['report'])


# ──────────────────────────────────────────────────────────────────
# Confusion Matrix plot
# ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*1.5),
                                    max(5, len(class_names)*1.3)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Confusion matrix saved → {save_path}")
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────
# ROC Curve plot  (per-class OvR + macro average)
# ──────────────────────────────────────────────────────────────────
def plot_roc_curves(y_true, y_prob, class_names, save_path=None,
                    title="ROC Curves"):
    y_true  = np.asarray(y_true)
    y_prob  = np.asarray(y_prob)
    classes = np.unique(y_true)
    y_bin   = label_binarize(y_true, classes=classes)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    fig, ax = plt.subplots(figsize=(8, 6))

    macro_fpr = np.linspace(0, 1, 200)
    macro_tpr = np.zeros_like(macro_fpr)

    for i, (cls, col) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{class_names[i]} (AUC={auc_val:.3f})")
        macro_tpr += np.interp(macro_fpr, fpr, tpr)

    macro_tpr /= len(classes)
    macro_auc  = roc_auc_score(y_true, y_prob,
                                multi_class='ovr', average='macro')
    ax.plot(macro_fpr, macro_tpr, 'k--', lw=2,
            label=f"Macro avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'gray', lw=1, linestyle=':')

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ ROC curve saved → {save_path}")
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────
# Training Curves  (Accuracy vs Epoch + Loss vs Epoch)
# ──────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_path=None,
                         title_prefix=""):
    """
    Args:
        history  : dict with keys:
                     'train_loss', 'val_loss',
                     'train_acc',  'val_acc'
                   (all lists, one value per epoch)
        save_path: if given, figure is saved there
        title_prefix: optional string prepended to subplot titles
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title_prefix} Training Curves",
                 fontsize=14, fontweight='bold', y=1.01)

    # ── Loss ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax.plot(epochs, history['val_loss'],   'r-o', markersize=4, label='Val Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} Loss vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Train Acc')
    ax.plot(epochs, history['val_acc'],   'r-o', markersize=4, label='Val Acc')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix} Accuracy vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Training curves saved → {save_path}")
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────
# Convenience: save all plots at once
# ──────────────────────────────────────────────────────────────────
def save_all_plots(metrics_dict, y_true, y_prob,
                   history=None, save_dir="results",
                   prefix="model"):
    """
    One-call helper to dump all required plots.

    Args:
        metrics_dict : output of compute_metrics()
        y_true       : ground-truth labels
        y_prob       : softmax probabilities [N, C]
        history      : training history dict (optional)
        save_dir     : output directory
        prefix       : filename prefix
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cn = metrics_dict['class_names']
    cm = metrics_dict['confusion_matrix']

    plot_confusion_matrix(cm, cn,
        save_path=f"{save_dir}/{prefix}_confusion_matrix.png",
        title=f"{prefix} – Confusion Matrix")

    if y_prob is not None:
        plot_roc_curves(y_true, y_prob, cn,
            save_path=f"{save_dir}/{prefix}_roc_curves.png",
            title=f"{prefix} – ROC Curves")

    if history is not None:
        plot_training_curves(history,
            save_path=f"{save_dir}/{prefix}_training_curves.png",
            title_prefix=prefix)

    print(f"\n  All plots saved to '{save_dir}/'")
