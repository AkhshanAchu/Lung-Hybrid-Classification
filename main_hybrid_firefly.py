"""
main_hybrid_firefly.py
=======================
Feature-selection comparison pipeline:

  A. Variance Threshold (fast baseline)
  B. Firefly + SVM       (original method)
  C. Firefly + MLP       (original method)
  D. PSO + SVM           (new)
  E. PSO + MLP           (new)

Each selector feeds into two final classifiers (SVM, MLP).
Full evaluation metrics are printed and plots saved.

Usage:
    python main_hybrid_firefly.py
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess.feature import extract_features
from utils.data_create import PreMaskedClassificationDataset
from models.classifier import MLPClassifier, BetterMLP
from train.firefly import FireflyFeatureSelectionSVM, FireflyFeatureSelectionMLP
from train.feature_selection import PSOFeatureSelection, VarianceThresholdSelection
from utils.metrics import (
    compute_metrics, print_metrics, save_all_plots,
    plot_training_curves,
)

# ──────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
MODEL_PATH   = r"C:\Users\akhsh\Downloads\LungClassification-hybrid\LungClassification-master\LungClassification-master\checkpoint_hybrid\hybrid_convnext_best.pth"
SAVE_DIR     = "results_feature_selection"
CLASS_NAMES  = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────


def eval_with_mlp(X_tr, y_tr, X_te, y_te, epochs=15):
    """Quick BetterMLP evaluation returning (preds, probs, acc)."""
    dev = DEVICE
    Xtr = torch.tensor(X_tr, dtype=torch.float32).to(dev)
    ytr = torch.tensor(y_tr, dtype=torch.long).to(dev)
    Xte = torch.tensor(X_te, dtype=torch.float32).to(dev)

    n_cls  = len(np.unique(y_tr))
    model  = BetterMLP(X_tr.shape[1], n_cls).to(dev)
    crit   = nn.CrossEntropyLoss()
    opt    = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(Xtr), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    acc = accuracy_score(y_te, preds)
    return preds, probs, acc


def eval_with_svm(X_tr, y_tr, X_te, y_te):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)
    acc   = accuracy_score(y_te, preds)
    return preds, probs, acc


def run_experiment(name, mask, X_tr, y_tr, X_te, y_te,
                   eval_method='mlp'):
    X_tr_sel = X_tr[:, mask == 1]
    X_te_sel = X_te[:, mask == 1]
    n_sel    = mask.sum()
    print(f"\n  [{name}] n_features selected: {n_sel}")

    t0 = time.time()
    if eval_method == 'mlp':
        preds, probs, acc = eval_with_mlp(X_tr_sel, y_tr, X_te_sel, y_te)
    else:
        preds, probs, acc = eval_with_svm(X_tr_sel, y_tr, X_te_sel, y_te)
    inf_ms = (time.time() - t0) / len(y_te) * 1000

    m = compute_metrics(y_te, preds, probs, class_names=CLASS_NAMES)
    print_metrics(m)
    print(f"  Avg inference time: {inf_ms:.3f} ms/sample")
    return m, inf_ms


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Feature extraction ────────────────────────────────────────
    dataset    = PreMaskedClassificationDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    features, labels = extract_features(MODEL_PATH, dataloader, DEVICE)
    X, y = features.copy(), labels.copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # ══════════════════════════════════════════════════════════════
    # A. Variance Threshold baseline
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print(" A. Variance Threshold (fast baseline)")
    print("═"*55)
    vt = VarianceThresholdSelection(keep_ratio=0.5, evaluator='svm', device=DEVICE)
    mask_vt, _ = vt.run(X, y)
    m_vt_svm, _ = run_experiment("VarThresh+SVM",  mask_vt, X_tr, y_tr, X_te, y_te, 'svm')
    m_vt_mlp, _ = run_experiment("VarThresh+MLP",  mask_vt, X_tr, y_tr, X_te, y_te, 'mlp')
    save_all_plots(m_vt_mlp, y_te,
                   None,  # y_prob not stored here – pass from run_experiment if needed
                   save_dir=SAVE_DIR, prefix="VarThresh_MLP")

    # ══════════════════════════════════════════════════════════════
    # B/C. Firefly (original)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print(" B. Firefly + SVM (original)")
    print("═"*55)
    fa_svm = FireflyFeatureSelectionSVM(
        n_fireflies=10, n_features=X.shape[1], max_iter=5)
    mask_ff_svm, _ = fa_svm.run(X, y)

    print("\n" + "═"*55)
    print(" C. Firefly + MLP (original)")
    print("═"*55)
    fa_mlp = FireflyFeatureSelectionMLP(
        n_fireflies=10, n_features=X.shape[1], max_iter=5, device=DEVICE)
    mask_ff_mlp, _ = fa_mlp.run(X, y)

    m_ff_svm, _ = run_experiment("Firefly+SVM", mask_ff_svm, X_tr, y_tr, X_te, y_te, 'svm')
    m_ff_mlp, _ = run_experiment("Firefly+MLP", mask_ff_mlp, X_tr, y_tr, X_te, y_te, 'mlp')

    # ══════════════════════════════════════════════════════════════
    # D/E. PSO (new)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print(" D. PSO + SVM (new)")
    print("═"*55)
    pso_svm = PSOFeatureSelection(
        n_particles=10, n_features=X.shape[1],
        evaluator='svm', max_iter=8)
    mask_pso_svm, _ = pso_svm.run(X, y)

    print("\n" + "═"*55)
    print(" E. PSO + MLP (new)")
    print("═"*55)
    pso_mlp = PSOFeatureSelection(
        n_particles=10, n_features=X.shape[1],
        evaluator='mlp', max_iter=8, device=DEVICE)
    mask_pso_mlp, _ = pso_mlp.run(X, y)

    m_pso_svm, _ = run_experiment("PSO+SVM", mask_pso_svm, X_tr, y_tr, X_te, y_te, 'svm')
    m_pso_mlp, _ = run_experiment("PSO+MLP", mask_pso_mlp, X_tr, y_tr, X_te, y_te, 'mlp')

    # ── Feature-selection comparison bar chart ────────────────────
    methods = [
        "VarThresh+SVM", "VarThresh+MLP",
        "Firefly+SVM",   "Firefly+MLP",
        "PSO+SVM",       "PSO+MLP",
    ]
    accs = [
        m_vt_svm['accuracy'],  m_vt_mlp['accuracy'],
        m_ff_svm['accuracy'],  m_ff_mlp['accuracy'],
        m_pso_svm['accuracy'], m_pso_mlp['accuracy'],
    ]
    colors = ['#4C72B0','#4C72B0','#55A868','#55A868','#C44E52','#C44E52']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, accs, color=colors, edgecolor='white', width=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Feature Selection Method Comparison", fontsize=13, fontweight='bold')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha='center', fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(f"{SAVE_DIR}/method_comparison.png", dpi=150)
    plt.close(fig)

    # ── Final table ───────────────────────────────────────────────
    all_results = {
        "VarThresh+SVM": m_vt_svm,  "VarThresh+MLP": m_vt_mlp,
        "Firefly+SVM":   m_ff_svm,  "Firefly+MLP":   m_ff_mlp,
        "PSO+SVM":       m_pso_svm, "PSO+MLP":       m_pso_mlp,
    }
    print("\n" + "═"*70)
    print("  FEATURE SELECTION COMPARISON TABLE")
    print("═"*70)
    hdr = f"{'Method':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'F1':>6}"
    print(hdr)
    print("-"*len(hdr))
    for name, m in all_results.items():
        print(f"{name:<18} {m['accuracy']:.4f} {m['precision']:.4f} "
              f"{m['recall_sensitivity']:.4f} {m['specificity']:.4f} "
              f"{m['f1_score']:.4f}")

    print(f"\nAll results and plots saved to '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
