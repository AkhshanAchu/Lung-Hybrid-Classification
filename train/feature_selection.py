"""
feature_selection.py  –  Feature Selection Suite
==================================================
Three complementary selectors, all sharing the same .run(X, y) interface
so they are plug-in replacements for FireflyFeatureSelection*:

  1. FireflyFeatureSelectionSVM / MLP  (kept from original firefly.py)
  2. PSOFeatureSelection   – Particle Swarm Optimisation  (NEW)
  3. VarianceThresholdSelection – fast statistical baseline  (NEW)

All return  (best_mask: np.ndarray[int], best_acc: float)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# local import (same package)
try:
    from models.classifier import MLPClassifier
except ImportError:
    from classifier import MLPClassifier


# ──────────────────────────────────────────────────────────────────
# 1.  PSO Feature Selection
# ──────────────────────────────────────────────────────────────────
class PSOFeatureSelection:
    """
    Discrete-PSO for binary feature selection.

    Position values are real-valued velocities mapped to
    binary masks via a sigmoid transfer function (S-shaped).

    Args:
        n_particles  : swarm size
        n_features   : total number of features
        evaluator    : 'svm' or 'mlp'
        max_iter     : number of PSO iterations
        w            : inertia weight
        c1           : cognitive coefficient
        c2           : social coefficient
        device       : torch device (for MLP evaluator)
        mlp_epochs   : quick MLP training budget per evaluation
    """

    def __init__(self, n_particles=20, n_features=768,
                 evaluator='svm', max_iter=15,
                 w=0.7, c1=1.5, c2=1.5,
                 device='cuda', mlp_epochs=10):
        self.n_particles = n_particles
        self.n_features  = n_features
        self.evaluator   = evaluator
        self.max_iter    = max_iter
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.device      = device
        self.mlp_epochs  = mlp_epochs

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    def _to_binary(self, velocity):
        prob = self._sigmoid(velocity)
        return (np.random.rand(self.n_features) < prob).astype(int)

    def _fitness_svm(self, X_tr, y_tr, X_val, y_val, mask):
        if mask.sum() == 0:
            return 0.0
        clf = SVC(kernel='linear', max_iter=2000)
        clf.fit(X_tr[:, mask == 1], y_tr)
        return accuracy_score(y_val, clf.predict(X_val[:, mask == 1]))

    def _fitness_mlp(self, X_tr, y_tr, X_val, y_val, mask):
        if mask.sum() == 0:
            return 0.0
        Xtr = torch.tensor(X_tr[:, mask == 1], dtype=torch.float32).to(self.device)
        ytr = torch.tensor(y_tr, dtype=torch.long).to(self.device)
        Xv  = torch.tensor(X_val[:, mask == 1], dtype=torch.float32).to(self.device)
        yv  = torch.tensor(y_val, dtype=torch.long).to(self.device)

        model = MLPClassifier(mask.sum(), len(np.unique(y_tr))).to(self.device)
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(self.mlp_epochs):
            opt.zero_grad()
            loss = criterion(model(Xtr), ytr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        return acc

    def _fitness(self, X_tr, y_tr, X_val, y_val, mask):
        if self.evaluator == 'mlp':
            return self._fitness_mlp(X_tr, y_tr, X_val, y_val, mask)
        return self._fitness_svm(X_tr, y_tr, X_val, y_val, mask)

    # ── main loop ─────────────────────────────────────────────────
    def run(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y)

        # initialise positions (real) and velocities
        positions  = np.random.uniform(-4, 4,
                                       (self.n_particles, self.n_features))
        velocities = np.random.uniform(-1, 1,
                                       (self.n_particles, self.n_features))
        masks      = np.array([self._to_binary(p) for p in positions])

        print("Evaluating initial PSO swarm …")
        fitness = np.array([
            self._fitness(X_tr, y_tr, X_val, y_val, m)
            for m in tqdm(masks, desc="Init swarm")])

        p_best_pos = positions.copy()
        p_best_fit = fitness.copy()
        g_best_idx = np.argmax(fitness)
        g_best_pos = positions[g_best_idx].copy()
        g_best_fit = fitness[g_best_idx]

        for t in range(self.max_iter):
            pbar = tqdm(range(self.n_particles),
                        desc=f"PSO iter {t+1}/{self.max_iter}")
            for i in pbar:
                r1, r2 = np.random.rand(2, self.n_features)
                velocities[i] = (
                    self.w  * velocities[i]
                    + self.c1 * r1 * (p_best_pos[i] - positions[i])
                    + self.c2 * r2 * (g_best_pos   - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -6, 6)
                positions[i]  = positions[i] + velocities[i]
                mask = self._to_binary(positions[i])

                fit = self._fitness(X_tr, y_tr, X_val, y_val, mask)
                if fit > p_best_fit[i]:
                    p_best_fit[i] = fit
                    p_best_pos[i] = positions[i].copy()
                if fit > g_best_fit:
                    g_best_fit = fit
                    g_best_pos = positions[i].copy()
                pbar.set_postfix(best=f"{g_best_fit:.4f}")

            print(f"PSO iter {t+1}: best acc = {g_best_fit:.4f}")

        best_mask = self._to_binary(g_best_pos)
        return best_mask, g_best_fit


# ──────────────────────────────────────────────────────────────────
# 2.  Variance Threshold Baseline
# ──────────────────────────────────────────────────────────────────
class VarianceThresholdSelection:
    """
    Fast statistical baseline: keep top-k features by variance,
    then evaluate with SVM or MLP.

    Run time: seconds vs. minutes for metaheuristics.
    Use to set a lower-bound performance reference.
    """

    def __init__(self, keep_ratio=0.5, evaluator='svm',
                 device='cuda', mlp_epochs=15):
        self.keep_ratio  = keep_ratio     # fraction of features to keep
        self.evaluator   = evaluator
        self.device      = device
        self.mlp_epochs  = mlp_epochs

    def run(self, X, y):
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        n_keep = max(1, int(X_sc.shape[1] * self.keep_ratio))
        variances = X_sc.var(axis=0)
        top_idx   = np.argsort(variances)[::-1][:n_keep]
        mask = np.zeros(X_sc.shape[1], dtype=int)
        mask[top_idx] = 1

        print(f"VarianceThreshold: keeping {n_keep}/{X_sc.shape[1]} features")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sc, y, test_size=0.2, stratify=y)

        if self.evaluator == 'mlp':
            Xtr = torch.tensor(X_tr[:, mask==1], dtype=torch.float32).to(self.device)
            ytr = torch.tensor(y_tr, dtype=torch.long).to(self.device)
            Xv  = torch.tensor(X_val[:, mask==1], dtype=torch.float32).to(self.device)
            yv  = torch.tensor(y_val, dtype=torch.long).to(self.device)
            model = MLPClassifier(n_keep, len(np.unique(y))).to(self.device)
            crit  = nn.CrossEntropyLoss()
            opt   = optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(self.mlp_epochs):
                opt.zero_grad(); loss = crit(model(Xtr), ytr)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()
        else:
            clf = SVC(kernel='linear')
            clf.fit(X_tr[:, mask==1], y_tr)
            acc = accuracy_score(y_val, clf.predict(X_val[:, mask==1]))

        print(f"Variance baseline accuracy: {acc:.4f}")
        return mask, acc
