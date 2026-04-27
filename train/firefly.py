import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.classifier import MLPClassifier

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class FireflyFeatureSelectionSVM:
    def __init__(self, n_fireflies, n_features, alpha=0.5, beta0=1, gamma=1, max_iter=20):
        self.n_fireflies = n_fireflies
        self.n_features = n_features
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iter = max_iter

    def initialize_population(self):
        return np.random.randint(0, 2, (self.n_fireflies, self.n_features))

    def fitness(self, X_train, y_train, X_val, y_val, mask):
        if np.sum(mask) == 0:
            return 0
        X_train_sel = X_train[:, mask == 1]
        X_val_sel = X_val[:, mask == 1]
        clf = SVC(kernel='linear')
        clf.fit(X_train_sel, y_train)
        preds = clf.predict(X_val_sel)
        return accuracy_score(y_val, preds)

    def move_firefly(self, xi, xj, beta):
        step = beta * (xj - xi) + self.alpha * (np.random.rand(self.n_features) - 0.5)
        prob = 1 / (1 + np.exp(-step))
        new_mask = (np.random.rand(self.n_features) < prob).astype(int)
        return new_mask

    def run(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        # Initial population fitness calculation with progress bar
        print("Evaluating initial population...")
        fireflies = self.initialize_population()
        fitness_vals = np.array([
            self.fitness(X_train, y_train, X_val, y_val, mask)
            for mask in tqdm(fireflies, desc="Initial Population")
        ])

        for t in range(self.max_iter):
            pbar = tqdm(total=self.n_fireflies * self.n_fireflies, desc=f"Iteration {t+1}/{self.max_iter}")
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness_vals[j] > fitness_vals[i]:
                        r = np.sum(fireflies[i] != fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)
                        fitness_vals[i] = self.fitness(X_train, y_train, X_val, y_val, fireflies[i])
                    pbar.update(1)
            pbar.close()

            print(f"Best Accuracy so far: {np.max(fitness_vals):.4f}")

        best_index = np.argmax(fitness_vals)
        return fireflies[best_index], fitness_vals[best_index]



class FireflyFeatureSelectionMLP:
    def __init__(self, n_fireflies, n_features, alpha=0.5, beta0=1, gamma=1, max_iter=20, device="cuda", mlp_epoch=10):
        self.n_fireflies = n_fireflies
        self.n_features = n_features
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device
        self.mlp_epoch = mlp_epoch

    def initialize_population(self):
        return np.random.randint(0, 2, (self.n_fireflies, self.n_features))

    def fitness(self, X_train, y_train, X_val, y_val, mask):
        if np.sum(mask) == 0:
            return 0

        # Select masked features
        X_train_sel = X_train[:, mask == 1]
        X_val_sel = X_val[:, mask == 1]

        # Convert to tensors
        X_train_t = torch.tensor(X_train_sel, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_t = torch.tensor(X_val_sel, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Create model
        model = MLPClassifier(X_train_sel.shape[1], len(np.unique(y_train))).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train small MLP (few epochs for speed)
        model.train()
        for _ in range(self.mlp_epoch):  # small number of epochs
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()

        return acc

    def move_firefly(self, xi, xj, beta):
        step = beta * (xj - xi) + self.alpha * (np.random.rand(self.n_features) - 0.5)
        prob = 1 / (1 + np.exp(-step))
        new_mask = (np.random.rand(self.n_features) < prob).astype(int)
        return new_mask

    def run(self, X, y):
        # Scale features for better MLP convergence
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        fireflies = self.initialize_population()
        fitness_vals = np.array([self.fitness(X_train, y_train, X_val, y_val, mask) for mask in tqdm(fireflies, desc="Initial Population")])

        for t in range(self.max_iter):
            pbar = tqdm(total=self.n_fireflies * self.n_fireflies, desc=f"Iteration {t+1}/{self.max_iter}")
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness_vals[j] > fitness_vals[i]:
                        r = np.sum(fireflies[i] != fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)
                        fitness_vals[i] = self.fitness(X_train, y_train, X_val, y_val, fireflies[i])
                    pbar.update(1)
            pbar.close()

            print(f"Best Accuracy so far: {np.max(fitness_vals):.4f}")

        best_index = np.argmax(fitness_vals)
        return fireflies[best_index], fitness_vals[best_index]
