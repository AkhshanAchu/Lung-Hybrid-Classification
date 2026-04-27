"""
train_classifier_hybrid.py
===========================
Training loop for HybridAttentionClassifier / ResNetHybridClassifier.
Tracks all required metrics per epoch and saves:
  • best model checkpoint
  • training_history.json
  • training curves PNG  (Acc vs Epoch, Loss vs Epoch)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import json
from pathlib import Path

from utils.metrics import plot_training_curves


def train_hybrid_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    lr=1e-4,
    device="cuda",
    save_path="best_hybrid_model.pth",
    model_name="HybridModel",
):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'lr':         [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]  –  {model_name}")
        print("-" * 50)

        # ── Training ──────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * images.size(0)
            t_correct += (outputs.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= t_total
        t_acc   = t_correct / t_total

        # ── Validation ────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                v_loss    += loss.item() * images.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)

        v_loss /= v_total
        v_acc   = v_correct / v_total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['lr'].append(current_lr)

        print(f"Train  Loss: {t_loss:.4f}  Acc: {t_acc:.4f}")
        print(f"Val    Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  LR: {current_lr:.2e}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print("  ✔ Saved best model")

    print(f"\nTraining complete – Best Val Acc: {best_val_acc:.4f}")

    # Save history JSON
    hist_path = Path(save_path).parent / f"{Path(save_path).stem}_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save training curves plot
    curves_path = Path(save_path).parent / f"{Path(save_path).stem}_curves.png"
    plot_training_curves(history,
                         save_path=str(curves_path),
                         title_prefix=model_name)

    return model, history
