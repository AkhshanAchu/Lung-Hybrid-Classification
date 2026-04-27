import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm

from utils.data_create import get_train_val_dataloaders
from models.classifier import ConvNeXtClassifier


def main():
    DATASET_PATH = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset"
    MODEL_PATH = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\project\best_model_mutliclass.pth"
    BATCH_SIZE = 8
    IMAGE_SIZE = (256, 256)
    NUM_CLASSES = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # Load Data
    # -------------------------------
    print("Data Loading...")
    _, val_loader = get_train_val_dataloaders(
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        image_size=IMAGE_SIZE,
        num_workers=0
    )

    # -------------------------------
    # Load Model
    # -------------------------------
    print("Model Loading...")
    model = ConvNeXtClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # -------------------------------
    # Validation Loop
    # -------------------------------
    print("Validating...")
    val_loss = 0.0
    all_labels = []
    all_preds = []
    times = []  # store per-image inference times

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)

            # time measurement
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            batch_time = end_time - start_time
            per_image_time = batch_time / images.size(0)
            times.extend([per_image_time] * images.size(0))

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)
    print("Done...")

    # -------------------------------
    # Metrics
    # -------------------------------
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print("\n--- Validation Results ---")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Avg Inference Time per Image: {avg_time*1000:.2f} ms")
    print(f"Fastest: {min_time*1000:.2f} ms, Slowest: {max_time*1000:.2f} ms")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # -------------------------------
    # Plots
    # -------------------------------
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
                yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Bar chart of metrics
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_values = [acc, prec, rec, f1]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=metrics_names, y=metrics_values)
    plt.ylim(0, 1)
    plt.title("Validation Metrics")
    for i, val in enumerate(metrics_values):
        plt.text(i, val + 0.01, f"{val:.2f}", ha='center')
    plt.savefig("validation_metrics.png")
    plt.close()

    # Inference time distribution
    plt.figure(figsize=(6, 4))
    plt.hist([t*1000 for t in times], bins=30, color="purple", alpha=0.7)
    plt.xlabel("Inference Time per Image (ms)")
    plt.ylabel("Frequency")
    plt.title("Inference Time Distribution")
    plt.savefig("inference_times.png")
    plt.close()

    print("\nSaved 'confusion_matrix.png', 'validation_metrics.png' and 'inference_times.png'")


if __name__ == "__main__":
    main()
