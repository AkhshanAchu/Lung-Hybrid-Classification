"""
main_hybrid_unet.py
====================
Train the new AttentionHybridUNet (with MultiScaleHybridAttention skip gates)
alongside the classic AttU_Net, log all metrics, and save plots.

Usage:
    python main_hybrid_unet.py
"""

import os
import torch
from pathlib import Path

from utils.data_create import get_loaders_combined
from models.unet import AttU_Net
from models.attention_unet import AttentionHybridUNet
from train.train_unet import train_segmentation_model

# ──────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
SAVE_DIR     = "checkpoint_hybrid_unet"
EPOCHS       = 20
BATCH_SIZE   = 8
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = (256, 256)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    print("Creating data loaders …")
    train_loader, val_loader = get_loaders_combined(
        base_dir   = DATASET_PATH,
        image_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
    )


    # ══════════════════════════════════════════════════════════════
    # MODEL 2  –  AttentionHybridUNet (our new model)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print(" MODEL 2  –  AttentionHybridUNet (MultiScaleHybrid skip gates)")
    print("═"*55)
    model2 = AttentionHybridUNet(
        img_ch    = 3,
        output_ch = 1,
        img_size  = IMAGE_SIZE[0],
        num_heads = 4,
        head_dim  = 32,
    )
    trainer2, hist2 = train_segmentation_model(
        model        = model2,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = EPOCHS,
        learning_rate= 1e-3,
        device       = DEVICE,
    )
    Path("checkpoints").rename(Path(SAVE_DIR) / "hybrid_unet_checkpoints")
    if Path("training_plots.png").exists():
        Path("training_plots.png").rename(
            Path(SAVE_DIR) / "hybrid_unet_training_plots.png")
        
    # ══════════════════════════════════════════════════════════════
    # MODEL 1  –  Classic AttU_Net (baseline)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print(" MODEL 1  –  Classic AttU_Net (baseline)")
    print("═"*55)
    model1 = AttU_Net(img_ch=3, output_ch=1)
    trainer1, hist1 = train_segmentation_model(
        model        = model1,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = EPOCHS,
        learning_rate= 1e-3,
        device       = DEVICE,
    )
    # rename output files so they don't clash
    Path("checkpoints").rename(Path(SAVE_DIR) / "attunet_checkpoints")
    if Path("training_plots.png").exists():
        Path("training_plots.png").rename(
            Path(SAVE_DIR) / "attunet_training_plots.png")


    # ── Comparison summary ────────────────────────────────────────
    def best(hist, key): return max(hist[key])

    print("\n" + "═"*55)
    print("  SEGMENTATION COMPARISON")
    print("═"*55)
    print(f"{'Model':<30} {'Best Val IoU':>13} {'Best Val Dice':>14}")
    print("-"*55)
    print(f"{'AttU_Net (baseline)':<30} "
          f"{best(hist1,'val_iou'):>13.4f} "
          f"{best(hist1,'val_dice'):>14.4f}")
    print(f"{'AttentionHybridUNet (ours)':<30} "
          f"{best(hist2,'val_iou'):>13.4f} "
          f"{best(hist2,'val_dice'):>14.4f}")
    print(f"\nAll results saved to '{SAVE_DIR}/'")


if __name__ == "__main__":
    main()
