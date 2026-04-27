import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from tqdm import tqdm

# YOUR actual model import
from models.attention_classifier import HybridAttentionClassifier


class FeatureExtractor(nn.Module):
    """
    Extract features from trained HybridAttentionClassifier
    """

    def __init__(self, trained_model):
        super().__init__()

        # Your model uses self.features
        self.features = trained_model.features

    def forward(self, x):
        # Forward through ConvNeXt + attention backbone
        x = self.features(x)

        # Global Average Pooling
        x = x.mean(dim=[-2, -1])

        return x


def extract_features(
    model_path: str,
    dataloader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:

    print("Loading trained HybridAttentionClassifier model...")

    # SAME model used during training
    model = HybridAttentionClassifier(
        num_classes=4,
        input_channels=6
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Handle both direct state_dict and wrapped checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load safely
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Model loaded successfully")

    if len(missing) > 0:
        print(f"Missing keys: {len(missing)}")

    if len(unexpected) > 0:
        print(f"Unexpected keys: {len(unexpected)}")

    model.eval()

    # Feature extractor
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()

    features_list = []
    labels_list = []

    print("Extracting features...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)

            # Extract deep features
            features = feature_extractor(images)

            features_list.append(
                features.cpu().numpy()
            )

            labels_list.append(
                labels.cpu().numpy()
            )

    # Merge batches
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    return features, labels