import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=2, input_channels=6):
        super().__init__()
        self.backbone = convnext_tiny(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(input_channels, 96, kernel_size=4, stride=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.classifier(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),       # bigger first layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),             # second hidden layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)      # output layer
        )

    def forward(self, x):
        return self.net(x)
    

class BetterMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BetterMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
