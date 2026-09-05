"""
Model definition: EfficientNet-B0 pretrained on ImageNet, fine-tuned for
5-class diabetic retinopathy severity grading.
"""

import torch.nn as nn
from torchvision import models


class DRModel(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=False):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.backbone = backbone
        # exposed so Grad-CAM can hook the last conv block
        self.target_layer = self.backbone.features[-1]

    def forward(self, x):
        return self.backbone(x)
