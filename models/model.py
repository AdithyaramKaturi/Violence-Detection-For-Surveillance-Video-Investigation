"""
Deep Learning Architecture: MobileNetV2 for Surveillance Violence Detection
===========================================================================
This module defines the deep convolutional neural network architecture based
on MobileNetV2 with an Inverted Residual & Linear Bottleneck backbone and a
custom binary classification head with regularization (Dropout + GAP).

Deep Learning Concepts Implemented:
-----------------------------------
1. Depthwise Separable Convolutions:
   - Factorizes standard convolution into Depthwise (spatial 3x3) and Pointwise (1x1 cross-channel).
   - Computational cost reduced by ~8-9x: (1/N + 1/D_k^2) compared to standard CNNs.

2. Inverted Residuals & Linear Bottlenecks:
   - Expands channels with expansion factor t=6, applies Depthwise + ReLU6,
     and projects back through a linear bottleneck to prevent manifold degradation.
   - Identity shortcut connects bottlenecks when stride=1.

3. Global Average Pooling (GAP):
   - Reduces spatial dimensions (7x7 -> 1x1) before classification, eliminating
     dense parameter explosion and reducing spatial variance sensitivity.

4. Binary Probability Estimation:
   - Single-logit linear projection passed through a Sigmoid activation:
     P(Violent | x) = 1 / (1 + exp(-z)).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ViolenceDetectionMobileNetV2(nn.Module):
    """
    MobileNetV2-based Binary Classifier for Video Surveillance Violence Detection.
    """
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.4):
        super(ViolenceDetectionMobileNetV2, self).__init__()
        
        # Load MobileNetV2 backbone (features extractor: 18 inverted residual blocks)
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
        else:
            weights = None
            
        base_mobilenet = models.mobilenet_v2(weights=weights)
        
        # Feature extraction backbone (outputs [Batch, 1280, 7, 7] for 224x224 input)
        self.features = base_mobilenet.features
        
        # Global Average Pooling compresses spatial dimensions [B, 1280, 7, 7] -> [B, 1280, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Deep Learning Classifier Head with Regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, 1)  # Single raw logit z
        )
        
        # Sigmoid activation converts raw logit z to calibrated probability P in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the deep network.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, 3, 224, 224]
        Returns:
            torch.Tensor: Calibrated violence probability [Batch, 1]
        """
        # 1. High-level spatial feature extraction through inverted residual blocks
        feat = self.features(x)
        
        # 2. Global Average Pooling
        pooled = self.gap(feat)
        flattened = torch.flatten(pooled, 1)
        
        # 3. Dense layer classification
        logits = self.classifier(flattened)
        
        # 4. Non-linear Sigmoid probability calibration
        probabilities = self.sigmoid(logits)
        return probabilities

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw pre-sigmoid logits (used for BCEWithLogitsLoss during training)."""
        feat = self.features(x)
        pooled = self.gap(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def get_target_layer_for_cam(self) -> nn.Module:
        """
        Returns the final convolutional feature layer for Grad-CAM (Explainable AI).
        In MobileNetV2, features[-1] is the final 1x1 Pointwise Conv layer (1280 channels).
        """
        return self.features[-1]


def get_dl_preprocessing_transform():
    """
    Standard Deep Learning normalization pipeline matching ImageNet statistics.
    Scales pixel values to zero-mean and unit-variance:
    x_norm = (x - mean) / std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB mean
            std=[0.229, 0.224, 0.225]    # ImageNet RGB std
        )
    ])


def preprocess_frame_for_inference(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Preprocesses a raw OpenCV BGR frame into a 4D PyTorch tensor [1, 3, 224, 224].
    """
    # 1. Color Space Conversion: BGR -> RGB
    frame_rgb = frame_bgr[:, :, ::-1]
    pil_img = Image.fromarray(frame_rgb)
    
    # 2. Apply Spatial Resizing & Channel-wise Standardization
    transform = get_dl_preprocessing_transform()
    tensor = transform(pil_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return tensor.to(device)


def get_architecture_summary(model: ViolenceDetectionMobileNetV2) -> dict:
    """
    Provides architectural and computational statistics for academic reporting.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "Backbone": "MobileNetV2 (Inverted Residuals & Linear Bottlenecks)",
        "Depth": 53,
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Frozen Parameters": frozen_params,
        "Input Dimensions": "(3, 224, 224)",
        "Final Feature Map Shape": "(1280, 7, 7)",
        "Output": "Binary Probability (Violent / Non-Violent)",
        "Computational Advantage": "Uses Depthwise Separable Convolutions reducing FLOPs by ~8.5x compared to standard VGG/ResNet networks."
    }
