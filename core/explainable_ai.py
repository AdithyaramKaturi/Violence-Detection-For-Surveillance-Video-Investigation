"""
Explainable AI (XAI) Engine: Grad-CAM for Deep Neural Networks
==============================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to provide
visual transparency into the decision-making process of the MobileNetV2 model.

Deep Learning Theory:
--------------------
Standard deep neural networks are often treated as "black boxes." Grad-CAM
uses the gradient of any target concept (here, the violence class score y^c)
with respect to the feature map A^k of the final convolutional layer of the
network to compute an importance weight alpha_k^c for each channel:

    alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)

Where:
    - Z = u * v is the spatial dimension (7 * 7 = 49).
    - d y^c / d A_{i,j}^k is the gradient flowing back from backpropagation.

The class-discriminative localization map L_Grad-CAM is obtained by a weighted
combination of forward activation maps, passed through a ReLU activation:

    L_Grad-CAM^c = ReLU( sum_k alpha_k^c * A^k )

ReLU ensures only features with a POSITIVE correlation to the violent class
are emphasized, suppressing non-violent context.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for MobileNetV2.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks to capture feature maps and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains gradients with respect to output feature map
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Cleans up hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Executes forward pass, backpropagation, and computes the 2D Grad-CAM heatmap.
        Args:
            input_tensor (torch.Tensor): Preprocessed input tensor [1, 3, 224, 224]
        Returns:
            np.ndarray: Normalized 2D Grad-CAM heatmap of shape (224, 224) in range [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # 1. Forward pass to get raw logits
        if hasattr(self.model, 'forward_logits'):
            logits = self.model.forward_logits(input_tensor)
        else:
            logits = self.model(input_tensor)

        # 2. Target concept: score for the single binary class logit
        target_score = logits[0, 0]

        # 3. Backward pass to compute gradients d(target_score) / d(feature_maps)
        target_score.backward(retain_graph=True)

        # Gradients: [1, 1280, 7, 7], Activations: [1, 1280, 7, 7]
        gradients = self.gradients
        activations = self.activations

        # 4. Global Average Pooling of gradients to compute channel importance weights alpha_k
        # alpha shape: [1, 1280, 1, 1]
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 5. Weighted linear combination of activation maps
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)  # Shape: [1, 1, 7, 7]

        # 6. Apply ReLU: retain features that positively contribute to violence prediction
        cam = torch.relu(cam)

        # Convert to numpy array and remove batch/channel dimensions
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_normalized = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = np.zeros_like(cam_np)

        # Resize heatmap to 224x224 matching input dimensions
        heatmap_resized = cv2.resize(cam_normalized, (224, 224))
        return heatmap_resized

    def overlay_heatmap_on_frame(
        self, 
        frame_bgr: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.55
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlays the 2D Grad-CAM attention heatmap onto the original video frame.
        Args:
            frame_bgr (np.ndarray): Original frame (H, W, 3) in BGR format
            heatmap (np.ndarray): 2D Grad-CAM heatmap (224, 224)
            alpha (float): Opacity blend ratio between original frame and heatmap
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Grad-CAM Color Map, Superimposed Visualization)
        """
        h, w = frame_bgr.shape[:2]
        
        # Upsample heatmap to full frame dimensions using Bicubic interpolation
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Scale to [0, 255] integer range
        heatmap_uint8 = np.uint8(255 * heatmap_full)
        
        # Apply JET colormap: Red = High Attention (Violent Cues), Blue = Low Attention
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Alpha blending: Superimpose heatmap over original frame
        superimposed = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        
        return heatmap_color, superimposed
