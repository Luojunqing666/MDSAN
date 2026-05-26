"""
MDSAN: Modified Deep Subdomain Adaptation Network.

Combines a CNN+Attention feature extractor with LMMD (Local MMD) loss
for unsupervised cross-domain fault diagnosis under speed fluctuation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNBackbone
from .lmmd import LMMDLoss


class MDSAN(nn.Module):
    """
    Modified Deep Subdomain Adaptation Network.

    Architecture:
        Input -> CNNBackbone (feature extractor) -> Classifier FC
        + LMMD loss for subdomain alignment between source and target.

    Args:
        num_classes: Number of fault classes.
        feature_dim: Feature dimension from backbone (default: 256).
        in_channels: Number of input signal channels (default: 1).
    """

    def __init__(self, num_classes=10, feature_dim=256, in_channels=1):
        super().__init__()
        self.feature_layers = CNNBackbone(in_channels=in_channels, output_dim=feature_dim)
        self.cls_fc = nn.Linear(feature_dim, num_classes)
        self.lmmd_loss = LMMDLoss(num_classes=num_classes)

    def forward(self, source, target, s_label):
        """
        Forward pass with domain adaptation.

        Args:
            source: Source domain signals [batch_size, channels, signal_length].
            target: Target domain signals [batch_size, channels, signal_length].
            s_label: Source domain labels [batch_size].

        Returns:
            Tuple of (source_predictions, lmmd_loss).
        """
        # Extract features
        source_feat = self.feature_layers(source)
        target_feat = self.feature_layers(target)

        # Classification
        s_pred = self.cls_fc(source_feat)
        t_pred = self.cls_fc(target_feat)

        # Compute LMMD loss for subdomain alignment
        loss_lmmd = self.lmmd_loss(
            source_feat, target_feat, s_label, F.softmax(t_pred, dim=1)
        )

        return s_pred, loss_lmmd

    def predict(self, x):
        """
        Inference-only forward pass.

        Args:
            x: Input signal [batch_size, channels, signal_length].

        Returns:
            Class logits [batch_size, num_classes].
        """
        feat = self.feature_layers(x)
        return self.cls_fc(feat)
