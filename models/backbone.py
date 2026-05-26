"""
1D CNN backbone with multi-head self-attention for vibration signal feature extraction.

The backbone consists of 4 convolutional blocks followed by a self-attention layer,
designed for extracting discriminative features from 1D vibration signals.
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        hidden_size: Input/output feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = num_heads * self.head_dim

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """Reshape for multi-head attention: [B, L, D] -> [B, H, L, D/H]."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size].

        Returns:
            Output tensor [batch_size, seq_len, hidden_size].
        """
        Q = self.transpose_for_scores(self.query(x))  # [B, H, L, D/H]
        K = self.transpose_for_scores(self.key(x))
        V = self.transpose_for_scores(self.value(x))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [B, H, L, D/H]
        context = context.permute(0, 2, 1, 3).contiguous()  # [B, L, H, D/H]
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        return context


class CNNBackbone(nn.Module):
    """
    1D CNN feature extractor with self-attention.

    Architecture:
        Conv1d(1->16, k=15) -> Conv1d(16->32, k=3) + MaxPool ->
        Conv1d(32->64, k=3) -> Conv1d(64->128, k=3) + AdaptiveMaxPool ->
        MultiHeadSelfAttention -> FC(512->256)

    Args:
        in_channels: Number of input channels (default: 1).
        output_dim: Output feature dimension (default: 256).
        num_attention_heads: Number of self-attention heads (default: 8).
        attention_dropout: Dropout rate in attention (default: 0.1).
    """

    def __init__(self, in_channels=1, output_dim=256, num_attention_heads=8,
                 attention_dropout=0.1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )

        self.attention = MultiHeadSelfAttention(
            hidden_size=128,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: Input signal [batch_size, in_channels, signal_length].

        Returns:
            Feature vector [batch_size, output_dim].
        """
        x = self.conv_layers(x)           # [B, 128, 4]
        x = x.transpose(1, 2)            # [B, 4, 128]
        x = self.attention(x)            # [B, 4, 128]
        x = x.reshape(x.size(0), -1)    # [B, 512]
        x = self.fc(x)                   # [B, 256]
        return x
