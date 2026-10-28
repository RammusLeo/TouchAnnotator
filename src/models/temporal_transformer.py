import torch
import torch.nn as nn
import math
from typing import Optional


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for modeling temporal dependencies in video.
    Uses divided space-time attention (spatial and temporal separately).
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window_size: int = 16,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Input projection (from vision encoder dim to temporal transformer dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding for temporal dimension
        self.temporal_pos_encoding = PositionalEncoding(embed_dim, max_len=window_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        print(f"[TemporalTransformer] embed_dim={embed_dim}, layers={num_layers}, heads={num_heads}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) - batch of frame-wise features
                B = batch size
                T = temporal length
                N = number of spatial tokens (patches)
                D = embed_dim
        
        Returns:
            out: (B, T, N, D) - temporally enhanced features
        """
        B, T, N, D = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, T, N, D)
        
        # Add temporal positional encoding
        # Reshape to (B*N, T, D) for temporal encoding
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.temporal_pos_encoding(x)  # (B*N, T, D)
        x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)
        
        return x


class TemporalTransformerLayer(nn.Module):
    """Single layer of temporal transformer with divided attention."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Temporal attention (across time)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Spatial attention (across patches)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D)
        
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # Temporal attention (pre-norm): attend across time for each spatial location
        # Reshape to (B*N, T, D)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        x_normed = self.temporal_norm(x_temporal)
        attn_out, _ = self.temporal_attn(x_normed, x_normed, x_normed)
        x_temporal = x_temporal + attn_out
        
        # Reshape back to (B, T, N, D)
        x = x_temporal.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        # Spatial attention (pre-norm): attend across patches for each time step
        # Reshape to (B*T, N, D)
        x_spatial = x.reshape(B * T, N, D)
        
        x_normed = self.spatial_norm(x_spatial)
        attn_out, _ = self.spatial_attn(x_normed, x_normed, x_normed)
        x_spatial = x_spatial + attn_out
        
        # Reshape back to (B, T, N, D)
        x = x_spatial.reshape(B, T, N, D)
        
        # MLP (pre-norm)
        x = x + self.mlp(self.mlp_norm(x))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        
        Returns:
            x: (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
