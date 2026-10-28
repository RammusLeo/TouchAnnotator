import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class PoseDecoder(nn.Module):
    """
    Decoder for predicting hand poses from temporal features.
    
    Supports three decoder types:
    - "mlp": Global average pooling + MLP (original, simple baseline)
    - "attention": Learnable attention pooling + MLP (better spatial awareness)
    - "joint_query": DETR-style joint queries with cross-attention (best performance)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 150,  # num_joints * 3 coords
        dropout: float = 0.1,
        decoder_type: str = "mlp",
        num_query_layers: int = 2,
        num_query_heads: int = 8,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decoder_type = decoder_type
        self.num_joints = output_dim // 3
        
        if decoder_type == "mlp":
            self.decoder = self._build_mlp(input_dim, hidden_dims, output_dim, dropout)
        
        elif decoder_type == "attention":
            # Learnable attention pooling
            self.attn_pool = AttentionPooling(input_dim, num_heads=num_query_heads)
            self.decoder = self._build_mlp(input_dim, hidden_dims, output_dim, dropout)
        
        elif decoder_type == "joint_query":
            # DETR-style: each joint has a learnable query
            self.joint_queries = nn.Parameter(
                torch.randn(1, self.num_joints, input_dim) * 0.02
            )
            self.query_pos_embed = nn.Parameter(
                torch.randn(1, self.num_joints, input_dim) * 0.02
            )
            
            # Cross-attention layers
            self.cross_attn_layers = nn.ModuleList([
                JointQueryLayer(
                    embed_dim=input_dim,
                    num_heads=num_query_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_query_layers)
            ])
            self.query_norm = nn.LayerNorm(input_dim)
            
            # Per-joint regression head: D -> 3
            self.regression_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 3),
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        self._print_info()
    
    def _print_info(self):
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[PoseDecoder] type={self.decoder_type}, input_dim={self.input_dim}, "
              f"output_dim={self.output_dim}, params={num_params:,}")
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ) -> nn.Module:
        """Build MLP-based decoder."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) - temporal features from transformer
        
        Returns:
            poses: (B, T, J, 3) - predicted 3D hand poses
        """
        B, T, N, D = x.shape
        
        if self.decoder_type == "mlp":
            return self._forward_mlp(x, B, T, N, D)
        elif self.decoder_type == "attention":
            return self._forward_attention(x, B, T, N, D)
        elif self.decoder_type == "joint_query":
            return self._forward_joint_query(x, B, T, N, D)
    
    def _forward_mlp(self, x, B, T, N, D):
        """Original MLP decoder with global average pooling."""
        x = x.mean(dim=2)  # (B, T, D)
        x = x.reshape(B * T, D)
        poses_flat = self.decoder(x)  # (B*T, output_dim)
        return poses_flat.reshape(B, T, self.num_joints, 3)
    
    def _forward_attention(self, x, B, T, N, D):
        """Attention pooling + MLP decoder."""
        # Merge batch and temporal dims
        x = x.reshape(B * T, N, D)
        # Attention pooling: (B*T, N, D) -> (B*T, D)
        x = self.attn_pool(x)
        # MLP decode
        poses_flat = self.decoder(x)  # (B*T, output_dim)
        return poses_flat.reshape(B, T, self.num_joints, 3)
    
    def _forward_joint_query(self, x, B, T, N, D):
        """DETR-style joint query decoder with cross-attention."""
        # Merge batch and temporal dims for processing
        x = x.reshape(B * T, N, D)  # spatial features (keys/values)
        
        # Expand joint queries for batch: (1, J, D) -> (B*T, J, D)
        queries = self.joint_queries.expand(B * T, -1, -1)
        query_pos = self.query_pos_embed.expand(B * T, -1, -1)
        
        # Cross-attention layers: queries attend to spatial features
        for layer in self.cross_attn_layers:
            queries = layer(queries, x, query_pos)
        
        queries = self.query_norm(queries)  # (B*T, J, D)
        
        # Per-joint 3D coordinate regression
        poses = self.regression_head(queries)  # (B*T, J, 3)
        
        return poses.reshape(B, T, self.num_joints, 3)


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over spatial tokens.
    Uses a learnable query to compute attention weights.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) spatial features
        Returns:
            pooled: (B, D)
        """
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(query, x, x)  # (B, 1, D)
        pooled = self.norm(pooled.squeeze(1))  # (B, D)
        return pooled


class JointQueryLayer(nn.Module):
    """
    Single cross-attention layer for joint query decoder.
    Joint queries attend to spatial features, then self-attend for inter-joint reasoning.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention: joint queries attend to spatial features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # Self-attention: inter-joint reasoning
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        spatial_features: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, J, D) joint queries
            spatial_features: (B, N, D) spatial features from encoder
            query_pos: (B, J, D) positional embeddings for queries
        
        Returns:
            queries: (B, J, D) updated joint queries
        """
        # Cross-attention (pre-norm)
        q = self.cross_norm(queries)
        q = q + query_pos  # add positional embedding to queries
        attn_out, _ = self.cross_attn(q, spatial_features, spatial_features)
        queries = queries + attn_out
        
        # Self-attention (pre-norm) - inter-joint reasoning
        q = self.self_norm(queries)
        q = q + query_pos
        attn_out, _ = self.self_attn(q, q, q)
        queries = queries + attn_out
        
        # FFN (pre-norm)
        queries = queries + self.ffn(self.ffn_norm(queries))
        
        return queries


class TactileDecoder(nn.Module):
    """
    Original decoder for predicting tactile pressure distribution.
    Uses ConvTranspose2d spatial upsampling for structured 16x16 output.
    
    Input: fused features (B, T, D) from vision+pose fusion (global)
    Output: (B, T, 2, 16, 16) pressure maps [left_hand, right_hand]
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        conv_channels: int = 128,
        tactile_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.tactile_size = tactile_size
        self.conv_channels = conv_channels
        
        # Project to spatial seed: D → C*4*4
        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, conv_channels * 4 * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ConvTranspose upsampling: 4x4 → 8x8 → 16x16
        self.upsample = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(conv_channels, conv_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels // 2),
            nn.GELU(),
            # 8x8 → 16x16
            nn.ConvTranspose2d(conv_channels // 2, conv_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels // 4),
            nn.GELU(),
        )
        
        # Final conv: produce 2 channels (left + right hand)
        self.head = nn.Sequential(
            nn.Conv2d(conv_channels // 4, 2, kernel_size=3, padding=1),
            nn.Sigmoid(),  # pressure values in [0, 1]
        )
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[TactileDecoder] input_dim={input_dim}, tactile_size={tactile_size}x{tactile_size}, "
              f"params={num_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) fused features (already pooled from spatial tokens)
        
        Returns:
            tactile: (B, T, 2, 16, 16) pressure maps [left, right]
        """
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        
        # Project to spatial seed
        x = self.spatial_proj(x)  # (B*T, C*4*4)
        x = x.reshape(B * T, self.conv_channels, 4, 4)
        
        # Upsample to 16x16
        x = self.upsample(x)  # (B*T, C//4, 16, 16)
        
        # Final head: 2 channels
        tactile = self.head(x)  # (B*T, 2, 16, 16)
        
        return tactile.reshape(B, T, 2, self.tactile_size, self.tactile_size)


class JointLevelTactileDecoder(nn.Module):
    """
    Joint-level tactile decoder inspired by PressureFormer.
    
    Instead of decoding from a single global feature vector, this decoder
    takes per-joint features (B, T, J, D) from cross-attention fusion and
    leverages the spatial structure of hand joints to generate pressure maps.
    
    Design:
    1. Separate left/right hand joints (24 each for 48-joint input)
    2. Project each joint to a spatial feature
    3. Scatter joint features onto a learnable spatial grid
    4. Conv refinement + upsampling to 16x16
    5. Output 2-channel pressure map [left, right]
    
    Input: per-joint fused features (B, T, J, D)
    Output: (B, T, 2, 16, 16) pressure maps
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_joints: int = 48,
        conv_channels: int = 128,
        tactile_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.joints_per_hand = num_joints // 2
        self.tactile_size = tactile_size
        self.conv_channels = conv_channels
        
        # Per-joint feature projection: D -> C
        self.joint_proj = nn.Sequential(
            nn.Linear(input_dim, conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Learnable spatial positions for each joint on the grid
        # Maps each joint to a soft location on the 8x8 intermediate grid
        self.grid_size = 8
        self.joint_grid_weights = nn.Parameter(
            torch.randn(self.joints_per_hand, self.grid_size * self.grid_size) * 0.02
        )
        
        # Conv refinement on 8x8 grid
        self.refine = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.GELU(),
            nn.Conv2d(conv_channels, conv_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels // 2),
            nn.GELU(),
        )
        
        # Upsample: 8x8 -> tactile_size (dynamically adjusted)
        # For 16x16: 8 -> 16 (stride=2, one upsampling stage)
        # For 21x21: 8 -> 21 (requires a more flexible upsampling strategy)
        if tactile_size == 16:
            # 8x8 -> 16x16: single 2x upsampling stage
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(conv_channels // 2, conv_channels // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(conv_channels // 4),
                nn.GELU(),
            )
        elif tactile_size == 21:
            # 8x8 -> 21x21: first upsample to 16x16, then interpolate to 21x21
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(conv_channels // 2, conv_channels // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(conv_channels // 4),
                nn.GELU(),
                nn.Upsample(size=(tactile_size, tactile_size), mode='bilinear', align_corners=False),
            )
        else:
            raise ValueError(f"Unsupported tactile_size: {tactile_size}. Only 16 and 21 are supported.")
        
        # Per-hand output head: C//4 -> 1 channel
        self.head = nn.Sequential(
            nn.Conv2d(conv_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[JointLevelTactileDecoder] input_dim={input_dim}, joints={num_joints}, "
              f"tactile_size={tactile_size}x{tactile_size}, params={num_params:,}")
    
    def _scatter_to_grid(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Scatter per-joint features onto a 2D spatial grid using learned soft weights.
        
        Args:
            joint_features: (B, J_per_hand, C) features for one hand
        
        Returns:
            grid: (B, C, grid_size, grid_size) spatial feature grid
        """
        B, J, C = joint_features.shape
        G = self.grid_size
        
        # Soft assignment weights: (J, G*G) -> softmax over grid positions
        weights = F.softmax(self.joint_grid_weights, dim=-1)  # (J, G*G)
        
        # Weighted scatter: (B, J, C) x (J, G*G) -> (B, G*G, C)
        # weights: (J, G*G) -> (1, J, G*G)
        # joint_features: (B, J, C)
        # grid = sum_j weight[j, pos] * feature[j]  for each position pos
        grid = torch.einsum('bjc,jg->bgc', joint_features, weights)  # (B, G*G, C)
        
        # Reshape to spatial grid
        grid = grid.reshape(B, G, G, C).permute(0, 3, 1, 2)  # (B, C, G, G)
        
        return grid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D) per-joint fused features
        
        Returns:
            tactile: (B, T, 2, 16, 16) pressure maps [left, right]
        """
        B, T, J, D = x.shape
        x = x.reshape(B * T, J, D)
        
        # Project joint features
        x = self.joint_proj(x)  # (B*T, J, C)
        
        # Split left/right hand
        left_joints = x[:, :self.joints_per_hand, :]   # (B*T, 24, C)
        right_joints = x[:, self.joints_per_hand:, :]   # (B*T, 24, C)
        
        # Scatter to spatial grids
        left_grid = self._scatter_to_grid(left_joints)   # (B*T, C, 8, 8)
        right_grid = self._scatter_to_grid(right_joints)  # (B*T, C, 8, 8)
        
        # Refine + upsample + head for each hand
        left_pressure = self.head(self.upsample(self.refine(left_grid)))    # (B*T, 1, 16, 16)
        right_pressure = self.head(self.upsample(self.refine(right_grid)))  # (B*T, 1, 16, 16)
        
        # Combine
        tactile = torch.cat([left_pressure, right_pressure], dim=1)  # (B*T, 2, 16, 16)
        
        return tactile.reshape(B, T, 2, self.tactile_size, self.tactile_size)


def build_tactile_decoder(config: dict, embed_dim: int) -> nn.Module:
    """
    Factory function: build tactile decoder from config.
    
    Args:
        config: tactile_decoder config dict
        embed_dim: model embedding dimension
    
    Returns:
        TactileDecoder or JointLevelTactileDecoder
    """
    decoder_type = config.get('type', 'conv')
    dropout = config.get('dropout', 0.1)
    conv_channels = config.get('conv_channels', 128)
    tactile_size = config.get('tactile_size', 16)
    
    if decoder_type == 'conv':
        return TactileDecoder(
            input_dim=embed_dim,
            conv_channels=conv_channels,
            tactile_size=tactile_size,
            dropout=dropout,
        )
    elif decoder_type == 'joint_level':
        return JointLevelTactileDecoder(
            input_dim=embed_dim,
            num_joints=config.get('num_joints', 48),
            conv_channels=conv_channels,
            tactile_size=tactile_size,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown tactile decoder type: {decoder_type}")
