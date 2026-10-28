"""
Vision-Pose Fusion modules for tactile prediction.

Supports multiple fusion strategies:
- "concat": Original concat + linear projection (baseline)
- "cross_attention": Joint-level cross-attention with visual features
                     (inspired by PressureFormer's vertex-centric design)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcatFusion(nn.Module):
    """
    Original fusion: pool visual features globally, concat with pose, project.
    
    Input:  visual (B, T, N, D) + pose (B, T, D)
    Output: fused  (B, T, D)
    """
    
    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        self.output_type = 'global'
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[ConcatFusion] embed_dim={embed_dim}, params={num_params:,}")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, N, D) image patch features
            pose_features: (B, T, D) or (B, T, J, D) pose features
                           If per-joint, pools to global first.
        
        Returns:
            fused: (B, T, D) fused features
        """
        # Pool visual features spatially
        visual_pooled = visual_features.mean(dim=2)  # (B, T, D)
        
        # Pool pose features if per-joint
        if pose_features.dim() == 4:
            pose_features = pose_features.mean(dim=2)  # (B, T, D)
        
        # Concat + project
        fused = torch.cat([visual_pooled, pose_features], dim=-1)  # (B, T, 2D)
        fused = self.proj(fused)  # (B, T, D)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion inspired by PressureFormer.
    
    Each joint token queries the visual feature tokens via cross-attention,
    then self-attention models inter-joint reasoning.
    This produces per-joint features that encode both pose geometry and
    visual context from the corresponding image regions.
    
    Input:  visual (B, T, N, D) + pose_joints (B, T, J, D)
    Output: fused  (B, T, J, D) per-joint features with visual context
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_type = 'per_joint'
        
        # Cross-attention + Self-attention layers
        self.layers = nn.ModuleList([
            FusionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[CrossAttentionFusion] embed_dim={embed_dim}, layers={num_layers}, "
              f"heads={num_heads}, params={num_params:,}")
    
    def forward(
        self,
        visual_features: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, T, N, D) image patch features
            pose_features: (B, T, J, D) per-joint pose features
        
        Returns:
            fused: (B, T, J, D) per-joint features enriched with visual context
        """
        B, T, N, D = visual_features.shape
        J = pose_features.shape[2]
        
        # Flatten batch and time
        visual = visual_features.reshape(B * T, N, D)
        joints = pose_features.reshape(B * T, J, D)
        
        # Cross-attention layers: joints query visual features
        for layer in self.layers:
            joints = layer(joints, visual)
        
        joints = self.norm(joints)  # (B*T, J, D)
        
        return joints.reshape(B, T, J, D)


class FusionLayer(nn.Module):
    """
    Single fusion layer: cross-attention (joint→visual) + self-attention (joint↔joint) + FFN.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention: joints attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_dropout = nn.Dropout(dropout)
        
        # Self-attention: inter-joint reasoning
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_dropout = nn.Dropout(dropout)
        
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
        joint_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            joint_features: (B, J, D) per-joint features (query)
            visual_features: (B, N, D) visual patch features (key/value)
        
        Returns:
            joint_features: (B, J, D) updated joint features
        """
        # Cross-attention (pre-norm): joints query visual features
        q = self.cross_norm(joint_features)
        attn_out, _ = self.cross_attn(query=q, key=visual_features, value=visual_features)
        joint_features = joint_features + self.cross_dropout(attn_out)
        
        # Self-attention (pre-norm): inter-joint reasoning
        q = self.self_norm(joint_features)
        attn_out, _ = self.self_attn(query=q, key=q, value=q)
        joint_features = joint_features + self.self_dropout(attn_out)
        
        # FFN (pre-norm)
        joint_features = joint_features + self.ffn(self.ffn_norm(joint_features))
        
        return joint_features


def build_fusion(config: dict, embed_dim: int) -> nn.Module:
    """
    Factory function: build fusion module from config.
    
    Args:
        config: fusion config dict
        embed_dim: model embedding dimension
    
    Returns:
        ConcatFusion or CrossAttentionFusion
    """
    fusion_type = config.get('type', 'concat')
    dropout = config.get('dropout', 0.1)
    
    if fusion_type == 'concat':
        return ConcatFusion(
            embed_dim=embed_dim,
            dropout=dropout,
        )
    elif fusion_type == 'cross_attention':
        return CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 2),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
