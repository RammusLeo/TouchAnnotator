"""
Pose Encoder: encodes 3D hand pose input into feature embeddings.
Used for tactile prediction where pose is an input modality.

Supports two encoder types:
- "mlp": Original flatten + MLP (simple baseline)
- "transformer": Per-joint embedding + Transformer (preserves spatial structure)
"""
import torch
import torch.nn as nn
import math
from typing import List, Optional


def _zero_module_dependency(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Keep optional branch parameters in the autograd graph for DDP.

    Some datasets do not provide an inter-wrist relation.  Returning the
    relation-free result is still semantically correct, but every DDP rank
    must see the same trainable parameter set.  A zero-valued dependency
    yields zero gradients instead of missing gradients on those batches.
    """
    dependency = reference.new_zeros(())
    for parameter in module.parameters():
        dependency = dependency + parameter.sum() * 0.0
    return dependency


class PoseEncoder(nn.Module):
    """
    Encode hand pose (B, T, J, 3) into feature embeddings (B, T, D).
    
    Architecture: flatten joints → MLP → output embedding.
    The embedding is used to fuse with visual features for tactile prediction.
    """
    
    def __init__(
        self,
        num_joints: int = 48,
        embed_dim: int = 768,
        hidden_dims: List[int] = [512, 512],
        dropout: float = 0.1,
        use_inter_wrist_relation: bool = False,
        inter_wrist_scale_m: float = 0.5,
        inter_wrist_max_distance_m: float = 1.0,
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.use_inter_wrist_relation = use_inter_wrist_relation
        input_dim = num_joints * 3  # flatten (J, 3) → J*3
        
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
        
        layers.append(nn.Linear(prev_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))
        
        self.encoder = nn.Sequential(*layers)
        self._init_inter_wrist_encoder(
            embed_dim, inter_wrist_scale_m, inter_wrist_max_distance_m
        )
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[PoseEncoder] joints={num_joints}, embed_dim={embed_dim}, params={num_params:,}")
    
    def _init_inter_wrist_encoder(self, embed_dim, scale_m, max_distance_m):
        if not self.use_inter_wrist_relation:
            return
        if scale_m <= 0 or max_distance_m <= 0:
            raise ValueError("inter-wrist scale and max distance must be positive")
        self.inter_wrist_scale_m = float(scale_m)
        self.inter_wrist_max_distance_m = float(max_distance_m)
        self.inter_wrist_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def _encode_inter_wrist(self, delta, valid):
        if not self.use_inter_wrist_relation or delta is None:
            return None
        if delta.ndim != 3 or delta.shape[-1] != 3:
            raise ValueError(
                f"inter_wrist_delta must have shape (B,T,3), got {delta.shape}"
            )
        clean = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.vector_norm(clean, dim=-1, keepdim=True).clamp_min(1e-8)
        clean = clean * torch.clamp(
            self.inter_wrist_max_distance_m / norm, max=1.0
        )
        encoded = self.inter_wrist_encoder(clean / self.inter_wrist_scale_m)
        if valid is not None:
            encoded = encoded * valid.to(encoded.dtype).unsqueeze(-1)
        return encoded

    def forward(
        self,
        poses: torch.Tensor,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            poses: (B, T, J, 3) hand pose positions
        
        Returns:
            pose_features: (B, T, D) pose embeddings
        """
        B, T, J, C = poses.shape
        x = poses.reshape(B, T, J * C)  # (B, T, J*3)
        x = x.reshape(B * T, J * C)
        features = self.encoder(x)  # (B*T, D)
        features = features.reshape(B, T, self.embed_dim)
        relation = self._encode_inter_wrist(inter_wrist_delta, inter_wrist_valid)
        if relation is not None:
            features = features + relation
        elif self.use_inter_wrist_relation:
            features = features + _zero_module_dependency(
                self.inter_wrist_encoder,
                features,
            )
        return features


class TransformerPoseEncoder(nn.Module):
    """
    Transformer-based Pose Encoder that preserves per-joint spatial structure.
    
    Inspired by PressureFormer: instead of flattening all joints into one vector,
    each joint is independently embedded and then a Transformer models inter-joint
    relationships via self-attention.
    
    Can output either:
    - Global feature (B, T, D) via pooling   (output_mode='global')
    - Per-joint features (B, T, J, D)        (output_mode='per_joint')
    
    Per-joint output enables downstream cross-attention with visual features,
    which is the key insight from PressureFormer.
    """
    
    def __init__(
        self,
        num_joints: int = 48,
        embed_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_mode: str = 'per_joint',
        normalization: str = 'legacy_camera',
        wrist_relative_scale_m: float = 0.2,
        use_inter_wrist_relation: bool = False,
        inter_wrist_scale_m: float = 0.5,
        inter_wrist_max_distance_m: float = 1.0,
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.output_mode = output_mode
        self.normalization = normalization
        self.use_inter_wrist_relation = use_inter_wrist_relation
        
        # Input normalization: fixed scaling to normalize pose coordinates
        # Expected input range: x/y in [-5, 5], z in [0, 100]
        # Scale to approximately [-1, 1] range: x/5, y/5, (z-50)/50
        # This is a fixed transformation, not learnable
        if normalization == 'legacy_camera':
            pose_scale = torch.tensor([5.0, 5.0, 50.0])
            pose_offset = torch.tensor([0.0, 0.0, 50.0])
        elif normalization == 'wrist_relative_m':
            if wrist_relative_scale_m <= 0:
                raise ValueError("wrist_relative_scale_m must be positive")
            pose_scale = torch.full((3,), float(wrist_relative_scale_m))
            pose_offset = torch.zeros(3)
        elif normalization == 'none':
            pose_scale = torch.ones(3)
            pose_offset = torch.zeros(3)
        else:
            raise ValueError(
                f"Unknown pose normalization: {normalization}. "
                "Use legacy_camera, wrist_relative_m, or none."
            )
        self.register_buffer('pose_scale', pose_scale)
        self.register_buffer('pose_offset', pose_offset)
        
        # Per-joint embedding: 3 -> D
        self.joint_embed = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        if self.use_inter_wrist_relation:
            if inter_wrist_scale_m <= 0 or inter_wrist_max_distance_m <= 0:
                raise ValueError("inter-wrist scale and max distance must be positive")
            self.inter_wrist_scale_m = float(inter_wrist_scale_m)
            self.inter_wrist_max_distance_m = float(inter_wrist_max_distance_m)
            self.inter_wrist_encoder = nn.Sequential(
                nn.Linear(3, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim),
            )
        
        # Learnable joint-type positional embedding
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, num_joints, embed_dim) * 0.02
        )
        
        # Transformer encoder: models inter-joint relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[TransformerPoseEncoder] joints={num_joints}, embed_dim={embed_dim}, "
              f"layers={num_layers}, output_mode={output_mode}, "
              f"normalization={normalization}, params={num_params:,}")
    
    def forward(
        self,
        poses: torch.Tensor,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            poses: (B, T, J, 3) hand pose positions
        
        Returns:
            If output_mode == 'per_joint':
                joint_features: (B, T, J, D) per-joint embeddings
            If output_mode == 'global':
                pose_features: (B, T, D) global pose embedding
        """
        B, T, J, C = poses.shape
        
        # Normalize input coordinates using fixed scaling
        # x/y: [-5, 5] -> [-1, 1]
        # z: [0, 100] -> [-1, 1] via (z - 50) / 50
        x = poses.reshape(B * T, J, C)
        
        # Debug: print range before normalization (only first batch)
        # if not hasattr(self, '_debug_printed'):
        #     print(f"[PoseEncoder DEBUG] Before norm: min={x.min().item():.2f}, max={x.max().item():.2f}")
        
        x = (x - self.pose_offset) / self.pose_scale  # (B*T, J, 3) - normalize to [-1, 1]
        
        # Debug: print range after normalization (only first batch)
        # if not hasattr(self, '_debug_printed'):
        #     print(f"[PoseEncoder DEBUG] After norm: min={x.min().item():.2f}, max={x.max().item():.2f}")
        #     print(f"[PoseEncoder DEBUG] pose_scale={self.pose_scale}, pose_offset={self.pose_offset}")
        #     self._debug_printed = True
        
        # Per-joint embedding
        x = self.joint_embed(x)  # (B*T, J, D)

        if self.use_inter_wrist_relation and inter_wrist_delta is not None:
            if inter_wrist_delta.shape != (B, T, 3):
                raise ValueError(
                    "inter_wrist_delta must have shape (B,T,3), got "
                    f"{tuple(inter_wrist_delta.shape)}"
                )
            delta = torch.nan_to_num(
                inter_wrist_delta, nan=0.0, posinf=0.0, neginf=0.0
            ).reshape(B * T, 3)
            delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
            delta = delta * torch.clamp(
                self.inter_wrist_max_distance_m / delta_norm, max=1.0
            )
            relation = self.inter_wrist_encoder(delta / self.inter_wrist_scale_m)
            if inter_wrist_valid is not None:
                if inter_wrist_valid.shape != (B, T):
                    raise ValueError(
                        "inter_wrist_valid must have shape (B,T), got "
                        f"{tuple(inter_wrist_valid.shape)}"
                    )
                relation = relation * inter_wrist_valid.reshape(B * T, 1).to(relation.dtype)
            x = x + relation.unsqueeze(1)
        elif self.use_inter_wrist_relation:
            x = x + _zero_module_dependency(self.inter_wrist_encoder, x)
        
        # Add joint-type positional encoding
        x = x + self.joint_pos_embed
        
        # Transformer: inter-joint reasoning
        x = self.transformer(x)  # (B*T, J, D)
        x = self.norm(x)
        
        if self.output_mode == 'per_joint':
            return x.reshape(B, T, J, self.embed_dim)
        else:
            # Global pooling
            x = x.mean(dim=1)  # (B*T, D)
            return x.reshape(B, T, self.embed_dim)


def build_pose_encoder(config: dict, embed_dim: int) -> nn.Module:
    """
    Factory function: build pose encoder from config.
    
    Args:
        config: pose_encoder config dict
        embed_dim: model embedding dimension
    
    Returns:
        PoseEncoder or TransformerPoseEncoder
    """
    encoder_type = config.get('type', 'mlp')
    num_joints = config.get('num_joints', 48)
    dropout = config.get('dropout', 0.1)
    relation_cfg = config.get('inter_wrist_relation', {})
    relation_kwargs = {
        'use_inter_wrist_relation': relation_cfg.get('enabled', False),
        'inter_wrist_scale_m': relation_cfg.get('scale_m', 0.5),
        'inter_wrist_max_distance_m': relation_cfg.get('max_distance_m', 1.0),
    }
    
    if encoder_type == 'mlp':
        return PoseEncoder(
            num_joints=num_joints,
            embed_dim=embed_dim,
            hidden_dims=config.get('hidden_dims', [512, 512]),
            dropout=dropout,
            **relation_kwargs,
        )
    elif encoder_type == 'transformer':
        return TransformerPoseEncoder(
            num_joints=num_joints,
            embed_dim=embed_dim,
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=dropout,
            output_mode=config.get('output_mode', 'per_joint'),
            normalization=config.get('normalization', 'legacy_camera'),
            wrist_relative_scale_m=config.get('wrist_relative_scale_m', 0.2),
            **relation_kwargs,
        )
    else:
        raise ValueError(f"Unknown pose encoder type: {encoder_type}")
