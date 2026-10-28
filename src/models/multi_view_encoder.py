"""
Multi-View Vision Encoder with View Dropout.

Processes multiple camera views (egocentric + wrist cameras) through a shared
video backbone, adds view-specific embeddings, and fuses via cross-view attention.

Key features:
- Shared vision encoder across views (parameter efficient)
- Learnable view-type embeddings to distinguish camera perspectives
- View dropout during training → robust to missing views at inference
- Cross-view attention fusion → output same shape as single-view encoder

Training: randomly drop views so model learns all subsets
Inference: feed any subset of views (ego only, wrist only, or all three)

Output: (B, T, N, D) — identical to single-view VisionEncoder output,
        so all downstream modules (TemporalTransformer, Fusion, Decoder) are unchanged.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional


VIEW_NAMES = ['ego', 'wrist_left', 'wrist_right']


class MultiViewEncoder(nn.Module):
    """
    Multi-view vision encoder with view dropout and cross-view fusion.

    Architecture:
        1. Each view → shared video encoder → (B,T,N,D) patch tokens
        2. Add learnable view embedding per view type
        3. View dropout: randomly skip views during training
        4. Cross-view fusion: attention on view summaries → gated combination
        5. Output: (B,T,N,D) — same as single-view encoder
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        embed_dim: int = 768,
        num_cross_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        view_dropout_prob: float = 0.5,
        summary_fusion: bool = True,
    ):
        """
        Args:
            vision_encoder: Shared video VisionEncoder instance
            embed_dim: Feature dimension (must match vision_encoder output)
            num_cross_layers: Number of cross-view transformer layers
            num_heads: Attention heads for cross-view transformer
            mlp_ratio: MLP expansion ratio in cross-view transformer
            dropout: Dropout rate
            view_dropout_prob: Probability of dropping each view during training
                               (at least one view is always kept)
            summary_fusion: Build the legacy summary-attention fusion path. The
                            tactile-query path keeps per-view local tokens and
                            does not need these parameters.
        """
        super().__init__()

        self.vision_encoder = vision_encoder
        self.embed_dim = embed_dim
        self.view_dropout_prob = view_dropout_prob
        self.summary_fusion_enabled = summary_fusion

        # Learnable view-type embeddings (added to all patch tokens of a view)
        self.view_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
            for name in VIEW_NAMES
        })
        for v in self.view_embeddings.values():
            nn.init.normal_(v, std=0.02)

        if self.summary_fusion_enabled:
            # Cross-view fusion transformer operating on one summary per view.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.cross_view_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_cross_layers,
            )
            self.view_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )
            self.output_norm = nn.LayerNorm(embed_dim)
        else:
            self.cross_view_transformer = None
            self.view_gate = None
            self.output_norm = None

        num_params = sum(p.numel() for p in self.parameters()) - \
                     sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"[MultiViewEncoder] views={VIEW_NAMES}, "
              f"cross_layers={num_cross_layers}, heads={num_heads}, "
              f"view_dropout={view_dropout_prob}, "
              f"summary_fusion={self.summary_fusion_enabled}, "
              f"extra_params={num_params:,}")

    def _apply_view_dropout(self, available_views: List[str]) -> List[str]:
        """
        Randomly drop wrist views during training. Ego view is always kept.

        Args:
            available_views: List of view names that have input data

        Returns:
            keep: List of view names to actually process
        """
        if not self.training or len(available_views) <= 1:
            return available_views

        if self.view_dropout_prob <= 0:
            return available_views

        keep = []
        for name in available_views:
            if name == 'ego':
                # Ego view is always kept
                keep.append(name)
            elif torch.rand(1).item() > self.view_dropout_prob:
                keep.append(name)

        # Fallback: if ego wasn't in available_views, keep at least one
        if not keep:
            keep = [available_views[0]]

        return keep

    def encode_views(
        self,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode active views without destroying their spatial correspondence.

        All active streams are concatenated along the batch dimension and sent
        through the shared frozen tower once. The returned dictionary is used
        by hand-centric tactile queries; the legacy ``forward`` method still
        performs summary-gated fusion for older configurations.
        """
        available_views = [name for name in VIEW_NAMES if name in views]
        if view_valid is not None:
            available_views = [
                name for name in available_views
                if name not in view_valid or bool(view_valid[name].any().item())
            ]
        if not available_views:
            raise ValueError(
                f"At least one view is required, got keys={list(views.keys())}"
            )
        active_views = self._apply_view_dropout(available_views)
        batch_size = views[active_views[0]].shape[0]
        reference_shape = tuple(views[active_views[0]].shape[1:])
        for name in active_views:
            if tuple(views[name].shape[1:]) != reference_shape:
                raise ValueError(
                    f"All active views must share shape after the batch axis; "
                    f"{name} has {tuple(views[name].shape)}"
                )

        stacked_views = torch.cat([views[name] for name in active_views], dim=0)
        stacked_features = self.vision_encoder(stacked_views)
        feature_chunks = stacked_features.split(batch_size, dim=0)
        encoded = {
            name: features + self.view_embeddings[name]
            for name, features in zip(active_views, feature_chunks)
        }
        # DDP can skip its expensive unused-parameter graph traversal when
        # dropped/unavailable views still contribute an explicit zero gradient.
        inactive_names = [name for name in VIEW_NAMES if name not in active_views]
        if inactive_names:
            zero_dependency = sum(
                self.view_embeddings[name].sum()
                for name in inactive_names
            ) * 0.0
            first_name = active_views[0]
            encoded[first_name] = encoded[first_name] + zero_dependency
        return encoded

    def forward(
        self,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            views: Dict mapping view_name → (B, T, 3, H, W) video tensor.
                   Any subset of VIEW_NAMES can be provided.
                   Must provide at least one view.
            view_valid: Optional per-view boolean tensors with shape (B,). Invalid
                        views are excluded from attention and gated fusion per sample.

        Returns:
            features: (B, T, N, D) — fused multi-view features,
                      same shape as single-view VisionEncoder output.
        """
        if not self.summary_fusion_enabled:
            raise RuntimeError(
                'Legacy MultiViewEncoder.forward requires summary_fusion=true; '
                'use encode_views() for tactile-query fusion'
            )
        encoded = self.encode_views(views, view_valid=view_valid)

        B, T, N, D = next(iter(encoded.values())).shape

        # Single view → skip cross-view attention
        if len(encoded) == 1:
            return self.output_norm(next(iter(encoded.values())))

        # --- Multi-view fusion ---
        # 1. Compute per-view summary tokens (mean pool patches)
        ordered_names = [n for n in VIEW_NAMES if n in encoded]
        features_list = []  # (B*T, N, D) per view
        summaries = []      # (B*T, D) per view

        for name in ordered_names:
            feat_flat = encoded[name].reshape(B * T, N, D)
            features_list.append(feat_flat)
            summaries.append(feat_flat.mean(dim=1))  # (B*T, D)

        V = len(summaries)

        # 2. Cross-view attention on summary tokens
        summary_stack = torch.stack(summaries, dim=1)  # (B*T, V, D)
        valid_columns = []
        for name in ordered_names:
            if view_valid is None or name not in view_valid:
                valid = torch.ones(B, dtype=torch.bool, device=summary_stack.device)
            else:
                valid = view_valid[name].to(summary_stack.device).bool().reshape(B)
            valid_columns.append(valid[:, None].expand(B, T).reshape(B * T))
        valid_stack = torch.stack(valid_columns, dim=1)  # (B*T, V)
        summary_fused = self.cross_view_transformer(
            summary_stack,
            src_key_padding_mask=~valid_stack,
        )  # (B*T, V, D)

        # 3. Compute gated view weights
        view_logits = self.view_gate(summary_fused).squeeze(-1)  # (B*T, V)
        view_logits = view_logits.masked_fill(~valid_stack, torch.finfo(view_logits.dtype).min)
        view_weights = torch.softmax(view_logits, dim=-1)        # (B*T, V)

        # 4. Weighted combination of view features
        fused = torch.zeros(B * T, N, D, device=summary_stack.device,
                            dtype=summary_stack.dtype)
        for i in range(V):
            w = view_weights[:, i:i+1].unsqueeze(-1)  # (B*T, 1, 1)
            fused = fused + w * features_list[i]

        fused = self.output_norm(fused)
        return fused.reshape(B, T, N, D)


def build_multi_view_encoder(
    vision_encoder: nn.Module,
    config: dict,
    embed_dim: int,
) -> MultiViewEncoder:
    """
    Factory function: build MultiViewEncoder from config.

    Args:
        vision_encoder: Shared VisionEncoder instance
        config: multi_view config dict from YAML
        embed_dim: Model embedding dimension

    Returns:
        MultiViewEncoder instance
    """
    return MultiViewEncoder(
        vision_encoder=vision_encoder,
        embed_dim=embed_dim,
        num_cross_layers=config.get('num_cross_layers', 2),
        num_heads=config.get('num_heads', 8),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        view_dropout_prob=config.get('view_dropout_prob', 0.5),
        summary_fusion=config.get('summary_fusion', True),
    )
