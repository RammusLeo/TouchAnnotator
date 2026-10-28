"""Hand-centric ROI selection, tactile queries, and tri-modal alignment."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HAND_NAMES = ("left", "right")
HAND_VIEW_ROUTES = {
    0: ("ego", "wrist_left"),
    1: ("ego", "wrist_right"),
}


class _QueryFusionLayer(nn.Module):
    """Pose and ROI-biased visual attention for tactile query tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        use_pose: bool = True,
    ):
        super().__init__()
        self.use_pose = bool(use_pose)
        if self.use_pose:
            self.pose_norm = nn.LayerNorm(embed_dim)
            self.pose_attn = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.visual_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_dim = int(embed_dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        pose_tokens: Optional[torch.Tensor],
        visual_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_pose:
            if pose_tokens is None:
                raise ValueError("Pose-conditioned query fusion requires pose tokens")
            query_norm = self.pose_norm(queries)
            update, _ = self.pose_attn(
                query=query_norm,
                key=pose_tokens,
                value=pose_tokens,
                need_weights=False,
            )
            queries = queries + update

        query_norm = self.visual_norm(queries)
        update, _ = self.visual_attn(
            query=query_norm,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=visual_padding_mask,
            need_weights=False,
        )
        queries = queries + update

        query_norm = self.self_norm(queries)
        update, _ = self.self_attn(
            query=query_norm,
            key=query_norm,
            value=query_norm,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


class TactileQueryFusion(nn.Module):
    """Fuse optional MANO joint tokens with hand-local visual ROI tokens.

    The egocentric stream is shared by both hands. A left tactile query may
    additionally attend to the left wrist stream, while a right query may
    attend to the right wrist stream. ROI logits use the corresponding hand
    pose when enabled, or learned hand-specific seeds in video-only mode, and
    can be supervised by EgoTouch glove masks.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_queries_per_hand: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        roi_floor: float = 0.05,
        use_pose: bool = True,
    ):
        super().__init__()
        query_grid = math.isqrt(num_queries_per_hand)
        if query_grid * query_grid != num_queries_per_hand:
            raise ValueError(
                "num_queries_per_hand must be a square number, "
                f"got {num_queries_per_hand}"
            )
        self.embed_dim = embed_dim
        self.num_queries_per_hand = num_queries_per_hand
        self.query_grid = query_grid
        self.roi_floor = float(roi_floor)
        self.use_pose = bool(use_pose)

        self.query_embed = nn.Parameter(
            torch.randn(2, num_queries_per_hand, embed_dim) * 0.02
        )
        self.hand_embed = nn.Parameter(torch.randn(2, 1, embed_dim) * 0.02)
        self.roi_query_proj = nn.Linear(embed_dim, embed_dim)
        self.roi_key_proj = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                _QueryFusionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_pose=self.use_pose,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _view_valid_mask(
        view_name: str,
        view_valid: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if view_valid is None or view_name not in view_valid:
            valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            valid = view_valid[view_name].to(device).bool().reshape(batch_size)
        return valid[:, None].expand(batch_size, num_frames).reshape(-1)

    def forward(
        self,
        view_features: Dict[str, torch.Tensor],
        pose_features: Optional[torch.Tensor] = None,
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if "ego" not in view_features:
            raise ValueError("Tactile query fusion requires the ego view")

        ego_features = view_features["ego"]
        if ego_features.ndim != 4:
            raise ValueError(
                "Tactile queries require ego features with shape (B,T,N,D), "
                f"got {tuple(ego_features.shape)}"
            )
        batch_size, num_frames, _, embed_dim = ego_features.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Visual feature width {embed_dim} != query width {self.embed_dim}"
            )
        if self.use_pose:
            if (
                pose_features is None
                or pose_features.ndim != 4
                or pose_features.shape[2] % 2
            ):
                shape = None if pose_features is None else tuple(pose_features.shape)
                raise ValueError(
                    "Pose-conditioned tactile queries require per-joint features "
                    f"with shape (B,T,2J,D), got {shape}"
                )
            if pose_features.shape[:2] != (batch_size, num_frames):
                raise ValueError(
                    "Pose and visual features must share batch/time dimensions"
                )
            num_joints = pose_features.shape[2]
            joints_per_hand = num_joints // 2
        batch_time = batch_size * num_frames

        fused_hands = []
        visual_contexts = []
        pose_contexts = []
        ego_roi_logits = []

        for hand_index in range(2):
            if self.use_pose:
                pose_tokens = pose_features[
                    :, :, hand_index * joints_per_hand:(hand_index + 1) * joints_per_hand
                ].reshape(batch_time, joints_per_hand, embed_dim)
                pose_summary = pose_tokens.mean(dim=1)
                roi_seed = pose_summary
            else:
                pose_tokens = None
                pose_summary = None
                # Learned, hand-specific visual queries replace MANO-conditioned
                # ROI localization in the video-only ablation.
                roi_seed = (
                    self.query_embed[hand_index].mean(dim=0)
                    + self.hand_embed[hand_index, 0]
                ).unsqueeze(0).expand(batch_time, -1)
            roi_query = self.roi_query_proj(roi_seed)

            routed_tokens = []
            routed_padding = []
            routed_logits = []
            routed_names = []
            for view_name in HAND_VIEW_ROUTES[hand_index]:
                if view_name not in view_features:
                    continue
                features = view_features[view_name]
                if features.shape[:2] != (batch_size, num_frames):
                    raise ValueError(
                        f"View {view_name} has incompatible shape {tuple(features.shape)}"
                    )
                tokens = features.reshape(batch_time, features.shape[2], embed_dim)
                token_keys = self.roi_key_proj(tokens)
                roi_logits = torch.einsum(
                    "bd,bnd->bn",
                    roi_query,
                    token_keys,
                ) / math.sqrt(embed_dim)
                valid = self._view_valid_mask(
                    view_name,
                    view_valid,
                    batch_size,
                    num_frames,
                    tokens.device,
                )
                padding = (~valid)[:, None].expand_as(roi_logits)
                roi_gate = torch.sigmoid(roi_logits).unsqueeze(-1)
                gated_tokens = tokens * (
                    self.roi_floor + (1.0 - self.roi_floor) * roi_gate
                )

                routed_tokens.append(gated_tokens)
                routed_padding.append(padding)
                routed_logits.append(roi_logits)
                routed_names.append(view_name)

            if not routed_tokens:
                raise ValueError(
                    f"No routed visual view is available for {HAND_NAMES[hand_index]} hand"
                )

            visual_tokens = torch.cat(routed_tokens, dim=1)
            visual_padding_mask = torch.cat(routed_padding, dim=1)
            if bool(visual_padding_mask.all(dim=1).any().item()):
                raise ValueError(
                    f"At least one sample has no valid visual view for "
                    f"{HAND_NAMES[hand_index]} hand"
                )

            all_roi_logits = torch.cat(routed_logits, dim=1)
            all_roi_logits = all_roi_logits.masked_fill(
                visual_padding_mask,
                torch.finfo(all_roi_logits.dtype).min,
            )
            roi_weights = torch.softmax(all_roi_logits, dim=-1)
            visual_summary = torch.einsum(
                "bn,bnd->bd",
                roi_weights,
                torch.cat(
                    [
                        view_features[name].reshape(
                            batch_time,
                            view_features[name].shape[2],
                            embed_dim,
                        )
                        for name in routed_names
                    ],
                    dim=1,
                ),
            )

            queries = (
                self.query_embed[hand_index]
                + self.hand_embed[hand_index]
            ).unsqueeze(0).expand(batch_time, -1, -1)
            for layer in self.layers:
                queries = layer(
                    queries,
                    pose_tokens,
                    visual_tokens,
                    visual_padding_mask,
                )
            queries = self.output_norm(queries)

            fused_hands.append(
                queries.reshape(
                    batch_size,
                    num_frames,
                    self.num_queries_per_hand,
                    embed_dim,
                )
            )
            visual_contexts.append(
                visual_summary.reshape(batch_size, num_frames, embed_dim)
            )
            if pose_summary is not None:
                pose_contexts.append(
                    pose_summary.reshape(batch_size, num_frames, embed_dim)
                )

            if "ego" in routed_names:
                ego_index = routed_names.index("ego")
                ego_roi_logits.append(
                    routed_logits[ego_index].reshape(
                        batch_size,
                        num_frames,
                        -1,
                    )
                )

        output = {
            "queries": torch.stack(fused_hands, dim=2),
            "visual_context": torch.stack(visual_contexts, dim=2),
        }
        if self.use_pose:
            output["pose_context"] = torch.stack(pose_contexts, dim=2)
        if len(ego_roi_logits) == 2:
            output["ego_roi_logits"] = torch.stack(ego_roi_logits, dim=2)
        return output


def roi_supervision_loss(
    roi_logits: torch.Tensor,
    roi_masks: torch.Tensor,
    roi_valid: torch.Tensor,
    dilation_tokens: int = 1,
) -> torch.Tensor:
    """BCE + Dice supervision for hand-local Qwen token selection."""
    batch_size, num_frames, num_hands, num_tokens = roi_logits.shape
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"ROI token count must be square, got {num_tokens}")

    targets = roi_masks.to(roi_logits.device, roi_logits.dtype).reshape(
        batch_size * num_frames * num_hands,
        1,
        roi_masks.shape[-2],
        roi_masks.shape[-1],
    )
    targets = F.interpolate(
        targets,
        size=(grid_size, grid_size),
        mode="nearest",
    )
    if dilation_tokens > 0:
        kernel = 2 * dilation_tokens + 1
        targets = F.max_pool2d(
            targets,
            kernel_size=kernel,
            stride=1,
            padding=dilation_tokens,
        )
    targets = targets.reshape_as(roi_logits)

    valid = roi_valid.to(roi_logits.device).bool().reshape(
        batch_size,
        num_frames,
        num_hands,
        1,
    )
    valid_float = valid.to(roi_logits.dtype).expand_as(roi_logits)
    if valid_float.sum().detach().item() == 0:
        return roi_logits.sum() * 0.0

    positive = (targets * valid_float).sum()
    negative = ((1.0 - targets) * valid_float).sum()
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        roi_logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    bce = (bce * valid_float).sum() / valid_float.sum().clamp_min(1.0)

    probabilities = torch.sigmoid(roi_logits)
    intersection = (probabilities * targets * valid_float).sum(dim=-1)
    denominator = (
        ((probabilities + targets) * valid_float).sum(dim=-1)
    ).clamp_min(1e-6)
    valid_hands = valid.squeeze(-1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice = dice[valid_hands].mean()
    return bce + dice


class TactileQueryDecoder(nn.Module):
    """Decode canonical per-hand query grids into pressure and contact maps.

    The two tasks share the spatial decoder trunk but have independent output
    projections.  In particular, binary-contact BCE is never applied to the
    pressure logits, so it cannot directly force the continuous-pressure head
    toward a binary representation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_queries_per_hand: int = 64,
        hidden_channels: int = 128,
        tactile_size: int = 21,
        dropout: float = 0.1,
    ):
        super().__init__()
        query_grid = math.isqrt(num_queries_per_hand)
        if query_grid * query_grid != num_queries_per_hand:
            raise ValueError("num_queries_per_hand must form a square grid")
        self.query_grid = query_grid
        self.tactile_size = tactile_size
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.GELU(),
        )
        output_channels = hidden_channels // 2
        self.pressure_head = nn.Conv2d(
            output_channels,
            1,
            3,
            padding=1,
        )
        self.contact_head = nn.Conv2d(
            output_channels,
            1,
            3,
            padding=1,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate a legacy shared ``head`` into both task-specific heads."""
        for parameter_name in ("weight", "bias"):
            legacy_key = f"{prefix}head.{parameter_name}"
            if legacy_key not in state_dict:
                continue
            legacy_value = state_dict.pop(legacy_key)
            state_dict.setdefault(
                f"{prefix}pressure_head.{parameter_name}",
                legacy_value,
            )
            state_dict.setdefault(
                f"{prefix}contact_head.{parameter_name}",
                legacy_value.clone(),
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        queries: torch.Tensor,
        return_logits: bool = False,
        return_outputs: bool = False,
    ):
        batch_size, num_frames, num_hands, num_queries, _ = queries.shape
        features = self.input_proj(queries)
        features = features.reshape(
            batch_size * num_frames * num_hands,
            self.query_grid,
            self.query_grid,
            -1,
        ).permute(0, 3, 1, 2)
        features = self.refine(features)
        features = F.interpolate(
            features,
            size=(self.tactile_size, self.tactile_size),
            mode="bilinear",
            align_corners=False,
        )
        output_shape = (
            batch_size,
            num_frames,
            num_hands,
            self.tactile_size,
            self.tactile_size,
        )
        pressure_logits = self.pressure_head(features).reshape(output_shape)
        contact_logits = self.contact_head(features).reshape(output_shape)
        pressure = torch.sigmoid(pressure_logits)
        contact = torch.sigmoid(contact_logits)
        if return_outputs:
            return {
                "pressure": pressure,
                "pressure_logits": pressure_logits,
                "contact": contact,
                "contact_logits": contact_logits,
            }
        if return_logits:
            # Preserve the original two-value decoder contract for callers
            # that only request the continuous-pressure pre-sigmoid values.
            return pressure, pressure_logits
        return pressure


class _TactileTeacherEncoder(nn.Module):
    """Encode ground-truth touch while retaining a reconstruction constraint."""

    def __init__(self, token_dim: int, query_grid: int):
        super().__init__()
        hidden = max(token_dim // 2, 32)
        self.query_grid = query_grid
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, token_dim, 3, padding=1),
            nn.GELU(),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(token_dim, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(
        self,
        tactile: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_hands, height, width = tactile.shape
        tactile_flat = tactile.reshape(-1, 1, height, width)
        mask_flat = mask.reshape(-1, 1, height, width)
        features = self.encoder(torch.cat([tactile_flat * mask_flat, mask_flat], dim=1))
        features = F.adaptive_avg_pool2d(
            features,
            (self.query_grid, self.query_grid),
        )
        reconstruction = torch.sigmoid(self.reconstruction_head(features))
        reconstruction = F.interpolate(
            reconstruction,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        tokens = features.flatten(2).transpose(1, 2).reshape(
            batch_size,
            num_frames,
            num_hands,
            self.query_grid * self.query_grid,
            -1,
        )
        reconstruction = reconstruction.reshape(
            batch_size,
            num_frames,
            num_hands,
            height,
            width,
        )
        return tokens, reconstruction


class TriModalAlignmentHead(nn.Module):
    """SigLIP-style alignment of vision, optional hand pose, and touch."""

    def __init__(
        self,
        embed_dim: int = 768,
        alignment_dim: int = 256,
        num_queries_per_hand: int = 64,
        contact_threshold: float = 0.1,
        visual_touch_weight: float = 0.25,
        pose_touch_weight: float = 0.25,
        visual_pose_weight: float = 0.1,
        reconstruction_weight: float = 0.25,
        use_pose: bool = True,
    ):
        super().__init__()
        query_grid = math.isqrt(num_queries_per_hand)
        self.contact_threshold = float(contact_threshold)
        self.visual_touch_weight = float(visual_touch_weight)
        self.pose_touch_weight = float(pose_touch_weight)
        self.visual_pose_weight = float(visual_pose_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.use_pose = bool(use_pose)

        teacher_dim = max(alignment_dim, 64)
        self.tactile_teacher = _TactileTeacherEncoder(teacher_dim, query_grid)
        self.fused_proj = nn.Linear(embed_dim, alignment_dim)
        self.visual_proj = nn.Linear(embed_dim, alignment_dim)
        if self.use_pose:
            self.pose_proj = nn.Linear(embed_dim, alignment_dim)
        self.tactile_proj = nn.Linear(teacher_dim, alignment_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))

    def _siglip_pair(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        first = F.normalize(first.reshape(-1, first.shape[-1]), dim=-1)
        second = F.normalize(second.reshape(-1, second.shape[-1]), dim=-1)
        active = active.reshape(-1)
        first = first[active]
        second = second[active]
        if first.shape[0] == 0:
            # A no-contact batch has no valid SigLIP pairs, but every DDP rank
            # must still mark both projection branches and the learned
            # temperature/bias as participating in the loss. Preserve an
            # explicit zero-valued dependency instead of dropping those
            # parameters from the autograd graph.
            return (
                first.sum()
                + second.sum()
                + self.logit_scale
                + self.logit_bias
            ) * 0.0

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * first @ second.transpose(0, 1) + self.logit_bias
        positive_mask = torch.eye(
            logits.shape[0],
            dtype=torch.bool,
            device=logits.device,
        )
        positive_loss = -F.logsigmoid(logits[positive_mask]).mean()
        if logits.shape[0] == 1:
            return positive_loss
        negative_loss = -F.logsigmoid(-logits[~positive_mask]).mean()
        # Equal positive/negative contributions prevent the O(M^2) negatives
        # from making an all-negative solution attractive at small batch sizes.
        return 0.5 * (positive_loss + negative_loss)

    def forward(
        self,
        fused_queries: torch.Tensor,
        visual_context: torch.Tensor,
        pose_context: Optional[torch.Tensor],
        tactile_targets: torch.Tensor,
        tactile_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if tactile_mask is None:
            tactile_mask = torch.ones_like(tactile_targets)
        tactile_mask = tactile_mask.to(
            tactile_targets.device,
            tactile_targets.dtype,
        )
        tactile_tokens, reconstruction = self.tactile_teacher(
            tactile_targets,
            tactile_mask,
        )

        # Contact-weighted temporal pooling prevents the many no-contact frames
        # in a clip from washing out the alignment signal. Ground truth is used
        # only inside this training-time auxiliary head.
        frame_contact = (tactile_targets * tactile_mask).amax(dim=(3, 4))
        temporal_weights = frame_contact / frame_contact.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        fused_per_frame = fused_queries.mean(dim=3)
        tactile_per_frame = tactile_tokens.mean(dim=3)
        fused_embedding = self.fused_proj(
            (fused_per_frame * temporal_weights.unsqueeze(-1)).sum(dim=1)
        )
        visual_embedding = self.visual_proj(
            (visual_context * temporal_weights.unsqueeze(-1)).sum(dim=1)
        )
        tactile_embedding = self.tactile_proj(
            (tactile_per_frame * temporal_weights.unsqueeze(-1)).sum(dim=1)
        )

        active = frame_contact.amax(dim=1) > self.contact_threshold
        fused_touch = self._siglip_pair(
            fused_embedding,
            tactile_embedding,
            active,
        )
        visual_touch = self._siglip_pair(
            visual_embedding,
            tactile_embedding,
            active,
        )
        reconstruction_error = (reconstruction - tactile_targets).square()
        reconstruction_loss = (
            reconstruction_error * tactile_mask
        ).sum() / tactile_mask.sum().clamp_min(1.0)

        total = (
            fused_touch
            + self.visual_touch_weight * visual_touch
            + self.reconstruction_weight * reconstruction_loss
        )
        output = {
            "loss": total,
            "fused_touch": fused_touch,
            "visual_touch": visual_touch,
            "tactile_reconstruction": reconstruction_loss,
            "active_hands": active.to(tactile_targets.dtype).sum(),
        }
        if self.use_pose:
            if pose_context is None:
                raise ValueError("Pose alignment requires pose context")
            pose_embedding = self.pose_proj(
                (pose_context * temporal_weights.unsqueeze(-1)).sum(dim=1)
            )
            pose_touch = self._siglip_pair(
                pose_embedding,
                tactile_embedding,
                active,
            )
            visual_pose = self._siglip_pair(
                visual_embedding,
                pose_embedding,
                active,
            )
            output["pose_touch"] = pose_touch
            output["visual_pose"] = visual_pose
            output["loss"] = (
                output["loss"]
                + self.pose_touch_weight * pose_touch
                + self.visual_pose_weight * visual_pose
            )
        return output
