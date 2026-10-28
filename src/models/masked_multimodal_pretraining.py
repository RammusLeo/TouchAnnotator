"""Masked multimodal self-distillation for video-to-tactile prediction.

This module deliberately wraps the existing TouchAnything model instead of
changing its forward path. During training an EMA teacher sees the clean
video/pose clip, while the student sees spatiotemporal video-tube masks and
joint-time pose masks. The student reconstructs the teacher's tactile-query
latents and remains consistent at the tactile output.

Evaluation bypasses the teacher and masking path, so the wrapped student keeps
the original model's input/output contract.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_masked_pretraining_config(config: dict) -> dict:
    """Return the effective masked-pretraining settings, including defaults."""
    model_cfg = config.get("model", {})
    vision_cfg = model_cfg.get("vision_encoder", {})
    configured = model_cfg.get("masked_pretraining", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise TypeError("model.masked_pretraining must be a mapping")

    embed_dim = int(vision_cfg["embed_dim"])
    defaults = {
        "enabled": True,
        "video_mask_ratio": 0.5,
        "patch_size": int(vision_cfg.get("patch_size", 16)),
        "video_temporal_block_size": 2,
        "video_mask_value": 0.0,
        "pose_mask_ratio": 0.2,
        "pose_temporal_block_size": 2,
        "teacher_momentum": 0.996,
        "predictor_hidden_dim": embed_dim * 2,
        "tactile_consistency_weight": 0.25,
    }
    return {**defaults, **configured}


def _exact_random_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Return an exact-count random boolean mask for every batch element."""
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    num_masked = int(round(num_tokens * mask_ratio))
    if mask_ratio > 0.0:
        num_masked = max(1, num_masked)
    num_masked = min(num_masked, max(num_tokens - 1, 0))
    mask = torch.zeros(
        batch_size,
        num_tokens,
        dtype=torch.bool,
        device=device,
    )
    if num_masked:
        indices = torch.rand(
            batch_size,
            num_tokens,
            device=device,
        ).topk(num_masked, dim=1, largest=False).indices
        mask.scatter_(1, indices, True)
    return mask


def mask_video_tubes(
    video: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    temporal_block_size: int,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask contiguous patch tubes before the Qwen vision attention stack.

    Args:
        video: normalized video with shape ``(B,T,C,H,W)``.

    Returns:
        masked video and a token-grid mask with shape ``(B,T,H/P,W/P)``.
    """
    if video.ndim != 5:
        raise ValueError(
            f"video must have shape (B,T,C,H,W), got {tuple(video.shape)}"
        )
    if patch_size < 1 or temporal_block_size < 1:
        raise ValueError("patch_size and temporal_block_size must be positive")

    batch_size, num_frames, _, height, width = video.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"video size {(height, width)} must be divisible by patch_size={patch_size}"
        )
    grid_height = height // patch_size
    grid_width = width // patch_size
    temporal_groups = math.ceil(num_frames / temporal_block_size)
    group_tokens = temporal_groups * grid_height * grid_width
    group_mask = _exact_random_mask(
        batch_size,
        group_tokens,
        mask_ratio,
        video.device,
    ).reshape(batch_size, temporal_groups, grid_height, grid_width)
    token_mask = group_mask.repeat_interleave(
        temporal_block_size,
        dim=1,
    )[:, :num_frames]
    pixel_mask = token_mask.repeat_interleave(
        patch_size,
        dim=2,
    ).repeat_interleave(
        patch_size,
        dim=3,
    )
    masked_video = video.masked_fill(pixel_mask.unsqueeze(2), float(mask_value))
    return masked_video, token_mask


def mask_pose_tokens(
    poses: torch.Tensor,
    mask_ratio: float,
    temporal_block_size: int,
    mask_token: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask joint-time pose tokens in short temporal spans."""
    if poses.ndim != 4 or poses.shape[-1] != 3:
        raise ValueError(
            f"poses must have shape (B,T,J,3), got {tuple(poses.shape)}"
        )
    if temporal_block_size < 1:
        raise ValueError("temporal_block_size must be positive")

    batch_size, num_frames, num_joints, _ = poses.shape
    temporal_groups = math.ceil(num_frames / temporal_block_size)
    group_mask = _exact_random_mask(
        batch_size,
        temporal_groups * num_joints,
        mask_ratio,
        poses.device,
    ).reshape(batch_size, temporal_groups, num_joints)
    pose_mask = group_mask.repeat_interleave(
        temporal_block_size,
        dim=1,
    )[:, :num_frames]
    replacement = mask_token.to(device=poses.device, dtype=poses.dtype)
    masked_poses = torch.where(
        pose_mask.unsqueeze(-1),
        replacement,
        poses,
    )
    return masked_poses, pose_mask


class MaskedMultimodalPretraining(nn.Module):
    """Add masked video/pose self-distillation around an existing model."""

    def __init__(self, student: nn.Module, config: dict):
        super().__init__()
        ssl_cfg = resolve_masked_pretraining_config(config)
        if not bool(ssl_cfg.get("enabled", True)):
            raise ValueError(
                "MaskedMultimodalPretraining requires "
                "model.masked_pretraining.enabled=true"
            )

        embed_dim = int(config["model"]["vision_encoder"]["embed_dim"])
        hidden_dim = int(ssl_cfg["predictor_hidden_dim"])
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.pose_mask_token = nn.Parameter(torch.zeros(1, 1, 1, 3))

        self.video_mask_ratio = float(ssl_cfg["video_mask_ratio"])
        self.pose_mask_ratio = float(ssl_cfg["pose_mask_ratio"])
        self.patch_size = int(ssl_cfg["patch_size"])
        self.video_temporal_block = int(ssl_cfg["video_temporal_block_size"])
        self.pose_temporal_block = int(ssl_cfg["pose_temporal_block_size"])
        self.video_mask_value = float(ssl_cfg["video_mask_value"])
        self.teacher_momentum = float(ssl_cfg["teacher_momentum"])
        self.tactile_consistency_weight = float(
            ssl_cfg["tactile_consistency_weight"]
        )
        if not 0.0 <= self.teacher_momentum < 1.0:
            raise ValueError(
                f"teacher_momentum must be in [0, 1), got {self.teacher_momentum}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        # The teacher is always deterministic and never receives gradients.
        self.teacher.eval()
        return self

    @torch.no_grad()
    def sync_teacher(self) -> None:
        """Reset the teacher to the current student, e.g. after checkpoint load."""
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.eval()

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA-update trainable teacher tensors after an optimizer step."""
        teacher_parameters = dict(self.teacher.named_parameters())
        for name, student_parameter in self.student.named_parameters():
            if not student_parameter.requires_grad:
                continue
            teacher_parameters[name].lerp_(
                student_parameter.detach(),
                1.0 - self.teacher_momentum,
            )
        # Keep non-parameter state exactly synchronized.
        teacher_buffers = dict(self.teacher.named_buffers())
        for name, student_buffer in self.student.named_buffers():
            teacher_buffers[name].copy_(student_buffer)

    def auxiliary_state_dict(self) -> dict:
        """Return only small SSL state; the EMA teacher is rebuilt on resume."""
        return {
            "predictor": self.predictor.state_dict(),
            "pose_mask_token": self.pose_mask_token.detach(),
            "teacher_momentum": self.teacher_momentum,
        }

    def load_auxiliary_state_dict(self, state: Optional[dict]) -> None:
        if not state:
            self.sync_teacher()
            return
        self.predictor.load_state_dict(state["predictor"], strict=True)
        self.pose_mask_token.data.copy_(
            state["pose_mask_token"].to(
                self.pose_mask_token.device,
                self.pose_mask_token.dtype,
            )
        )
        self.sync_teacher()

    def _masked_inputs(
        self,
        frames: Optional[torch.Tensor],
        views: Optional[Dict[str, torch.Tensor]],
        poses: torch.Tensor,
    ):
        video_ratios = []
        masked_frames = None
        masked_views = None
        if views is not None:
            masked_views = {}
            for name, video in views.items():
                masked_video, token_mask = mask_video_tubes(
                    video,
                    mask_ratio=self.video_mask_ratio,
                    patch_size=self.patch_size,
                    temporal_block_size=self.video_temporal_block,
                    mask_value=self.video_mask_value,
                )
                masked_views[name] = masked_video
                video_ratios.append(token_mask.float().mean())
        elif frames is not None:
            masked_frames, token_mask = mask_video_tubes(
                frames,
                mask_ratio=self.video_mask_ratio,
                patch_size=self.patch_size,
                temporal_block_size=self.video_temporal_block,
                mask_value=self.video_mask_value,
            )
            video_ratios.append(token_mask.float().mean())
        else:
            raise ValueError("Must provide either frames or views")

        masked_poses, pose_mask = mask_pose_tokens(
            poses,
            mask_ratio=self.pose_mask_ratio,
            temporal_block_size=self.pose_temporal_block,
            mask_token=self.pose_mask_token,
        )
        video_ratio = torch.stack(video_ratios).mean()
        return (
            masked_frames,
            masked_views,
            masked_poses,
            video_ratio,
            pose_mask.float().mean(),
        )

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
        **student_kwargs,
    ) -> Dict[str, torch.Tensor]:
        if poses is None:
            raise ValueError("Masked multimodal training requires pose input")

        # Validation/inference retains the exact original model behavior.
        if not self.training:
            return self.student(
                frames=frames,
                poses=poses,
                inter_wrist_delta=inter_wrist_delta,
                inter_wrist_valid=inter_wrist_valid,
                views=views,
                view_valid=view_valid,
                **student_kwargs,
            )

        with torch.no_grad():
            teacher_outputs = self.teacher(
                frames=frames,
                poses=poses,
                inter_wrist_delta=inter_wrist_delta,
                inter_wrist_valid=inter_wrist_valid,
                views=views,
                view_valid=view_valid,
            )

        (
            masked_frames,
            masked_views,
            masked_poses,
            video_mask_ratio,
            pose_mask_ratio,
        ) = self._masked_inputs(frames, views, poses)
        outputs = self.student(
            frames=masked_frames,
            poses=masked_poses,
            inter_wrist_delta=inter_wrist_delta,
            inter_wrist_valid=inter_wrist_valid,
            views=masked_views,
            view_valid=view_valid,
            **student_kwargs,
        )
        if "tactile_queries" not in outputs or "tactile_queries" not in teacher_outputs:
            raise RuntimeError(
                "Masked pretraining requires tactile_queries.enabled=true"
            )

        predicted_queries = self.predictor(outputs["tactile_queries"])
        predicted_queries = F.normalize(predicted_queries.float(), dim=-1)
        target_queries = F.normalize(
            teacher_outputs["tactile_queries"].detach().float(),
            dim=-1,
        )
        query_reconstruction = (
            2.0 - 2.0 * (predicted_queries * target_queries).sum(dim=-1)
        ).mean()

        tactile_consistency = F.smooth_l1_loss(
            outputs["tactile"].float(),
            teacher_outputs["tactile"].detach().float(),
        )
        ssl_loss = (
            query_reconstruction
            + self.tactile_consistency_weight * tactile_consistency
        )
        outputs["ssl_loss"] = ssl_loss
        outputs["ssl"] = {
            "query_reconstruction": query_reconstruction,
            "tactile_consistency": tactile_consistency,
            "video_mask_ratio": video_mask_ratio,
            "pose_mask_ratio": pose_mask_ratio,
        }
        return outputs
