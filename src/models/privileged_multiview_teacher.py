"""Hybrid asymmetric distillation for dual-branch tactile pretraining.

The deployed student keeps the ordinary :class:`TouchAnything` interface.
During training, an EMA teacher observes clean inputs while the student sees a
moderately degraded video stream.  Samples with real wrist cameras use a
full-view privileged teacher; single-view HOI samples use a clean ego-view
teacher.  This extends the useful clean/degraded idea from visual on-policy
distillation to dense tactile prediction without pretending that the dense
heads perform autoregressive on-policy rollout.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIVILEGED_VIEW_NAMES = ("wrist_left", "wrist_right")
DISTILLATION_OUTPUT_KEYS = ("tactile_queries", "tactile", "contact_logits")
TEACHER_CONTEXT_KEYS = (
    "history_frames",
    "history_poses",
    "history_inter_wrist_delta",
    "history_inter_wrist_valid",
    "history_views",
    "history_view_valid",
    "history_valid",
)


def resolve_privileged_teacher_config(config: dict) -> dict:
    """Return validated teacher/student settings with stable defaults."""
    model_cfg = config.get("model", {})
    configured = model_cfg.get("privileged_multiview_teacher", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise TypeError("model.privileged_multiview_teacher must be a mapping")

    vision_cfg = model_cfg.get("vision_encoder", {})
    if "embed_dim" not in vision_cfg:
        raise KeyError(
            "model.vision_encoder.embed_dim is required by the privileged teacher"
        )
    embed_dim = int(vision_cfg["embed_dim"])
    defaults = {
        "enabled": False,
        "teacher_momentum": 0.996,
        "frozen_teacher": False,
        "teacher_checkpoint": None,
        "predictor_hidden_dim": embed_dim * 2,
        # View removal is intentionally milder than in the first version: the
        # pixel degradation below already creates a meaningful teacher gap.
        "ego_only_probability": 0.25,
        "wrist_drop_probability": 0.25,
        # Single-view HOI data now receives clean-teacher supervision too.
        "distill_clean_ego": True,
        "teacher_ego_only": False,
        "student_augmentation_probability": 0.80,
        "student_downscale_min": 0.72,
        "student_downscale_max": 0.90,
        "student_noise_probability": 0.50,
        "student_noise_std_min": 0.02,
        "student_noise_std_max": 0.06,
        # Student-only hand-pose corruption. Noise and dropout are sampled as
        # mutually exclusive per-sample modes and shared by the current clip
        # and all raw history clips. Values are in the pose input's native
        # coordinate units (metres for wrist_relative_m configs).
        "student_pose_noise_probability": 0.0,
        "student_pose_dropout_probability": 0.0,
        "student_pose_noise_std_min": 0.0,
        "student_pose_noise_std_max": 0.0,
        "student_pose_preserve_wrist_origin": True,
        # Bounds cover the full ImageNet-normalized RGB range used by every
        # source in the current data mixture.
        "student_input_clip_min": -2.20,
        "student_input_clip_max": 2.70,
        # Reject both trivial teacher/student pairs and unstable outliers.
        "gap_gate_enabled": True,
        "query_gap_min": 0.002,
        "query_gap_max": 0.50,
        "contact_gap_min": 1.0e-5,
        "contact_gap_max": 0.10,
        "contact_teacher_confidence": 0.60,
        "query_weight": 1.0,
        "pressure_weight": 0.50,
        "contact_weight": 0.50,
        "contact_temperature": 1.0,
    }
    resolved = {**defaults, **configured}
    if bool(resolved["frozen_teacher"]) and not resolved["teacher_checkpoint"]:
        raise ValueError("frozen_teacher requires teacher_checkpoint")
    if resolved["teacher_checkpoint"] and not bool(resolved["frozen_teacher"]):
        raise ValueError("teacher_checkpoint requires frozen_teacher=true")

    for name in (
        "ego_only_probability",
        "wrist_drop_probability",
        "student_augmentation_probability",
        "student_noise_probability",
        "student_pose_noise_probability",
        "student_pose_dropout_probability",
        "contact_teacher_confidence",
    ):
        value = float(resolved[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    momentum = float(resolved["teacher_momentum"])
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"teacher_momentum must be in [0, 1), got {momentum}")
    temperature = float(resolved["contact_temperature"])
    if temperature <= 0.0:
        raise ValueError(
            f"contact_temperature must be positive, got {temperature}"
        )
    for minimum, maximum in (
        ("student_downscale_min", "student_downscale_max"),
        ("student_noise_std_min", "student_noise_std_max"),
        ("query_gap_min", "query_gap_max"),
        ("contact_gap_min", "contact_gap_max"),
    ):
        low = float(resolved[minimum])
        high = float(resolved[maximum])
        if low < 0.0 or high < low:
            raise ValueError(
                f"Expected 0 <= {minimum} <= {maximum}, got {low} and {high}"
            )
    if float(resolved["student_downscale_max"]) > 1.0:
        raise ValueError("student_downscale_max must not exceed 1.0")
    pose_corruption_probability = (
        float(resolved["student_pose_noise_probability"])
        + float(resolved["student_pose_dropout_probability"])
    )
    if pose_corruption_probability > 1.0:
        raise ValueError(
            "student_pose_noise_probability + "
            "student_pose_dropout_probability must not exceed 1"
        )
    pose_noise_min = float(resolved["student_pose_noise_std_min"])
    pose_noise_max = float(resolved["student_pose_noise_std_max"])
    if pose_noise_min < 0.0 or pose_noise_max < pose_noise_min:
        raise ValueError(
            "Expected 0 <= student_pose_noise_std_min <= "
            f"student_pose_noise_std_max, got {pose_noise_min} and "
            f"{pose_noise_max}"
        )
    clip_min = float(resolved["student_input_clip_min"])
    clip_max = float(resolved["student_input_clip_max"])
    if clip_max <= clip_min:
        raise ValueError("student_input_clip_max must exceed student_input_clip_min")
    for name in ("query_weight", "pressure_weight", "contact_weight"):
        if float(resolved[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if int(resolved["predictor_hidden_dim"]) < 1:
        raise ValueError("predictor_hidden_dim must be positive")
    return resolved


def _batch_validity(
    view_name: str,
    views: Dict[str, torch.Tensor],
    view_valid: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve one view's per-sample availability to a boolean vector."""
    if view_name not in views:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if view_valid is None or view_name not in view_valid:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    validity = view_valid[view_name].to(device=device, dtype=torch.bool)
    if validity.numel() != batch_size:
        raise ValueError(
            f"view_valid[{view_name!r}] must contain {batch_size} values, "
            f"got shape {tuple(validity.shape)}"
        )
    return validity.reshape(batch_size)


def _select_batch(tensor: Optional[torch.Tensor], mask: torch.Tensor):
    if tensor is None:
        return None
    if tensor.ndim < 1 or tensor.shape[0] != mask.shape[0]:
        raise ValueError(
            "Teacher inputs must share the video batch dimension; "
            f"got tensor {tuple(tensor.shape)} and mask {tuple(mask.shape)}"
        )
    return tensor[mask]


def _select_batch_mapping(
    values: Optional[Dict[str, torch.Tensor]],
    mask: torch.Tensor,
    selected_names: Optional[tuple[str, ...]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    if values is None:
        return None
    return {
        name: _select_batch(value, mask)
        for name, value in values.items()
        if selected_names is None or name in selected_names
    }


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=reference.device, dtype=reference.dtype)
    try:
        return torch.broadcast_to(weights, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f"Distillation mask {tuple(weights.shape)} cannot broadcast to "
            f"prediction {tuple(reference.shape)}"
        ) from error


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = _broadcast_mask(mask, values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_mean_per_sample(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a masked mean and support flag for each batch element."""
    if values.ndim < 1:
        raise ValueError("Per-sample reduction requires a batch dimension")
    weights = torch.ones_like(values) if mask is None else _broadcast_mask(mask, values)
    flat_values = values.reshape(values.shape[0], -1)
    flat_weights = weights.reshape(values.shape[0], -1)
    denominator = flat_weights.sum(dim=1)
    means = (flat_values * flat_weights).sum(dim=1) / denominator.clamp_min(1.0)
    return means, denominator > 0


def _task_mask(
    sensor_mask: Optional[torch.Tensor],
    availability: Optional[torch.Tensor],
    active_samples: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Combine cell validity, task-domain availability, and sample activity."""
    batch_size = reference.shape[0]
    if active_samples.numel() != batch_size:
        raise ValueError("active_samples must match the prediction batch")
    combined = active_samples.to(reference).reshape(
        batch_size,
        *([1] * (reference.ndim - 1)),
    )
    if sensor_mask is not None:
        if sensor_mask.shape[0] != batch_size:
            raise ValueError("tactile_mask must match the prediction batch")
        combined = combined * sensor_mask.to(reference)
    if availability is not None:
        if availability.numel() != batch_size:
            raise ValueError("distillation availability must match the batch")
        available = availability.to(reference).reshape(
            batch_size,
            *([1] * (reference.ndim - 1)),
        )
        combined = combined * available
    return combined


def bernoulli_js_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise symmetric JSD between Bernoulli contact predictions.

    The returned teacher probability is detached and can be reused for the
    confidence mask.  With temperature one, each JSD value is bounded by
    ``log(2)`` and is zero exactly when the probabilities match.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_probability = torch.sigmoid(student_logits.float() / temperature)
    teacher_probability = torch.sigmoid(
        teacher_logits.detach().float() / temperature
    )
    student_probability = student_probability.clamp(eps, 1.0 - eps)
    teacher_probability = teacher_probability.clamp(eps, 1.0 - eps)
    mixture = 0.5 * (student_probability + teacher_probability)

    def _bernoulli_kl(probability: torch.Tensor) -> torch.Tensor:
        return (
            probability * (probability.log() - mixture.log())
            + (1.0 - probability)
            * ((1.0 - probability).log() - (1.0 - mixture).log())
        )

    divergence = 0.5 * (
        _bernoulli_kl(teacher_probability)
        + _bernoulli_kl(student_probability)
    )
    return divergence * (temperature**2), teacher_probability.detach()


class PrivilegedMultiViewTeacher(nn.Module):
    """Train an inference-compatible student from clean EMA targets."""

    def __init__(self, student: nn.Module, config: dict):
        super().__init__()
        teacher_cfg = resolve_privileged_teacher_config(config)
        if not bool(teacher_cfg["enabled"]):
            raise ValueError(
                "PrivilegedMultiViewTeacher requires "
                "model.privileged_multiview_teacher.enabled=true"
            )
        task_model = getattr(student, "task_model", student)
        if not bool(getattr(task_model, "multi_view_enabled", False)):
            raise ValueError("Privileged multi-view training requires multi_view.enabled=true")
        if not bool(getattr(task_model, "query_fusion_enabled", False)):
            raise ValueError(
                "Privileged multi-view training requires tactile_queries.enabled=true"
            )

        # The wrapper owns per-sample view removal.  Disable the legacy
        # whole-batch dropout so the teacher/student contract stays explicit.
        if hasattr(task_model, "multi_view_encoder"):
            task_model.multi_view_encoder.view_dropout_prob = 0.0

        self.student = student
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        embed_dim = int(config["model"]["vision_encoder"]["embed_dim"])
        hidden_dim = int(teacher_cfg["predictor_hidden_dim"])
        self.query_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.teacher_momentum = float(teacher_cfg["teacher_momentum"])
        self.frozen_teacher = bool(teacher_cfg["frozen_teacher"])
        self.teacher_checkpoint = teacher_cfg["teacher_checkpoint"]
        self._frozen_teacher_loaded = False
        self.ego_only_probability = float(teacher_cfg["ego_only_probability"])
        self.wrist_drop_probability = float(
            teacher_cfg["wrist_drop_probability"]
        )
        self.distill_clean_ego = bool(teacher_cfg["distill_clean_ego"])
        self.teacher_ego_only = bool(teacher_cfg["teacher_ego_only"])
        self.student_augmentation_probability = float(
            teacher_cfg["student_augmentation_probability"]
        )
        self.student_downscale_min = float(teacher_cfg["student_downscale_min"])
        self.student_downscale_max = float(teacher_cfg["student_downscale_max"])
        self.student_noise_probability = float(
            teacher_cfg["student_noise_probability"]
        )
        self.student_noise_std_min = float(teacher_cfg["student_noise_std_min"])
        self.student_noise_std_max = float(teacher_cfg["student_noise_std_max"])
        self.student_pose_noise_probability = float(
            teacher_cfg["student_pose_noise_probability"]
        )
        self.student_pose_dropout_probability = float(
            teacher_cfg["student_pose_dropout_probability"]
        )
        self.student_pose_noise_std_min = float(
            teacher_cfg["student_pose_noise_std_min"]
        )
        self.student_pose_noise_std_max = float(
            teacher_cfg["student_pose_noise_std_max"]
        )
        self.student_pose_preserve_wrist_origin = bool(
            teacher_cfg["student_pose_preserve_wrist_origin"]
        )
        self.student_input_clip_min = float(teacher_cfg["student_input_clip_min"])
        self.student_input_clip_max = float(teacher_cfg["student_input_clip_max"])
        self.gap_gate_enabled = bool(teacher_cfg["gap_gate_enabled"])
        self.query_gap_min = float(teacher_cfg["query_gap_min"])
        self.query_gap_max = float(teacher_cfg["query_gap_max"])
        self.contact_gap_min = float(teacher_cfg["contact_gap_min"])
        self.contact_gap_max = float(teacher_cfg["contact_gap_max"])
        self.contact_teacher_confidence = float(
            teacher_cfg["contact_teacher_confidence"]
        )
        self.query_weight = float(teacher_cfg["query_weight"])
        self.pressure_weight = float(teacher_cfg["pressure_weight"])
        self.contact_weight = float(teacher_cfg["contact_weight"])
        self.contact_temperature = float(teacher_cfg["contact_temperature"])

        vision_encoder = getattr(task_model, "vision_encoder", None)
        patch_size = int(getattr(vision_encoder, "patch_size", 1))
        spatial_merge = int(getattr(vision_encoder, "spatial_merge_size", 1))
        self.student_spatial_factor = max(patch_size * spatial_merge, 1)
        if self.frozen_teacher:
            self.sync_teacher()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def sync_teacher(self) -> None:
        """Restore a fixed teacher or synchronize an EMA teacher with the student."""
        if self.frozen_teacher:
            if not self._frozen_teacher_loaded:
                # mmap avoids materializing the checkpoint's optimizer tensors.
                checkpoint = torch.load(
                    self.teacher_checkpoint, map_location="cpu",
                    weights_only=False, mmap=True,
                )
                state = checkpoint.get("model_state_dict", checkpoint)
                while state and all(key.startswith("module.") for key in state):
                    state = {key[len("module."):]: value for key, value in state.items()}
                self.teacher.load_state_dict(state, strict=True)
                self._frozen_teacher_loaded = True
            self.teacher.eval()
            return
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.eval()

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA-update trainable tensors and exactly synchronize buffers."""
        if self.frozen_teacher:
            return
        teacher_parameters = dict(self.teacher.named_parameters())
        for name, student_parameter in self.student.named_parameters():
            if not student_parameter.requires_grad:
                continue
            teacher_parameters[name].lerp_(
                student_parameter.detach(),
                1.0 - self.teacher_momentum,
            )
        teacher_buffers = dict(self.teacher.named_buffers())
        for name, student_buffer in self.student.named_buffers():
            teacher_buffers[name].copy_(student_buffer)

    def auxiliary_state_dict(self) -> dict:
        """Save the predictor; rebuild the large EMA copy on resume."""
        return {
            "query_predictor": self.query_predictor.state_dict(),
            "teacher_momentum": self.teacher_momentum,
        }

    def load_auxiliary_state_dict(self, state: Optional[dict]) -> None:
        if state:
            self.query_predictor.load_state_dict(
                state["query_predictor"],
                strict=True,
            )
        self.sync_teacher()

    def _privileged_mask(
        self,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        reference = views.get("ego", next(iter(views.values())))
        batch_size = reference.shape[0]
        mask = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
        for view_name in PRIVILEGED_VIEW_NAMES:
            mask |= _batch_validity(
                view_name,
                views,
                view_valid,
                batch_size,
                reference.device,
            )
        return mask

    def _degraded_view_validity(
        self,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]],
        privileged_mask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = privileged_mask.shape[0]
        device = privileged_mask.device
        student_valid = {
            name: _batch_validity(
                name,
                views,
                view_valid,
                batch_size,
                device,
            )
            for name in views
        }
        force_ego_only = (
            torch.rand(batch_size, device=device) < self.ego_only_probability
        ) & privileged_mask
        retained_wrist = torch.zeros(batch_size, device=device)
        available_wrist = torch.zeros(batch_size, device=device)
        for view_name in PRIVILEGED_VIEW_NAMES:
            if view_name not in student_valid:
                continue
            original = student_valid[view_name]
            keep = torch.rand(batch_size, device=device) >= self.wrist_drop_probability
            degraded = original & keep & ~force_ego_only
            student_valid[view_name] = degraded
            retained_wrist += degraded.float()
            available_wrist += original.float()
        retained_ratio = retained_wrist.sum() / available_wrist.sum().clamp_min(1.0)
        return student_valid, force_ego_only, retained_ratio

    @staticmethod
    def _resize_video(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if video.shape[-2:] == (height, width):
            return video
        if video.ndim != 5:
            raise ValueError(
                "Student video degradation expects (B, T, C, H, W), got "
                f"{tuple(video.shape)}"
            )
        batch_size, num_frames, channels, _, _ = video.shape
        resized = F.interpolate(
            video.reshape(batch_size * num_frames, channels, *video.shape[-2:]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, num_frames, channels, height, width)

    def _degrade_student_views(
        self,
        views: Dict[str, torch.Tensor],
        student_valid: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], dict]:
        """Apply batch-shared resizing and per-sample sensor noise."""
        reference = views["ego"]
        device = reference.device
        batch_size = reference.shape[0]
        augment = bool(
            (
                torch.rand((), device=device)
                < self.student_augmentation_probability
            ).item()
        )
        scale = 1.0
        target_height, target_width = reference.shape[-2:]
        if augment:
            scale = float(
                torch.empty((), device=device).uniform_(
                    self.student_downscale_min,
                    self.student_downscale_max,
                ).item()
            )

            def _round_dimension(size: int) -> int:
                scaled = int(round(size * scale / self.student_spatial_factor))
                return max(scaled, 1) * self.student_spatial_factor

            target_height = min(_round_dimension(reference.shape[-2]), reference.shape[-2])
            target_width = min(_round_dimension(reference.shape[-1]), reference.shape[-1])

        degraded = {
            name: self._resize_video(video, target_height, target_width)
            for name, video in views.items()
        }
        noise_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if augment and self.student_noise_probability > 0.0:
            noise_samples = (
                torch.rand(batch_size, device=device)
                < self.student_noise_probability
            )
            noise_std = torch.empty(batch_size, device=device).uniform_(
                self.student_noise_std_min,
                self.student_noise_std_max,
            )
            for name, video in degraded.items():
                valid = student_valid[name] & noise_samples
                if not bool(valid.any().item()):
                    continue
                sample_scale = noise_std.to(video).reshape(
                    batch_size,
                    *([1] * (video.ndim - 1)),
                )
                valid_mask = valid.to(video).reshape(
                    batch_size,
                    *([1] * (video.ndim - 1)),
                )
                noisy = video + torch.randn_like(video) * sample_scale
                degraded[name] = torch.where(
                    valid_mask.bool(),
                    noisy.clamp(
                        self.student_input_clip_min,
                        self.student_input_clip_max,
                    ),
                    video,
                )

        actual_scale = 0.5 * (
            target_height / reference.shape[-2]
            + target_width / reference.shape[-1]
        )
        return degraded, {
            "augmentation_fraction": reference.new_tensor(float(augment)),
            "student_scale": reference.new_tensor(actual_scale),
            "noise_fraction": noise_samples.float().mean(),
        }

    def _degrade_student_history(
        self,
        views: Dict[str, torch.Tensor],
        student_valid: Dict[str, torch.Tensor],
        student_kwargs: dict,
    ) -> tuple[Dict[str, torch.Tensor], dict, dict]:
        """Apply one degradation plan to the current and historical clips."""
        history_views = student_kwargs.get("history_views")
        if history_views is None:
            degraded, stats = self._degrade_student_views(views, student_valid)
            return degraded, dict(student_kwargs), stats
        if student_kwargs.get("history_memory") is not None:
            raise ValueError(
                "OPD training cannot reuse student history_memory for the clean "
                "teacher; provide raw historical clips instead"
            )

        combined = {}
        history_shapes = {}
        for name, current in views.items():
            if name not in history_views:
                raise ValueError(f"Missing historical view {name!r} for OPD")
            historical = history_views[name]
            if historical.ndim != current.ndim + 1:
                raise ValueError(
                    f"history_views[{name!r}] must add a clip axis to "
                    f"{tuple(current.shape)}, got {tuple(historical.shape)}"
                )
            if historical.shape[0] != current.shape[0]:
                raise ValueError("Current and historical OPD batches must match")
            history_shapes[name] = historical.shape
            flattened = historical.reshape(
                historical.shape[0],
                historical.shape[1] * historical.shape[2],
                *historical.shape[3:],
            )
            combined[name] = torch.cat([current, flattened], dim=1)

        degraded_combined, stats = self._degrade_student_views(
            combined,
            student_valid,
        )
        degraded_current = {}
        degraded_history = {}
        for name, degraded in degraded_combined.items():
            current_frames = views[name].shape[1]
            degraded_current[name] = degraded[:, :current_frames]
            historical = degraded[:, current_frames:]
            original_shape = history_shapes[name]
            degraded_history[name] = historical.reshape(
                original_shape[0],
                original_shape[1],
                original_shape[2],
                *historical.shape[2:],
            )

        updated_kwargs = dict(student_kwargs)
        updated_kwargs["history_views"] = degraded_history
        history_view_valid = student_kwargs.get("history_view_valid")
        updated_history_valid = {}
        for name, historical in degraded_history.items():
            if history_view_valid is not None and name in history_view_valid:
                valid = history_view_valid[name].to(dtype=torch.bool)
            else:
                valid = torch.ones(
                    historical.shape[:2],
                    dtype=torch.bool,
                    device=historical.device,
                )
            valid = valid & student_valid[name].to(valid.device).unsqueeze(1)
            updated_history_valid[name] = valid
        updated_kwargs["history_view_valid"] = updated_history_valid
        return degraded_current, updated_kwargs, stats

    def _degrade_pose_tensor(
        self,
        poses: Optional[torch.Tensor],
        noise_samples: torch.Tensor,
        dropout_samples: torch.Tensor,
        noise_std: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Apply one student pose plan without mutating the clean teacher input."""
        if poses is None:
            return None
        if poses.shape[0] != noise_samples.shape[0]:
            raise ValueError("Student pose inputs must share the video batch dimension")
        if poses.ndim < 4 or poses.shape[-1] != 3:
            raise ValueError(
                "Student pose degradation expects (B,...,J,3), got "
                f"{tuple(poses.shape)}"
            )

        degraded = poses
        if bool(noise_samples.any().item()):
            noise = torch.randn_like(poses)
            num_joints = poses.shape[-2]
            if self.student_pose_preserve_wrist_origin:
                if num_joints % 2:
                    raise ValueError(
                        "Wrist-origin-preserving pose noise requires an even "
                        f"number of joints, got {num_joints}"
                    )
                joints_per_hand = num_joints // 2
                for start in (0, joints_per_hand):
                    stop = start + joints_per_hand
                    hand_noise = noise[..., start:stop, :]
                    hand_noise = hand_noise - hand_noise[..., :1, :]
                    # Wrist-relative invalid hands are represented by all-zero
                    # coordinates. Do not turn them into hallucinated hands.
                    hand_available = (
                        poses[..., start:stop, :]
                        .abs()
                        .amax(dim=(-2, -1), keepdim=True)
                        > 0
                    )
                    noise[..., start:stop, :] = hand_noise * hand_available
            sample_scale = noise_std.to(poses).reshape(
                poses.shape[0],
                *([1] * (poses.ndim - 1)),
            )
            noise_mask = noise_samples.reshape(
                poses.shape[0],
                *([1] * (poses.ndim - 1)),
            )
            degraded = torch.where(
                noise_mask,
                poses + noise * sample_scale,
                poses,
            )

        if bool(dropout_samples.any().item()):
            dropout_mask = dropout_samples.reshape(
                poses.shape[0],
                *([1] * (poses.ndim - 1)),
            )
            degraded = torch.where(dropout_mask, torch.zeros_like(poses), degraded)
        return degraded

    @staticmethod
    def _degrade_relation_tensor(
        delta: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        noise_samples: torch.Tensor,
        dropout_samples: torch.Tensor,
        noise_std: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Corrupt the auxiliary wrist relation consistently with pose dropout."""
        if delta is None:
            return None, valid
        if delta.shape[0] != noise_samples.shape[0] or delta.shape[-1] != 3:
            raise ValueError(
                "Student inter-wrist inputs must have shape (B,...,3), got "
                f"{tuple(delta.shape)}"
            )
        relation_valid = (
            valid.to(device=delta.device, dtype=torch.bool)
            if valid is not None
            else torch.ones(delta.shape[:-1], dtype=torch.bool, device=delta.device)
        )
        if tuple(relation_valid.shape) != tuple(delta.shape[:-1]):
            raise ValueError("Inter-wrist validity must match relation leading dimensions")

        degraded = delta
        if bool(noise_samples.any().item()):
            sample_scale = noise_std.to(delta).reshape(
                delta.shape[0],
                *([1] * (delta.ndim - 1)),
            )
            noise_mask = noise_samples.reshape(
                delta.shape[0],
                *([1] * (delta.ndim - 1)),
            )
            valid_mask = relation_valid.unsqueeze(-1)
            degraded = torch.where(
                noise_mask & valid_mask,
                delta + torch.randn_like(delta) * sample_scale,
                delta,
            )
        if bool(dropout_samples.any().item()):
            dropout_mask = dropout_samples.reshape(
                delta.shape[0],
                *([1] * (delta.ndim - 1)),
            )
            degraded = torch.where(dropout_mask, torch.zeros_like(delta), degraded)
            valid_dropout_mask = dropout_samples.reshape(
                relation_valid.shape[0],
                *([1] * (relation_valid.ndim - 1)),
            )
            relation_valid = relation_valid & ~valid_dropout_mask
        return degraded, relation_valid

    def _degrade_student_pose_inputs(
        self,
        poses: Optional[torch.Tensor],
        inter_wrist_delta: Optional[torch.Tensor],
        inter_wrist_valid: Optional[torch.Tensor],
        student_kwargs: dict,
        reference: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        dict,
        dict,
    ]:
        """Sample a categorical clean/noisy/dropped pose mode per sample."""
        batch_size = reference.shape[0]
        device = reference.device
        mode_draw = torch.rand(batch_size, device=device)
        dropout_samples = (
            mode_draw < self.student_pose_dropout_probability
        )
        noise_samples = (
            (mode_draw >= self.student_pose_dropout_probability)
            & (
                mode_draw
                < self.student_pose_dropout_probability
                + self.student_pose_noise_probability
            )
        )
        noise_std = torch.empty(batch_size, device=device).uniform_(
            self.student_pose_noise_std_min,
            self.student_pose_noise_std_max,
        )

        student_poses = self._degrade_pose_tensor(
            poses,
            noise_samples,
            dropout_samples,
            noise_std,
        )
        student_delta, student_relation_valid = self._degrade_relation_tensor(
            inter_wrist_delta,
            inter_wrist_valid,
            noise_samples,
            dropout_samples,
            noise_std,
        )
        updated_kwargs = dict(student_kwargs)
        if updated_kwargs.get("history_poses") is not None:
            updated_kwargs["history_poses"] = self._degrade_pose_tensor(
                updated_kwargs["history_poses"],
                noise_samples,
                dropout_samples,
                noise_std,
            )
        if updated_kwargs.get("history_inter_wrist_delta") is not None:
            history_delta, history_relation_valid = self._degrade_relation_tensor(
                updated_kwargs["history_inter_wrist_delta"],
                updated_kwargs.get("history_inter_wrist_valid"),
                noise_samples,
                dropout_samples,
                noise_std,
            )
            updated_kwargs["history_inter_wrist_delta"] = history_delta
            updated_kwargs["history_inter_wrist_valid"] = history_relation_valid

        noisy_count = noise_samples.float().sum()
        mean_noise_std = (
            (noise_std * noise_samples).sum() / noisy_count.clamp_min(1.0)
        )
        return (
            student_poses,
            student_delta,
            student_relation_valid,
            updated_kwargs,
            {
                "pose_noise_fraction": noise_samples.float().mean(),
                "pose_dropout_fraction": dropout_samples.float().mean(),
                "pose_noise_std": mean_noise_std,
            },
        )

    def _teacher_forward(
        self,
        sample_mask: torch.Tensor,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]],
        poses: Optional[torch.Tensor],
        inter_wrist_delta: Optional[torch.Tensor],
        inter_wrist_valid: Optional[torch.Tensor],
        teacher_context: dict,
        ego_only: bool,
    ) -> Dict[str, torch.Tensor]:
        selected_names = ("ego",) if ego_only else tuple(views)
        teacher_views = {
            name: _select_batch(views[name], sample_mask)
            for name in selected_names
            if name in views
        }
        teacher_valid = None
        if view_valid is not None:
            teacher_valid = {
                name: _select_batch(view_valid[name], sample_mask)
                for name in selected_names
                if name in view_valid
            }
        context = {}
        for key in TEACHER_CONTEXT_KEYS:
            value = teacher_context.get(key)
            if value is None:
                continue
            if key in {"history_views", "history_view_valid"}:
                context[key] = _select_batch_mapping(
                    value,
                    sample_mask,
                    selected_names=selected_names if ego_only else None,
                )
            else:
                context[key] = _select_batch(value, sample_mask)
        return self.teacher(
            views=teacher_views,
            view_valid=teacher_valid,
            poses=_select_batch(poses, sample_mask),
            inter_wrist_delta=_select_batch(inter_wrist_delta, sample_mask),
            inter_wrist_valid=_select_batch(inter_wrist_valid, sample_mask),
            **context,
        )

    @torch.no_grad()
    def _clean_teacher_targets(
        self,
        outputs: Dict[str, torch.Tensor],
        privileged_mask: torch.Tensor,
        clean_ego_mask: torch.Tensor,
        views: Dict[str, torch.Tensor],
        view_valid: Optional[Dict[str, torch.Tensor]],
        poses: Optional[torch.Tensor],
        inter_wrist_delta: Optional[torch.Tensor],
        inter_wrist_valid: Optional[torch.Tensor],
        teacher_context: dict,
    ) -> Dict[str, torch.Tensor]:
        targets = {key: torch.zeros_like(outputs[key]) for key in DISTILLATION_OUTPUT_KEYS}
        for sample_mask, ego_only in (
            (privileged_mask, self.teacher_ego_only),
            (clean_ego_mask, True),
        ):
            if not bool(sample_mask.any().item()):
                continue
            teacher_outputs = self._teacher_forward(
                sample_mask,
                views,
                view_valid,
                poses,
                inter_wrist_delta,
                inter_wrist_valid,
                teacher_context,
                ego_only=ego_only,
            )
            for key in DISTILLATION_OUTPUT_KEYS:
                if key not in teacher_outputs:
                    raise RuntimeError(f"Teacher output is missing required key {key!r}")
                if teacher_outputs[key].shape[1:] != targets[key].shape[1:]:
                    raise RuntimeError(
                        f"Teacher/student {key} shapes disagree: "
                        f"{tuple(teacher_outputs[key].shape)} versus "
                        f"{tuple(targets[key][sample_mask].shape)}"
                    )
                targets[key][sample_mask] = teacher_outputs[key].detach().to(targets[key])
        return targets

    @staticmethod
    def _gate_fraction(gate: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        support_float = support.float()
        return (gate & support).float().sum() / support_float.sum().clamp_min(1.0)

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
        distill_pressure_available: Optional[torch.Tensor] = None,
        distill_contact_available: Optional[torch.Tensor] = None,
        **student_kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Validation/inference remains exactly the existing dual-branch model.
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
        if views is None or "ego" not in views:
            raise ValueError(
                "Hybrid teacher training requires a views mapping with ego"
            )

        privileged_mask = self._privileged_mask(views, view_valid)
        ego_valid = _batch_validity(
            "ego",
            views,
            view_valid,
            privileged_mask.shape[0],
            privileged_mask.device,
        )
        clean_ego_mask = (~privileged_mask) & ego_valid & self.distill_clean_ego
        active_mask = privileged_mask | clean_ego_mask

        student_valid, force_ego_only, retained_ratio = self._degraded_view_validity(
            views,
            view_valid,
            privileged_mask,
        )
        teacher_context = {
            key: student_kwargs.get(key)
            for key in TEACHER_CONTEXT_KEYS
            if student_kwargs.get(key) is not None
        }
        student_views, student_kwargs, augmentation_stats = (
            self._degrade_student_history(
                views,
                student_valid,
                student_kwargs,
            )
        )
        (
            student_poses,
            student_inter_wrist_delta,
            student_inter_wrist_valid,
            student_kwargs,
            pose_augmentation_stats,
        ) = self._degrade_student_pose_inputs(
            poses,
            inter_wrist_delta,
            inter_wrist_valid,
            student_kwargs,
            views["ego"],
        )
        outputs = self.student(
            frames=frames,
            poses=student_poses,
            inter_wrist_delta=student_inter_wrist_delta,
            inter_wrist_valid=student_inter_wrist_valid,
            views=student_views,
            view_valid=student_valid,
            **student_kwargs,
        )
        for key in DISTILLATION_OUTPUT_KEYS:
            if key not in outputs:
                raise RuntimeError(
                    f"Hybrid distillation requires student output {key!r}"
                )

        # Always execute the predictor so every DDP rank has the same trainable
        # graph, including a rank that happens to receive no active sample.
        predicted_queries = self.query_predictor(outputs["tactile_queries"])
        teacher_targets = self._clean_teacher_targets(
            outputs,
            privileged_mask,
            clean_ego_mask,
            views,
            view_valid,
            poses,
            inter_wrist_delta,
            inter_wrist_valid,
            teacher_context,
        )

        student_queries = F.normalize(outputs["tactile_queries"].float(), dim=-1)
        target_queries = F.normalize(
            teacher_targets["tactile_queries"].float(),
            dim=-1,
        )
        query_gap_elements = 1.0 - (student_queries * target_queries).sum(dim=-1)
        query_gap, query_support = _masked_mean_per_sample(query_gap_elements)
        query_support &= active_mask
        if self.gap_gate_enabled:
            query_gate = (
                query_support
                & (query_gap >= self.query_gap_min)
                & (query_gap <= self.query_gap_max)
            )
        else:
            query_gate = query_support

        predicted_queries = F.normalize(predicted_queries.float(), dim=-1)
        query_error = 2.0 - 2.0 * (
            predicted_queries * target_queries
        ).sum(dim=-1)
        query_loss = _masked_mean(
            query_error,
            query_gate.reshape(
                query_gate.shape[0],
                *([1] * (query_error.ndim - 1)),
            ),
        )

        pressure_error = F.smooth_l1_loss(
            outputs["tactile"].float(),
            teacher_targets["tactile"].float(),
            reduction="none",
        )
        pressure_active = active_mask & query_gate
        pressure_mask = _task_mask(
            student_kwargs.get("tactile_mask"),
            distill_pressure_available,
            pressure_active,
            pressure_error,
        )
        pressure_loss = _masked_mean(pressure_error, pressure_mask)

        contact_error, teacher_contact_probability = bernoulli_js_divergence(
            outputs["contact_logits"],
            teacher_targets["contact_logits"],
            temperature=self.contact_temperature,
        )
        contact_base_mask = _task_mask(
            student_kwargs.get("tactile_mask"),
            distill_contact_available,
            active_mask,
            contact_error,
        )
        contact_gap, contact_support = _masked_mean_per_sample(
            contact_error,
            contact_base_mask,
        )
        if self.gap_gate_enabled:
            contact_gate = (
                contact_support
                & (contact_gap >= self.contact_gap_min)
                & (contact_gap <= self.contact_gap_max)
            )
        else:
            contact_gate = contact_support
        confident_contact = (
            torch.maximum(
                teacher_contact_probability,
                1.0 - teacher_contact_probability,
            )
            >= self.contact_teacher_confidence
        )
        contact_mask = (
            contact_base_mask
            * confident_contact.to(contact_error)
            * contact_gate.to(contact_error).reshape(
                contact_gate.shape[0],
                *([1] * (contact_error.ndim - 1)),
            )
        )
        contact_loss = _masked_mean(contact_error, contact_mask)

        distillation_loss = (
            self.query_weight * query_loss
            + self.pressure_weight * pressure_loss
            + self.contact_weight * contact_loss
        )
        outputs["privileged_distillation_loss"] = distillation_loss
        outputs["privileged_distillation"] = {
            "query": query_loss,
            "pressure": pressure_loss,
            "contact": contact_loss,
            "query_gap": _masked_mean(query_gap, query_support),
            "contact_gap": _masked_mean(contact_gap, contact_support),
            "query_gate_fraction": self._gate_fraction(query_gate, query_support),
            "contact_gate_fraction": self._gate_fraction(
                contact_gate,
                contact_support,
            ),
            # Fractions of the entire batch, after task masks and confidence gates.
            "eligible_fraction": active_mask.float().mean(),
            "query_effective_fraction": query_gate.float().mean(),
            "pressure_effective_fraction": (pressure_mask.flatten(1).sum(1) > 0).float().mean(),
            "contact_effective_fraction": (contact_mask.flatten(1).sum(1) > 0).float().mean(),
            "privileged_fraction": privileged_mask.float().mean(),
            "clean_ego_teacher_fraction": (active_mask if self.teacher_ego_only else clean_ego_mask).float().mean(),
            "student_ego_only_fraction": force_ego_only.float().mean(),
            "retained_wrist_fraction": retained_ratio,
            **augmentation_stats,
            **pose_augmentation_stats,
        }
        return outputs
