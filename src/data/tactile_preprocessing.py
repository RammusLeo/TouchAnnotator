"""Shared tactile-model preprocessing used by both training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml


POSE_COORDINATE_MODES = {'camera', 'wrist_relative'}


def preprocessing_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that must agree between a checkpoint run and inference."""
    data = config.get('data', {})
    model = config.get('model', {})
    pose_encoder = model.get('pose_encoder', {})
    return {
        'use_pose': bool(model.get('use_pose', True)),
        'pose_coordinate_mode': data.get('pose_coordinate_mode', 'camera'),
        'pose_normalization': pose_encoder.get('normalization', 'legacy_camera'),
        'pose_source': data.get('pose_source', 'rokoko'),
        'clip_length': int(data.get('clip_length', 8)),
        'frame_interval': int(data.get('frame_interval', 1)),
        'target_fps': None if data.get('target_fps') is None else float(data['target_fps']),
        'inter_wrist_relation': pose_encoder.get('inter_wrist_relation', {}),
    }


def validate_checkpoint_preprocessing(
    checkpoint_path: str | Path,
    inference_config: Dict[str, Any],
) -> Path | None:
    """Reject a checkpoint/config pair with different preprocessing semantics.

    Training writes a resolved ``config.yaml`` beside each checkpoint. Older
    external checkpoints may not have one; those remain loadable, but no
    contract can be verified automatically.
    """
    saved_config_path = Path(checkpoint_path).resolve().parent / 'config.yaml'
    if not saved_config_path.is_file():
        return None
    with saved_config_path.open('r', encoding='utf-8') as handle:
        saved_config = yaml.safe_load(handle) or {}
    saved_contract = preprocessing_contract(saved_config)
    inference_contract = preprocessing_contract(inference_config)
    mismatches = {
        key: (saved_contract[key], inference_contract[key])
        for key in saved_contract
        if saved_contract[key] != inference_contract[key]
    }
    if mismatches:
        details = ', '.join(
            f"{key}: checkpoint={saved!r}, inference={runtime!r}"
            for key, (saved, runtime) in mismatches.items()
        )
        raise ValueError(
            "Checkpoint preprocessing does not match the inference config: "
            f"{details}. Use the config.yaml saved beside the checkpoint."
        )
    return saved_config_path


def validate_pose_coordinate_mode(mode: str) -> str:
    if mode not in POSE_COORDINATE_MODES:
        raise ValueError(
            f"Unsupported pose_coordinate_mode: {mode}. "
            f"Expected one of {sorted(POSE_COORDINATE_MODES)}"
        )
    return mode


def align_hand_pose_coordinates(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two MANO-ordered hands while preserving metric hand geometry."""
    validate_pose_coordinate_mode(mode)
    left_xyz = np.asarray(left_xyz, dtype=np.float32).copy()
    right_xyz = np.asarray(right_xyz, dtype=np.float32).copy()
    if mode == 'wrist_relative':
        left_mask = np.asarray(left_valid).astype(bool)
        right_mask = np.asarray(right_valid).astype(bool)
        left_xyz[left_mask] -= left_xyz[left_mask, :1, :]
        right_xyz[right_mask] -= right_xyz[right_mask, :1, :]
    return left_xyz, right_xyz


def compute_inter_wrist_delta(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the right-wrist position relative to the left wrist.

    This must be called before per-hand wrist centering, otherwise both wrist
    joints are zero and the bimanual translation is irretrievably lost.  Joint
    zero is the wrist in the MANO-21 ordering used by all tactile datasets.

    Returns:
        delta: ``(T, 3)`` array containing ``right_wrist - left_wrist``.
        valid: ``(T,)`` boolean mask; invalid deltas are explicitly zeroed.
    """
    left_xyz = np.asarray(left_xyz, dtype=np.float32)
    right_xyz = np.asarray(right_xyz, dtype=np.float32)
    if left_xyz.ndim != 3 or right_xyz.ndim != 3:
        raise ValueError(
            "left_xyz and right_xyz must have shape (T,J,3), got "
            f"{left_xyz.shape} and {right_xyz.shape}"
        )
    if left_xyz.shape[0] != right_xyz.shape[0] or left_xyz.shape[-1] != 3 \
            or right_xyz.shape[-1] != 3:
        raise ValueError(
            "left_xyz and right_xyz must share T and have xyz coordinates, got "
            f"{left_xyz.shape} and {right_xyz.shape}"
        )

    left_valid = np.asarray(left_valid).astype(bool)
    right_valid = np.asarray(right_valid).astype(bool)
    if left_valid.shape != (left_xyz.shape[0],) or right_valid.shape != (right_xyz.shape[0],):
        raise ValueError(
            "left_valid and right_valid must have shape (T,), got "
            f"{left_valid.shape} and {right_valid.shape}"
        )

    delta = right_xyz[:, 0, :] - left_xyz[:, 0, :]
    valid = (
        left_valid
        & right_valid
        & np.isfinite(left_xyz[:, 0, :]).all(axis=-1)
        & np.isfinite(right_xyz[:, 0, :]).all(axis=-1)
    )
    delta = np.where(valid[:, None], delta, 0.0).astype(np.float32)
    return delta, valid


def prepare_bimanual_pose(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    coordinate_mode: str,
) -> np.ndarray:
    """Return the exact ``(T,42,3)`` pose tensor expected by the model.

    ``camera`` reproduces the legacy preprocessing. ``wrist_relative`` removes
    the incomparable weak-perspective/calibrated-camera translation used by
    EgoTouch WiLoR and HOI4D MANO respectively. Invalid hands are zero in the
    wrist-relative representation and retain the legacy ``-10`` marker in the
    legacy camera representation.
    """
    left_xyz, right_xyz = align_hand_pose_coordinates(
        left_xyz, right_xyz, left_valid, right_valid, coordinate_mode
    )
    invalid_value = 0.0 if coordinate_mode == 'wrist_relative' else -10.0
    left_xyz[~np.asarray(left_valid).astype(bool)] = invalid_value
    right_xyz[~np.asarray(right_valid).astype(bool)] = invalid_value
    poses = np.concatenate([left_xyz, right_xyz], axis=1).astype(np.float32)
    poses[:, :, 0] = np.clip(poses[:, :, 0], -10.0, 10.0)
    poses[:, :, 1] = np.clip(poses[:, :, 1], -10.0, 10.0)
    poses[:, :, 2] = np.clip(poses[:, :, 2], -10.0, 100.0)
    return poses


def resolve_frame_interval(
    source_fps: float,
    target_fps: float | None,
    fallback_interval: int,
) -> int:
    """Match training's per-trajectory FPS downsampling rule."""
    fallback_interval = int(fallback_interval)
    if fallback_interval <= 0:
        raise ValueError(f"fallback_interval must be positive, got {fallback_interval}")
    if target_fps is None or float(source_fps) <= 0:
        return fallback_interval
    if float(target_fps) <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    return max(1, int(round(float(source_fps) / float(target_fps))))
