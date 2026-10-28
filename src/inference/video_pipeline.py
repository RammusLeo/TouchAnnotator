"""Temporal utilities for video-to-tactile inference.

The model consumes fixed-length clips, while a user-facing video can have any
length.  This module owns the two pieces of temporal state needed by the
pipeline: conservative hand-pose stabilization and overlap-add clip fusion.
It intentionally has no OpenCV or model imports so its behavior can be tested
without the inference environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClipWindow:
    """A fixed-length clip plus the number of non-padding samples it contains."""

    start: int
    indices: np.ndarray
    valid_length: int


def build_overlapping_windows(
    num_frames: int,
    clip_length: int,
    overlap: int,
) -> list[ClipWindow]:
    """Cover a timeline with fixed-length, tail-aligned overlapping clips.

    Videos shorter than one clip are padded by repeating their last frame.
    For longer videos, the final window is aligned to the end so every model
    timestep is covered without padding.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}")
    if not 0 <= overlap < clip_length:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < clip_length; got "
            f"overlap={overlap}, clip_length={clip_length}"
        )

    if num_frames <= clip_length:
        indices = np.minimum(np.arange(clip_length), num_frames - 1)
        return [ClipWindow(start=0, indices=indices, valid_length=num_frames)]

    stride = clip_length - overlap
    last_start = num_frames - clip_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [
        ClipWindow(
            start=start,
            indices=np.arange(start, start + clip_length),
            valid_length=clip_length,
        )
        for start in starts
    ]


def temporal_blend_weights(clip_length: int, mode: str = "hann") -> np.ndarray:
    """Return strictly positive weights for overlap-add prediction fusion."""
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}")
    if mode == "uniform" or clip_length == 1:
        return np.ones(clip_length, dtype=np.float32)
    if mode != "hann":
        raise ValueError(f"Unknown blend mode {mode!r}; expected 'hann' or 'uniform'")
    # Taking the interior of a longer Hann window avoids zero-weight endpoints.
    return np.hanning(clip_length + 2)[1:-1].astype(np.float32)


def _fill_short_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    filled = np.zeros_like(valid)
    if max_gap <= 0 or len(valid) < 3:
        return values, valid, filled

    index = 0
    while index < len(valid):
        if valid[index]:
            index += 1
            continue
        gap_start = index
        while index < len(valid) and not valid[index]:
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start
        if (
            gap_length <= max_gap
            and gap_start > 0
            and gap_end < len(valid)
            and valid[gap_start - 1]
            and valid[gap_end]
        ):
            left = values[gap_start - 1]
            right = values[gap_end]
            for offset in range(gap_length):
                alpha = (offset + 1) / (gap_length + 1)
                values[gap_start + offset] = (1.0 - alpha) * left + alpha * right
            valid[gap_start:gap_end] = True
            filled[gap_start:gap_end] = True
    return values, valid, filled


def _smooth_valid_runs(values: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values
    result = values.copy()
    kernel = np.arange(1, radius + 2, dtype=np.float32)
    kernel = np.concatenate([kernel, kernel[-2::-1]])
    kernel /= kernel.sum()

    index = 0
    while index < len(valid):
        if not valid[index]:
            index += 1
            continue
        run_start = index
        while index < len(valid) and valid[index]:
            index += 1
        run_end = index
        run = values[run_start:run_end]
        if len(run) == 1:
            continue
        padded = np.pad(
            run,
            ((radius, radius),) + ((0, 0),) * (run.ndim - 1),
            mode="edge",
        )
        for local_index in range(len(run)):
            window = padded[local_index : local_index + len(kernel)]
            result[run_start + local_index] = np.tensordot(
                kernel, window, axes=(0, 0)
            )
    return result


def stabilize_pose_sequence(
    poses: np.ndarray,
    valid: np.ndarray,
    max_gap: int = 2,
    smooth_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only short internal gaps, then smooth contiguous valid runs.

    Long gaps and leading/trailing misses remain invalid and are set to zero.
    The returned ``filled`` mask distinguishes detector output from temporal
    interpolation for diagnostics.
    """
    poses = np.asarray(poses, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if poses.ndim != 3 or poses.shape[1:] != (21, 3):
        raise ValueError(f"Expected poses with shape (T,21,3), got {poses.shape}")
    if valid.shape != (len(poses),):
        raise ValueError(f"Expected valid with shape {(len(poses),)}, got {valid.shape}")
    if max_gap < 0:
        raise ValueError(f"max_gap must be non-negative, got {max_gap}")
    if smooth_radius < 0:
        raise ValueError(f"smooth_radius must be non-negative, got {smooth_radius}")

    finite = np.isfinite(poses).all(axis=(1, 2))
    effective_input = valid & finite
    clean = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    clean, effective_valid, filled = _fill_short_gaps(
        clean, effective_input, max_gap=max_gap
    )
    clean = _smooth_valid_runs(clean, effective_valid, radius=smooth_radius)
    clean[~effective_valid] = 0.0
    return clean.astype(np.float32), effective_valid, filled


def interpolate_valid_runs(
    values: np.ndarray,
    valid: np.ndarray,
    sample_positions: np.ndarray,
    query_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate contiguous valid runs without bridging long invalid gaps.

    Each valid run owns the query interval up to the midpoint between its edge
    sample and the neighboring invalid sample. A one-sample run is therefore
    shown for roughly one sampling period, while multi-sample runs are linearly
    interpolated and held only to those conservative boundaries.
    """
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    sample_positions = np.asarray(sample_positions, dtype=np.float64)
    query_positions = np.asarray(query_positions, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"values must have shape (T,...), got {values.shape}")
    if valid.shape != (len(values),):
        raise ValueError(f"valid must have shape {(len(values),)}, got {valid.shape}")
    if sample_positions.shape != (len(values),):
        raise ValueError(
            f"sample_positions must have shape {(len(values),)}, got "
            f"{sample_positions.shape}"
        )
    if query_positions.ndim != 1:
        raise ValueError("query_positions must be one-dimensional")
    if len(values) == 0:
        raise ValueError("values must contain at least one sample")
    if np.any(np.diff(sample_positions) <= 0):
        raise ValueError("sample_positions must be strictly increasing")
    if np.any(np.diff(query_positions) < 0):
        raise ValueError("query_positions must be non-decreasing")

    finite = np.isfinite(values).reshape(len(values), -1).all(axis=1)
    valid = valid & finite
    output = np.zeros((len(query_positions),) + values.shape[1:], dtype=np.float32)
    output_valid = np.zeros(len(query_positions), dtype=bool)

    index = 0
    while index < len(valid):
        if not valid[index]:
            index += 1
            continue
        run_start = index
        while index < len(valid) and valid[index]:
            index += 1
        run_end = index

        left_boundary = (
            query_positions[0]
            if run_start == 0
            else 0.5
            * (sample_positions[run_start - 1] + sample_positions[run_start])
        )
        right_boundary = (
            query_positions[-1]
            if run_end == len(valid)
            else 0.5 * (sample_positions[run_end - 1] + sample_positions[run_end])
        )
        query_indices = np.flatnonzero(
            (query_positions >= left_boundary) & (query_positions <= right_boundary)
        )
        if not len(query_indices):
            continue
        run_values = values[run_start:run_end]
        run_positions = sample_positions[run_start:run_end]
        if len(run_values) == 1:
            output[query_indices] = run_values[0]
        else:
            query = query_positions[query_indices]
            upper = np.searchsorted(run_positions, query, side="right")
            upper = np.clip(upper, 1, len(run_positions) - 1)
            lower = upper - 1
            denominator = run_positions[upper] - run_positions[lower]
            alpha = np.clip(
                (query - run_positions[lower]) / denominator, 0.0, 1.0
            ).astype(np.float32)
            alpha = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
            output[query_indices] = (
                values[run_start + lower] * (1.0 - alpha)
                + values[run_start + upper] * alpha
            )
        output_valid[query_indices] = True
    return output, output_valid


def interpolate_timeline(
    values: np.ndarray,
    sample_indices: np.ndarray,
    output_length: int,
) -> np.ndarray:
    """Linearly interpolate sampled predictions onto the source-frame timeline."""
    values = np.asarray(values)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if output_length <= 0:
        raise ValueError(f"output_length must be positive, got {output_length}")
    if len(values) == 0 or values.shape[0] != len(sample_indices):
        raise ValueError(
            "values and sample_indices must have the same non-zero leading length"
        )
    if np.any(np.diff(sample_indices) <= 0):
        raise ValueError("sample_indices must be strictly increasing")
    if sample_indices[0] < 0 or sample_indices[-1] >= output_length:
        raise ValueError("sample_indices must lie inside the output timeline")

    if len(values) == 1:
        return np.repeat(values, output_length, axis=0)

    source = values.reshape(len(values), -1)
    target_indices = np.arange(output_length)
    result = np.empty((output_length, source.shape[1]), dtype=np.float32)
    for feature_index in range(source.shape[1]):
        result[:, feature_index] = np.interp(
            target_indices,
            sample_indices,
            source[:, feature_index],
        )
    return result.reshape((output_length,) + values.shape[1:])


def select_timestamp_indices(
    timestamps: np.ndarray,
    target_fps: float,
) -> np.ndarray:
    """Select source frames nearest to a fixed-rate timestamp grid.

    The first source timestamp anchors the grid. Duplicate nearest neighbours
    are removed, which also makes this safe when the source rate is lower than
    ``target_fps``.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("timestamps must be a non-empty 1D array")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must all be finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if len(timestamps) == 1:
        return np.array([0], dtype=np.int64)

    relative = timestamps - timestamps[0]
    period = 1.0 / float(target_fps)
    targets = np.arange(0.0, relative[-1] + period * 1e-6, period)
    right = np.searchsorted(relative, targets, side="left")
    right = np.clip(right, 0, len(relative) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(relative[left] - targets) <= np.abs(
        relative[right] - targets
    )
    selected = np.where(choose_left, left, right)
    return np.unique(selected).astype(np.int64)


def interpolate_at_positions(
    values: np.ndarray,
    sample_positions: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate values between arbitrary monotonic positions."""
    values = np.asarray(values)
    sample_positions = np.asarray(sample_positions, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    if sample_positions.ndim != 1 or target_positions.ndim != 1:
        raise ValueError("sample_positions and target_positions must be 1D")
    if len(values) == 0 or values.shape[0] != len(sample_positions):
        raise ValueError(
            "values and sample_positions must have the same non-zero leading length"
        )
    if len(target_positions) == 0:
        raise ValueError("target_positions must not be empty")
    if not np.isfinite(sample_positions).all() or not np.isfinite(target_positions).all():
        raise ValueError("interpolation positions must all be finite")
    if np.any(np.diff(sample_positions) <= 0):
        raise ValueError("sample_positions must be strictly increasing")
    if np.any(np.diff(target_positions) < 0):
        raise ValueError("target_positions must be non-decreasing")
    if len(values) == 1:
        return np.repeat(values, len(target_positions), axis=0)

    source = values.reshape(len(values), -1)
    result = np.empty((len(target_positions), source.shape[1]), dtype=np.float32)
    for feature_index in range(source.shape[1]):
        result[:, feature_index] = np.interp(
            target_positions,
            sample_positions,
            source[:, feature_index],
        )
    return result.reshape((len(target_positions),) + values.shape[1:])
