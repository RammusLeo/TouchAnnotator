import numpy as np
import pytest

from src.inference.video_pipeline import (
    build_overlapping_windows,
    interpolate_at_positions,
    interpolate_valid_runs,
    interpolate_timeline,
    select_timestamp_indices,
    stabilize_pose_sequence,
    temporal_blend_weights,
)


def test_interpolate_valid_runs_does_not_bridge_invalid_gap():
    values = np.asarray([[0.0], [1.0], [0.0], [3.0]], dtype=np.float32)
    valid = np.asarray([True, True, False, True])
    sample_positions = np.asarray([0.0, 1.0, 2.0, 3.0])
    query_positions = np.arange(0.0, 3.01, 0.25)

    interpolated, output_valid = interpolate_valid_runs(
        values, valid, sample_positions, query_positions
    )

    np.testing.assert_allclose(interpolated[query_positions == 0.5, 0], 0.5)
    assert not output_valid[query_positions == 2.0].item()
    assert output_valid[query_positions == 3.0].item()
    np.testing.assert_allclose(interpolated[query_positions == 3.0, 0], 3.0)


def test_overlapping_windows_cover_tail_and_pad_short_video():
    windows = build_overlapping_windows(num_frames=17, clip_length=8, overlap=4)
    assert [window.start for window in windows] == [0, 4, 8, 9]
    covered = np.zeros(17, dtype=bool)
    for window in windows:
        covered[window.indices[: window.valid_length]] = True
    assert covered.all()

    short = build_overlapping_windows(num_frames=3, clip_length=8, overlap=4)[0]
    assert short.valid_length == 3
    assert short.indices.tolist() == [0, 1, 2, 2, 2, 2, 2, 2]


def test_overlap_and_blend_validation():
    with pytest.raises(ValueError, match="overlap"):
        build_overlapping_windows(num_frames=8, clip_length=8, overlap=8)
    weights = temporal_blend_weights(8, mode="hann")
    assert weights.shape == (8,)
    assert np.all(weights > 0)
    assert np.allclose(weights, weights[::-1])


def test_pose_stabilization_fills_only_short_internal_gaps():
    poses = np.zeros((7, 21, 3), dtype=np.float32)
    valid = np.array([False, True, False, True, False, False, True])
    poses[1] = 1.0
    poses[3] = 3.0
    poses[6] = 6.0

    stabilized, effective_valid, filled = stabilize_pose_sequence(
        poses, valid, max_gap=1, smooth_radius=0
    )
    assert effective_valid.tolist() == [False, True, True, True, False, False, True]
    assert filled.tolist() == [False, False, True, False, False, False, False]
    assert np.allclose(stabilized[2], 2.0)
    assert np.allclose(stabilized[0], 0.0)
    assert np.allclose(stabilized[4:6], 0.0)


def test_interpolate_timeline_preserves_samples_and_extends_edges():
    values = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    output = interpolate_timeline(values, np.array([0, 2, 4]), output_length=6)
    assert output[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]


def test_timestamp_selection_uses_nearest_source_frame():
    timestamps = np.arange(13, dtype=np.float64) / 90.0
    selected = select_timestamp_indices(timestamps, target_fps=15.0)
    assert selected.tolist() == [0, 6, 12]


def test_interpolate_at_timestamp_positions():
    values = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    output = interpolate_at_positions(
        values,
        sample_positions=np.array([0.0, 0.2, 0.4]),
        target_positions=np.array([0.0, 0.1, 0.3, 0.5]),
    )
    assert output[:, 0].tolist() == [0.0, 1.0, 3.0, 4.0]
