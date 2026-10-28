#!/usr/bin/env python3
"""Streaming Full-Qwen causal inference from one monocular video and hand poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference_from_video_wilor as common  # noqa: E402
from src.data.tactile_preprocessing import (  # noqa: E402
    compute_inter_wrist_delta,
    prepare_bimanual_pose,
    resolve_frame_interval,
)
from src.data.transforms import VideoTransform  # noqa: E402
from src.inference.video_pipeline import interpolate_at_positions  # noqa: E402
from src.models.qwen35_full_tactile import build_qwen35_full_tactile_model  # noqa: E402


def load_poses(path: Path, sampled_indices: np.ndarray, source_count: int):
    """Load sampled/source-timeline poses from NPZ or WiLoR JSONL."""
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        left = np.asarray(data["left_pose"], dtype=np.float32)
        right = np.asarray(data["right_pose"], dtype=np.float32)
        lv = np.asarray(data.get("left_valid", np.ones(len(left))), dtype=bool)
        rv = np.asarray(data.get("right_valid", np.ones(len(right))), dtype=bool)
        if len(left) == source_count:
            left, right, lv, rv = (x[sampled_indices] for x in (left, right, lv, rv))
        elif len(left) != len(sampled_indices):
            raise ValueError(
                f"Pose length {len(left)} must equal source ({source_count}) or sampled "
                f"video length ({len(sampled_indices)})"
            )
        return left, right, lv, rv

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_frame = {int(item["frame_index"]): item for item in records}
    left = np.zeros((len(sampled_indices), 21, 3), dtype=np.float32)
    right = np.zeros_like(left)
    lv = np.zeros(len(sampled_indices), dtype=bool)
    rv = np.zeros(len(sampled_indices), dtype=bool)
    for out_index, frame_index in enumerate(sampled_indices):
        item = by_frame.get(int(frame_index))
        if item is None:
            continue
        if item.get("left_valid", bool(item.get("left_pos"))):
            left[out_index] = np.asarray(item["left_pos"], dtype=np.float32).reshape(21, 3)
            lv[out_index] = True
        if item.get("right_valid", bool(item.get("right_pos"))):
            right[out_index] = np.asarray(item["right_pos"], dtype=np.float32).reshape(21, 3)
            rv[out_index] = True
    return left, right, lv, rv


def load_model(config: Path, checkpoint: Path, device: torch.device):
    original_builder = common.build_model
    try:
        common.build_model = build_qwen35_full_tactile_model
        model, cfg = common.load_touchanything(config, checkpoint, device)
    finally:
        common.build_model = original_builder
    memory = getattr(model, "clip_memory", None)
    if memory is None:
        raise ValueError("Config is not full causal: model.clip_memory.enabled must be true")
    if not cfg.get("model", {}).get("qwen_full_reasoner", {}).get("enabled", False):
        raise ValueError("Config is not Full-Qwen: model.qwen_full_reasoner.enabled must be true")
    return model, cfg


def infer(model, cfg, frames, left, right, lv, rv, device):
    clip_length = int(cfg["data"].get("clip_length", 8))
    transform = VideoTransform(
        image_size=tuple(cfg["data"]["image_size"]),
        normalize_mean=cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=cfg["data"].get("normalize_std", [0.229, 0.224, 0.225]),
        use_augmentation=False,
    )
    coordinate_mode = cfg["data"].get("pose_coordinate_mode", "camera")
    history_depth = model.clip_memory.history_clips
    memory_cache = []
    tactile_parts, contact_parts = [], []
    for start in tqdm(range(0, len(frames), clip_length), desc="Full-causal inference"):
        valid_length = min(clip_length, len(frames) - start)
        indices = np.minimum(np.arange(start, start + clip_length), len(frames) - 1)
        clip_left, clip_right = left[indices], right[indices]
        clip_lv, clip_rv = lv[indices], rv[indices]
        wrist_delta, wrist_valid = compute_inter_wrist_delta(
            clip_left, clip_right, clip_lv, clip_rv
        )
        pose = prepare_bimanual_pose(
            clip_left, clip_right, clip_lv, clip_rv, coordinate_mode
        )
        image_t = transform(frames[indices]).unsqueeze(0).to(device)
        pose_t = torch.from_numpy(pose).unsqueeze(0).to(device)
        delta_t = torch.from_numpy(wrist_delta).unsqueeze(0).to(device)
        valid_t = torch.from_numpy(wrist_valid).unsqueeze(0).to(device)
        memory_shape = (1, history_depth, model.clip_memory.memory_tokens_per_clip,
                        model.clip_memory.embed_dim)
        history_memory = image_t.new_zeros(memory_shape)
        history_valid = torch.zeros((1, history_depth), dtype=torch.bool, device=device)
        if memory_cache:
            recent = memory_cache[-history_depth:]
            history_memory[:, -len(recent):] = torch.stack(recent, dim=0)
            history_valid[:, -len(recent):] = True
        kwargs = dict(
            poses=pose_t, inter_wrist_delta=delta_t,
            inter_wrist_valid=valid_t, history_memory=history_memory,
            history_valid=history_valid,
        )
        if getattr(model, "multi_view_enabled", False):
            kwargs.update(views={"ego": image_t},
                          view_valid={"ego": torch.ones(1, dtype=torch.bool, device=device)})
        else:
            kwargs["frames"] = image_t
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(**kwargs)
        memory_cache.append(output["clip_memory"].detach()[0])
        memory_cache = memory_cache[-history_depth:]
        tactile_parts.append(output["tactile"][0, :valid_length].float().cpu().numpy())
        contact_parts.append(output["contact"][0, :valid_length].float().cpu().numpy())
    return np.concatenate(tactile_parts), np.concatenate(contact_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--hand-pose", required=True, help="NPZ or WiLoR poses.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    video, pose_path = Path(args.video).resolve(), Path(args.hand_pose).resolve()
    checkpoint, config = Path(args.checkpoint).resolve(), Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(config, checkpoint, device)
    cap = cv2.VideoCapture(str(video))
    fps_hint = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    interval = resolve_frame_interval(
        fps_hint, cfg["data"].get("target_fps"), cfg["data"].get("frame_interval", 1)
    )
    bgr, source_indices, source_count, source_fps = common.read_sampled_video(
        video, interval, args.max_frames
    )
    left, right, lv, rv = load_poses(pose_path, source_indices, source_count)
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in bgr])
    sampled_pressure, sampled_contact = infer(
        model, cfg, rgb, left, right, lv, rv, device
    )
    sampled_times = source_indices.astype(np.float64) / source_fps
    source_times = np.arange(source_count, dtype=np.float64) / source_fps
    pressure = interpolate_at_positions(sampled_pressure, sampled_times, source_times)
    contact_probability = interpolate_at_positions(
        sampled_contact, sampled_times, source_times
    )
    contact_state = contact_probability >= args.contact_threshold
    hand_contact = contact_state.reshape(source_count, 2, -1).any(axis=-1)
    np.savez_compressed(
        output_dir / "contact_predictions.npz", pressure=pressure,
        contact_probability=contact_probability, contact_state=contact_state,
        hand_contact=hand_contact, sampled_pressure=sampled_pressure,
        sampled_contact_probability=sampled_contact,
        sampled_source_indices=source_indices, left_pose=left, right_pose=right,
        left_valid=lv, right_valid=rv, source_fps=np.float32(source_fps),
        contact_threshold=np.float32(args.contact_threshold),
    )
    with (output_dir / "contact_state.jsonl").open("w") as handle:
        for i, state in enumerate(hand_contact):
            handle.write(json.dumps({"frame_index": i, "timestamp": i / source_fps,
                                     "left_contact": bool(state[0]),
                                     "right_contact": bool(state[1])}) + "\n")
    print(f"Saved {source_count} frames to {output_dir}")


if __name__ == "__main__":
    main()
