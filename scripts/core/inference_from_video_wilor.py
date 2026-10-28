#!/usr/bin/env python3
"""Run WiLoR pose estimation followed by TouchAnything video inference.

The source video is sampled at the checkpoint's target FPS.  WiLoR estimates
camera-frame MANO-21 joints, short detection gaps are stabilized, and
TouchAnything runs on fixed-length clips with overlap-add fusion.  Predictions
are finally interpolated back to the source-video timeline for visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any, Iterator
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tactile_preprocessing import (  # noqa: E402
    compute_inter_wrist_delta,
    prepare_bimanual_pose,
    resolve_frame_interval,
    validate_checkpoint_preprocessing,
)
from src.data.transforms import VideoTransform  # noqa: E402
from src.inference.video_pipeline import (  # noqa: E402
    build_overlapping_windows,
    interpolate_at_positions,
    interpolate_timeline,
    select_timestamp_indices,
    stabilize_pose_sequence,
    temporal_blend_weights,
)
from src.models import build_model  # noqa: E402
from src.utils.config import load_config_with_base  # noqa: E402
from src.utils.vis_pressure import (  # noqa: E402
    PRESS_H,
    SENSOR_NAN_MASK_L,
    SENSOR_NAN_MASK_R,
    pred_to_21,
    render_pressure_panel,
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def read_sampled_video(
    video_path: Path,
    frame_interval: int,
    max_frames: int | None,
) -> tuple[list[np.ndarray], np.ndarray, int, float]:
    """Decode BGR frames on one target-FPS sampling lattice."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0

    frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0
    while max_frames is None or frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            frames.append(frame)
            indices.append(frame_index)
        frame_index += 1
    cap.release()
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    return frames, np.asarray(indices, dtype=np.int64), frame_index, source_fps


ROS_IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


def resolve_mcap_image_topic(
    mcap_path: Path,
    requested_topic: str | None,
) -> tuple[str, str]:
    """Resolve one ROS2 image channel from an MCAP summary."""
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError(
            "MCAP input requires `pip install mcap mcap-ros2-support` in "
            "the TouchAnything inference environment."
        ) from exc

    with mcap_path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None:
        raise ValueError(f"MCAP has no readable summary: {mcap_path}")
    candidates: list[tuple[str, str, int]] = []
    counts = summary.statistics.channel_message_counts if summary.statistics else {}
    for channel_id, channel in summary.channels.items():
        schema = summary.schemas.get(channel.schema_id)
        schema_name = schema.name if schema is not None else ""
        if schema_name in ROS_IMAGE_TYPES:
            candidates.append(
                (channel.topic, schema_name, int(counts.get(channel_id, 0)))
            )
    if requested_topic is not None:
        for topic, schema_name, _ in candidates:
            if topic == requested_topic:
                return topic, schema_name
        available = ", ".join(
            f"{topic} ({schema}, {count} msgs)"
            for topic, schema, count in candidates
        ) or "none"
        raise ValueError(
            f"Requested MCAP image topic {requested_topic!r} was not found. "
            f"Available image topics: {available}"
        )
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1]
    if not candidates:
        raise ValueError(f"No sensor_msgs Image/CompressedImage topic in {mcap_path}")
    available = ", ".join(
        f"{topic} ({schema}, {count} msgs)" for topic, schema, count in candidates
    )
    raise ValueError(
        "MCAP contains multiple image topics; select ego input with "
        f"--image-topic. Available: {available}"
    )


def _mcap_message_timestamp(item: Any) -> float:
    message = item.ros_msg
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        seconds = int(getattr(stamp, "sec", 0))
        nanoseconds = int(getattr(stamp, "nanosec", 0))
        if seconds != 0 or nanoseconds != 0:
            return seconds + nanoseconds * 1e-9
    raw_message = getattr(item, "message", None)
    log_time = getattr(raw_message, "log_time", None)
    if log_time is None:
        raise ValueError("MCAP image has neither ROS header timestamp nor log_time")
    return int(log_time) * 1e-9


def _decode_ros_image(message: Any, schema_name: str) -> np.ndarray:
    if schema_name == "sensor_msgs/msg/CompressedImage":
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                f"OpenCV failed to decode CompressedImage format={message.format!r}"
            )
        return frame

    if schema_name != "sensor_msgs/msg/Image":
        raise ValueError(f"Unsupported ROS image schema: {schema_name}")
    encoding = str(message.encoding).lower()
    layouts = {
        "bgr8": (3, None),
        "rgb8": (3, cv2.COLOR_RGB2BGR),
        "bgra8": (4, cv2.COLOR_BGRA2BGR),
        "rgba8": (4, cv2.COLOR_RGBA2BGR),
        "mono8": (1, cv2.COLOR_GRAY2BGR),
    }
    if encoding not in layouts:
        raise ValueError(
            f"Unsupported sensor_msgs/Image encoding {message.encoding!r}; "
            f"supported encodings are {sorted(layouts)}"
        )
    channels, conversion = layouts[encoding]
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)
    pixels = rows[:, : width * channels]
    if channels == 1:
        frame = pixels.reshape(height, width)
    else:
        frame = pixels.reshape(height, width, channels)
    if conversion is not None:
        frame = cv2.cvtColor(frame, conversion)
    return np.ascontiguousarray(frame)


def iter_mcap_frames(
    mcap_path: Path,
    topic: str,
    schema_name: str,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as exc:
        raise RuntimeError(
            "MCAP input requires `pip install mcap mcap-ros2-support` in "
            "the TouchAnything inference environment."
        ) from exc
    for frame_index, item in enumerate(
        read_ros2_messages(str(mcap_path), topics=[topic])
    ):
        if max_frames is not None and frame_index >= max_frames:
            break
        yield _decode_ros_image(item.ros_msg, schema_name)


def read_sampled_mcap(
    mcap_path: Path,
    topic: str,
    schema_name: str,
    target_fps: float | None,
    fallback_interval: int,
    max_frames: int | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float, float]:
    """Read timestamp-selected frames without materializing an intermediate video."""
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as exc:
        raise RuntimeError(
            "MCAP input requires `pip install mcap mcap-ros2-support` in "
            "the TouchAnything inference environment."
        ) from exc

    timestamps: list[float] = []
    for frame_index, item in enumerate(
        read_ros2_messages(str(mcap_path), topics=[topic])
    ):
        if max_frames is not None and frame_index >= max_frames:
            break
        timestamps.append(_mcap_message_timestamp(item))
    if not timestamps:
        raise ValueError(f"No messages found on MCAP topic {topic!r}")
    source_timestamps = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(source_timestamps) <= 0):
        raise ValueError(f"MCAP topic timestamps are not strictly increasing: {topic}")
    source_timestamps -= source_timestamps[0]
    if len(source_timestamps) > 1:
        source_fps = (len(source_timestamps) - 1) / source_timestamps[-1]
    else:
        source_fps = float(target_fps or 1.0)
    effective_target_fps = (
        float(target_fps)
        if target_fps is not None
        else source_fps / max(1, int(fallback_interval))
    )
    source_indices = select_timestamp_indices(
        source_timestamps, target_fps=effective_target_fps
    )

    sampled_frames: list[np.ndarray] = []
    selected_cursor = 0
    for frame_index, item in enumerate(
        read_ros2_messages(str(mcap_path), topics=[topic])
    ):
        if selected_cursor >= len(source_indices):
            break
        if frame_index == int(source_indices[selected_cursor]):
            sampled_frames.append(_decode_ros_image(item.ros_msg, schema_name))
            selected_cursor += 1
    if len(sampled_frames) != len(source_indices):
        raise RuntimeError(
            f"Decoded {len(sampled_frames)} sampled MCAP frames, expected "
            f"{len(source_indices)}"
        )
    return (
        sampled_frames,
        source_indices,
        source_timestamps,
        float(source_fps),
        float(effective_target_fps),
    )


class CheckpointMANO(torch.nn.Module):
    """Inference-only MANO layer reconstructed from WiLoR checkpoint buffers.

    WiLoR checkpoints contain the exact LBS buffers used during training.  This
    fallback is useful on deployments where the separately licensed MANO pkl
    was intentionally not copied.  It never writes or exports those buffers.
    """

    BUFFER_NAMES = (
        "shapedirs",
        "faces_tensor",
        "v_template",
        "J_regressor",
        "posedirs",
        "parents",
        "lbs_weights",
        "hand_components",
        "hand_mean",
        "pose_mean",
        "extra_joints_idxs",
        "joint_map",
    )

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        missing = [
            f"mano.{name}"
            for name in self.BUFFER_NAMES
            if f"mano.{name}" not in state_dict
        ]
        if missing:
            raise KeyError(
                "WiLoR checkpoint cannot reconstruct its MANO layer; missing "
                + ", ".join(missing)
            )
        for name in self.BUFFER_NAMES:
            self.register_buffer(name, state_dict[f"mano.{name}"].detach().clone())
        self.vertex_joint_selector = torch.nn.Module()
        selector_key = "mano.vertex_joint_selector.extra_joints_idxs"
        selector_indices = state_dict.get(selector_key, self.extra_joints_idxs)
        self.vertex_joint_selector.register_buffer(
            "extra_joints_idxs", selector_indices.detach().clone()
        )
        self.faces = self.faces_tensor.detach().cpu().numpy()

    def forward(
        self,
        betas: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        hand_pose: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        from smplx.lbs import lbs

        if global_orient is None or hand_pose is None or betas is None:
            raise ValueError(
                "CheckpointMANO requires betas, global_orient, and hand_pose"
            )
        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        vertices, joints = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
        )
        extra_joints = torch.index_select(
            vertices, 1, self.extra_joints_idxs.to(vertices.device)
        )
        joints = torch.cat([joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map.to(joints.device), :]
        if transl is not None:
            vertices = vertices + transl[:, None, :]
            joints = joints + transl[:, None, :]
        return SimpleNamespace(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None,
        )


class WiLoREstimator:
    """Thin adapter around the deployed WiLoR image model and YOLO detector."""

    def __init__(
        self,
        wilor_dir: Path,
        checkpoint: Path,
        detector_checkpoint: Path,
        mano_dir: Path,
        device: torch.device,
        fast: bool = False,
        compile_backbone: bool = False,
    ) -> None:
        if not wilor_dir.is_dir():
            raise FileNotFoundError(f"WiLoR directory not found: {wilor_dir}")
        for path in (checkpoint, detector_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        if str(wilor_dir) not in sys.path:
            sys.path.insert(0, str(wilor_dir))

        try:
            from ultralytics import YOLO
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.configs import get_config
            from wilor.models import WiLoR
            import wilor.models.wilor as wilor_model_module
            from wilor.utils import recursive_to
            from wilor.utils.renderer import cam_crop_to_full
        except ImportError as exc:
            raise RuntimeError(
                "WiLoR dependencies are unavailable in this Python environment. "
                "Install WiLoR requirements (including ultralytics==8.1.34) in "
                "the same environment used for TouchAnything."
            ) from exc

        self.dataset_type = ViTDetDataset
        self.recursive_to = recursive_to
        self.cam_crop_to_full = cam_crop_to_full
        self.device = device
        self.fast = bool(fast and device.type == "cuda")

        config_path = wilor_dir / "pretrained_models" / "model_config.yaml"
        with working_directory(wilor_dir):
            self.model_cfg = get_config(str(config_path), update_cachedir=True)
            self.model_cfg.defrost()
            if (
                "vit" in self.model_cfg.MODEL.BACKBONE.TYPE
                and "BBOX_SHAPE" not in self.model_cfg.MODEL
            ):
                if self.model_cfg.MODEL.IMAGE_SIZE != 256:
                    raise ValueError(
                        "WiLoR ViT checkpoint requires MODEL.IMAGE_SIZE=256"
                    )
                self.model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            if "PRETRAINED_WEIGHTS" in self.model_cfg.MODEL.BACKBONE:
                self.model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            self.model_cfg.MANO.DATA_DIR = str(mano_dir)
            self.model_cfg.MANO.MODEL_PATH = str(mano_dir)
            self.model_cfg.MANO.MEAN_PARAMS = str(
                wilor_dir / "mano_data" / "mano_mean_params.npz"
            )
            self.model_cfg.freeze()
            mano_model_path = mano_dir / "MANO_RIGHT.pkl"
            if not mano_model_path.is_file():
                checkpoint_payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                checkpoint_state = checkpoint_payload.get(
                    "state_dict", checkpoint_payload
                )
                wilor_model_module.MANO = (
                    lambda **_: CheckpointMANO(checkpoint_state)
                )
                print(
                    "MANO_RIGHT.pkl not present; using the exact MANO/LBS "
                    "buffers embedded in the WiLoR checkpoint"
                )
            self.model = WiLoR.load_from_checkpoint(
                str(checkpoint), strict=False, cfg=self.model_cfg
            )
        if self.fast:
            torch.set_float32_matmul_precision("high")
            self.model = self.model.half()
            if hasattr(self.model.backbone, "skip_blocks"):
                self.model.backbone.skip_blocks = True
            if compile_backbone:
                self.model.backbone = torch.compile(self.model.backbone)
        self.model = self.model.to(device).eval()
        mano_faces = self.model.mano.faces
        if torch.is_tensor(mano_faces):
            mano_faces = mano_faces.detach().cpu().numpy()
        self.mano_faces = np.asarray(mano_faces, dtype=np.int32)
        if self.mano_faces.ndim != 2 or self.mano_faces.shape[1] != 3:
            raise ValueError(
                f"WiLoR MANO faces must have shape (F,3), got "
                f"{self.mano_faces.shape}"
            )
        # Ultralytics 8.1 predates PyTorch 2.6's ``weights_only=True`` default.
        # The detector is the trusted local checkpoint shipped with this WiLoR
        # deployment, so restore the historical argument only for its loader.
        original_torch_load = torch.load

        def load_trusted_detector(*args: Any, **kwargs: Any):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = load_trusted_detector
        try:
            self.detector = YOLO(str(detector_checkpoint)).to(device)
        finally:
            torch.load = original_torch_load

    @staticmethod
    def _best_hand_boxes(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep the highest-confidence YOLO detection for each hand side."""
        if result.boxes is None or len(result.boxes) == 0:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        sides = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        scores = result.boxes.conf.detach().cpu().numpy()
        selected: list[int] = []
        for side in (0, 1):
            candidates = np.flatnonzero(sides == side)
            if len(candidates):
                selected.append(int(candidates[np.argmax(scores[candidates])]))
        if not selected:
            empty = np.empty((0,), dtype=np.float32)
            return np.empty((0, 4), dtype=np.float32), empty, empty
        return (
            xyxy[selected].astype(np.float32),
            sides[selected].astype(np.float32),
            scores[selected].astype(np.float32),
        )

    def estimate(
        self,
        frames_bgr: list[np.ndarray],
        detector_batch_size: int,
        wilor_batch_size: int,
        confidence: float,
        rescale_factor: float,
        return_geometry: bool = False,
    ) -> dict[str, np.ndarray]:
        frame_count = len(frames_bgr)
        left = np.zeros((frame_count, 21, 3), dtype=np.float32)
        right = np.zeros((frame_count, 21, 3), dtype=np.float32)
        left_valid = np.zeros(frame_count, dtype=bool)
        right_valid = np.zeros(frame_count, dtype=bool)
        left_score = np.zeros(frame_count, dtype=np.float32)
        right_score = np.zeros(frame_count, dtype=np.float32)
        vertex_count = int(self.mano_faces.max()) + 1
        left_vertices = (
            np.zeros((frame_count, vertex_count, 3), dtype=np.float32)
            if return_geometry
            else None
        )
        right_vertices = (
            np.zeros((frame_count, vertex_count, 3), dtype=np.float32)
            if return_geometry
            else None
        )

        progress = tqdm(total=frame_count, desc="WiLoR pose estimation")
        for detector_start in range(0, frame_count, detector_batch_size):
            chunk = frames_bgr[detector_start : detector_start + detector_batch_size]
            detections = self.detector(chunk, conf=confidence, verbose=False)
            crop_samples: list[dict[str, Any]] = []
            crop_metadata: list[tuple[int, int, float]] = []
            for local_index, (frame, detection) in enumerate(zip(chunk, detections)):
                boxes, sides, scores = self._best_hand_boxes(detection)
                if len(boxes) == 0:
                    continue
                dataset = self.dataset_type(
                    self.model_cfg,
                    frame,
                    boxes,
                    sides,
                    rescale_factor=rescale_factor,
                    fp16=self.fast,
                )
                for detection_index in range(len(dataset)):
                    crop_samples.append(dataset[detection_index])
                    crop_metadata.append(
                        (
                            detector_start + local_index,
                            int(sides[detection_index]),
                            float(scores[detection_index]),
                        )
                    )

            for crop_start in range(0, len(crop_samples), wilor_batch_size):
                crop_end = crop_start + wilor_batch_size
                batch = default_collate(crop_samples[crop_start:crop_end])
                metadata = crop_metadata[crop_start:crop_end]
                batch = self.recursive_to(batch, self.device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.fast,
                ):
                    output = self.model(batch)

                multiplier = 2 * batch["right"] - 1
                pred_cam = output["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                image_size = batch["img_size"].float()
                focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH
                    / self.model_cfg.MODEL.IMAGE_SIZE
                    * image_size.max(dim=1).values
                )
                camera_translation = self.cam_crop_to_full(
                    pred_cam.float(),
                    box_center,
                    box_size,
                    image_size,
                    focal_length,
                )
                joints = output["pred_keypoints_3d"].float().clone()
                joints[:, :, 0] *= multiplier[:, None]
                joints = (joints + camera_translation[:, None, :]).cpu().numpy()
                vertices = None
                if return_geometry:
                    vertices_tensor = output["pred_vertices"].float().clone()
                    vertices_tensor[:, :, 0] *= multiplier[:, None]
                    vertices = (
                        vertices_tensor + camera_translation[:, None, :]
                    ).cpu().numpy()

                for batch_index, (prediction, (frame_index, side, score)) in enumerate(
                    zip(joints, metadata)
                ):
                    if not np.isfinite(prediction).all():
                        continue
                    vertex_prediction = (
                        None if vertices is None else vertices[batch_index]
                    )
                    if vertex_prediction is not None and not np.isfinite(
                        vertex_prediction
                    ).all():
                        continue
                    if side == 1:
                        right[frame_index] = prediction
                        right_valid[frame_index] = True
                        right_score[frame_index] = score
                        if right_vertices is not None:
                            right_vertices[frame_index] = vertex_prediction
                    else:
                        left[frame_index] = prediction
                        left_valid[frame_index] = True
                        left_score[frame_index] = score
                        if left_vertices is not None:
                            left_vertices[frame_index] = vertex_prediction
            progress.update(len(chunk))
        progress.close()
        result = {
            "left": left,
            "right": right,
            "left_valid": left_valid,
            "right_valid": right_valid,
            "left_score": left_score,
            "right_score": right_score,
        }
        if return_geometry:
            assert left_vertices is not None and right_vertices is not None
            result.update(
                {
                    "left_vertices": left_vertices,
                    "right_vertices": right_vertices,
                    "mano_faces": self.mano_faces.copy(),
                }
            )
        return result


def load_touchanything(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    # Training saves a fully resolved config next to each checkpoint, but some
    # historical runs retained a stale ``_base_: base.yaml`` entry without
    # copying that base file.  Such configs are already self-contained.
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    declared_base = raw_config.get("_base_")
    if declared_base is not None:
        declared_path = Path(str(declared_base)).expanduser()
        if not declared_path.is_absolute():
            declared_path = config_path.parent / declared_path
    else:
        declared_path = None
    if declared_path is not None and not declared_path.exists():
        raw_config.pop("_base_", None)
        cfg = raw_config
        print(
            f"Using self-contained checkpoint config; ignored missing historical "
            f"base reference: {declared_path}"
        )
    else:
        cfg = load_config_with_base(str(config_path))
    vision_cfg = cfg.get("model", {}).get("vision_encoder", {})
    pretrained_path = vision_cfg.get("pretrained_path")
    if pretrained_path and not Path(pretrained_path).is_absolute():
        vision_cfg["pretrained_path"] = str((PROJECT_ROOT / pretrained_path).resolve())
    reasoner = cfg.get("model", {}).get("qwen_full_reasoner", {})
    if reasoner.get("pretrained_path") and not Path(reasoner["pretrained_path"]).is_absolute():
        reasoner["pretrained_path"] = str((PROJECT_ROOT / reasoner["pretrained_path"]).resolve())
    saved_config = validate_checkpoint_preprocessing(checkpoint_path, cfg)
    if saved_config is not None:
        print(f"Preprocessing contract verified: {saved_config}")
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    # Older transformers releases exposed DINOv3 as ``encoder.model.layer``;
    # current releases expose the same tensors as ``encoder.layer``.  Remap
    # only keys that become an exact expected name with an identical shape.
    expected_state = model.state_dict()
    compatible_state = {}
    remapped_count = 0
    for key, value in state.items():
        candidate = key.replace(
            "vision_encoder.encoder.model.",
            "vision_encoder.encoder.",
        )
        if (
            candidate != key
            and candidate in expected_state
            and expected_state[candidate].shape == value.shape
        ):
            compatible_state[candidate] = value
            remapped_count += 1
        else:
            compatible_state[key] = value
    model.load_state_dict(compatible_state, strict=True)
    if remapped_count:
        print(f"Remapped {remapped_count} historical DINOv3 checkpoint keys")
    model = model.to(device).eval()
    return model, cfg


def run_touchanything_clips(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    frames_rgb: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    device: torch.device,
    batch_size: int,
    overlap: int,
    blend_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    clip_length = int(cfg["data"].get("clip_length", 8))
    windows = build_overlapping_windows(len(frames_rgb), clip_length, overlap)
    blend_weights = temporal_blend_weights(clip_length, blend_mode)
    transform = VideoTransform(
        image_size=tuple(cfg["data"]["image_size"]),
        normalize_mean=cfg["data"].get("normalize_mean", [0.485, 0.456, 0.406]),
        normalize_std=cfg["data"].get("normalize_std", [0.229, 0.224, 0.225]),
        use_augmentation=False,
    )
    coordinate_mode = cfg["data"].get("pose_coordinate_mode", "camera")
    prediction_sum: np.ndarray | None = None
    weight_sum = np.zeros(len(frames_rgb), dtype=np.float32)

    print(
        f"TouchAnything: {len(windows)} clips, length={clip_length}, "
        f"overlap={overlap}, blend={blend_mode}"
    )
    for batch_start in tqdm(
        range(0, len(windows), batch_size), desc="TouchAnything inference"
    ):
        batch_windows = windows[batch_start : batch_start + batch_size]
        batch_frames = []
        batch_poses = []
        batch_wrist_delta = []
        batch_wrist_valid = []
        for window in batch_windows:
            indices = window.indices
            batch_frames.append(transform(frames_rgb[indices]))
            clip_left = left_pose[indices]
            clip_right = right_pose[indices]
            clip_left_valid = left_valid[indices]
            clip_right_valid = right_valid[indices]
            wrist_delta, wrist_valid = compute_inter_wrist_delta(
                clip_left, clip_right, clip_left_valid, clip_right_valid
            )
            pose = prepare_bimanual_pose(
                clip_left,
                clip_right,
                clip_left_valid,
                clip_right_valid,
                coordinate_mode,
            )
            batch_poses.append(torch.from_numpy(pose))
            batch_wrist_delta.append(torch.from_numpy(wrist_delta))
            batch_wrist_valid.append(torch.from_numpy(wrist_valid))

        frames_tensor = torch.stack(batch_frames).to(device)
        poses_tensor = torch.stack(batch_poses).to(device)
        wrist_delta_tensor = torch.stack(batch_wrist_delta).to(device)
        wrist_valid_tensor = torch.stack(batch_wrist_valid).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            if getattr(model, "multi_view_enabled", False):
                outputs = model(
                    poses=poses_tensor,
                    inter_wrist_delta=wrist_delta_tensor,
                    inter_wrist_valid=wrist_valid_tensor,
                    views={"ego": frames_tensor},
                    view_valid={
                        "ego": torch.ones(
                            len(batch_windows), dtype=torch.bool, device=device
                        )
                    },
                )
            else:
                outputs = model(
                    frames=frames_tensor,
                    poses=poses_tensor,
                    inter_wrist_delta=wrist_delta_tensor,
                    inter_wrist_valid=wrist_valid_tensor,
                )
        predictions = outputs["tactile"].float().cpu().numpy()
        if prediction_sum is None:
            prediction_sum = np.zeros(
                (len(frames_rgb),) + predictions.shape[2:], dtype=np.float32
            )
        for prediction, window in zip(predictions, batch_windows):
            valid_indices = window.indices[: window.valid_length]
            weights = blend_weights[: window.valid_length]
            prediction_sum[valid_indices] += (
                prediction[: window.valid_length] * weights[:, None, None, None]
            )
            weight_sum[valid_indices] += weights

    if prediction_sum is None or np.any(weight_sum <= 0):
        raise RuntimeError("Overlap-add inference left one or more frames uncovered")
    predictions = prediction_sum / weight_sum[:, None, None, None]
    return predictions, weight_sum


def save_pose_jsonl(
    path: Path,
    source_indices: np.ndarray,
    poses: dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, source_index in enumerate(source_indices):
            record = {
                "frame_index": int(source_index),
                "left_pos": poses["left"][index].tolist()
                if poses["left_valid"][index]
                else [],
                "right_pos": poses["right"][index].tolist()
                if poses["right_valid"][index]
                else [],
                "left_valid": bool(poses["left_valid"][index]),
                "right_valid": bool(poses["right_valid"][index]),
                "left_detected": bool(poses["left_detected"][index]),
                "right_detected": bool(poses["right_detected"][index]),
                "left_interpolated": bool(poses["left_filled"][index]),
                "right_interpolated": bool(poses["right_filled"][index]),
                "left_score": float(poses["left_score"][index]),
                "right_score": float(poses["right_score"][index]),
            }
            handle.write(json.dumps(record) + "\n")


def render_prediction_frames(
    frames: Iterator[np.ndarray],
    output_path: Path,
    pressure: np.ndarray,
    fps: float,
) -> None:
    frame_iterator = iter(frames)
    try:
        first_frame = next(frame_iterator)
    except StopIteration as exc:
        raise ValueError("Cannot render an empty frame stream") from exc
    source_height, source_width = first_frame.shape[:2]
    camera_width = max(2, int(round(source_width * PRESS_H / source_height)))
    if camera_width % 2:
        camera_width += 1

    # render_pressure_panel uses a stable 720x360 layout.
    first_panel = render_pressure_panel(
        pred_to_21(pressure[0, 0], SENSOR_NAN_MASK_L),
        pred_to_21(pressure[0, 1], SENSOR_NAN_MASK_R),
        title="Predicted Pressure",
        vmax=1.0,
        global_max=1.0,
    )
    panel_width = first_panel.shape[1]
    output_width = camera_width + panel_width
    if output_width % 2:
        output_width += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{PRESS_H}", "-r", f"{fps:.6f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg input pipe")

    written = 0
    try:
        for frame_index, frame in enumerate(chain([first_frame], frame_iterator)):
            if frame_index >= len(pressure):
                break
            camera = cv2.resize(frame, (camera_width, PRESS_H))
            panel = render_pressure_panel(
                pred_to_21(pressure[frame_index, 0], SENSOR_NAN_MASK_L),
                pred_to_21(pressure[frame_index, 1], SENSOR_NAN_MASK_R),
                title="Predicted Pressure",
                vmax=1.0,
                global_max=1.0,
            )
            output = np.zeros((PRESS_H, output_width, 3), dtype=np.uint8)
            output[:, :camera_width] = camera
            output[:, camera_width : camera_width + panel_width] = panel
            cv2.putText(
                output,
                f"frame {frame_index}",
                (12, PRESS_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            process.stdin.write(np.ascontiguousarray(output).tobytes())
            written += 1
    finally:
        if not process.stdin.closed:
            process.stdin.close()
    return_code = process.wait()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr}")
    if written != len(pressure):
        raise RuntimeError(
            f"Rendered {written} frames but pressure timeline has {len(pressure)}"
        )


def render_prediction_video(
    source_video: Path,
    output_path: Path,
    pressure: np.ndarray,
    fps: float,
    max_frames: int | None,
) -> None:
    def frames() -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Failed to reopen video: {source_video}")
        frame_index = 0
        try:
            while max_frames is None or frame_index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
                frame_index += 1
        finally:
            cap.release()

    render_prediction_frames(frames(), output_path, pressure, fps)


def render_prediction_mcap(
    mcap_path: Path,
    topic: str,
    schema_name: str,
    output_path: Path,
    pressure: np.ndarray,
    fps: float,
    max_frames: int | None,
) -> None:
    frames = iter_mcap_frames(
        mcap_path,
        topic,
        schema_name,
        max_frames=max_frames,
    )
    render_prediction_frames(frames, output_path, pressure, fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiLoR hand pose -> TouchAnything tactile inference"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Input egocentric MP4")
    source.add_argument("--mcap", help="Input ROS2 MCAP")
    parser.add_argument(
        "--image-topic",
        default=None,
        help="MCAP ego Image/CompressedImage topic; auto-selected when unique",
    )
    parser.add_argument("--checkpoint", required=True, help="TouchAnything checkpoint")
    parser.add_argument("--config", required=True, help="Matching TouchAnything config")
    parser.add_argument("--output-dir", default="outputs/video_wilor_inference")
    parser.add_argument(
        "--wilor-dir", default="third_party/WiLoR"
    )
    parser.add_argument(
        "--wilor-checkpoint", default="pretrained_models/wilor_final.ckpt"
    )
    parser.add_argument(
        "--wilor-detector", default="pretrained_models/detector.pt"
    )
    parser.add_argument(
        "--mano-dir",
        default=None,
        help="Directory containing licensed MANO_RIGHT.pkl (default: WILOR_DIR/mano_data)",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--wilor-batch-size", type=int, default=32)
    parser.add_argument(
        "--clip-overlap",
        type=int,
        default=None,
        help="Adjacent clip overlap in target-FPS frames (default: half a clip)",
    )
    parser.add_argument("--blend-mode", choices=("hann", "uniform"), default="hann")
    parser.add_argument("--pose-max-gap", type=int, default=2)
    parser.add_argument("--pose-smooth-radius", type=int, default=2)
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wilor-fast", action="store_true")
    parser.add_argument("--wilor-compile", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.detector_batch_size <= 0 or args.wilor_batch_size <= 0:
        raise ValueError("All batch sizes must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if not 0.0 < args.detector_confidence <= 1.0:
        raise ValueError("detector_confidence must lie in (0, 1]")

    input_path = resolve_path(args.video if args.video is not None else args.mcap)
    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wilor_dir = resolve_path(args.wilor_dir)
    wilor_checkpoint = resolve_path(args.wilor_checkpoint, wilor_dir)
    wilor_detector = resolve_path(args.wilor_detector, wilor_dir)
    mano_dir = (
        resolve_path(args.mano_dir)
        if args.mano_dir is not None
        else wilor_dir / "mano_data"
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print("Loading TouchAnything...")
    model, cfg = load_touchanything(config_path, checkpoint_path, device)
    clip_length = int(cfg["data"].get("clip_length", 8))
    overlap = args.clip_overlap
    if overlap is None:
        overlap = clip_length // 2
    if not 0 <= overlap < clip_length:
        raise ValueError(
            f"clip_overlap must be in [0, {clip_length - 1}], got {overlap}"
        )

    image_topic: str | None = None
    image_schema: str | None = None
    if args.video is not None:
        input_kind = "video"
        cap = cv2.VideoCapture(str(input_path))
        source_fps_hint = (
            float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        )
        cap.release()
        frame_interval = resolve_frame_interval(
            source_fps_hint,
            cfg["data"].get("target_fps"),
            cfg["data"].get("frame_interval", 1),
        )
        (
            sampled_bgr,
            source_indices,
            source_frame_count,
            source_fps,
        ) = read_sampled_video(
            input_path, frame_interval=frame_interval, max_frames=args.max_frames
        )
        source_timestamps = np.arange(source_frame_count, dtype=np.float64) / source_fps
        effective_target_fps = source_fps / frame_interval
        print(
            f"Video: {source_frame_count} source frames at {source_fps:.3f} FPS; "
            f"sampled {len(sampled_bgr)} frames with interval {frame_interval}"
        )
    else:
        input_kind = "mcap"
        image_topic, image_schema = resolve_mcap_image_topic(
            input_path, args.image_topic
        )
        (
            sampled_bgr,
            source_indices,
            source_timestamps,
            source_fps,
            effective_target_fps,
        ) = read_sampled_mcap(
            input_path,
            topic=image_topic,
            schema_name=image_schema,
            target_fps=cfg["data"].get("target_fps"),
            fallback_interval=cfg["data"].get("frame_interval", 1),
            max_frames=args.max_frames,
        )
        source_frame_count = len(source_timestamps)
        frame_interval = max(1, int(round(source_fps / effective_target_fps)))
        print(
            f"MCAP: topic={image_topic} ({image_schema}), "
            f"{source_frame_count} source frames at {source_fps:.3f} FPS; "
            f"timestamp-sampled {len(sampled_bgr)} frames at "
            f"{effective_target_fps:.3f} FPS"
        )

    print("Loading WiLoR...")
    wilor = WiLoREstimator(
        wilor_dir=wilor_dir,
        checkpoint=wilor_checkpoint,
        detector_checkpoint=wilor_detector,
        mano_dir=mano_dir,
        device=device,
        fast=args.wilor_fast,
        compile_backbone=args.wilor_compile,
    )
    raw_pose = wilor.estimate(
        sampled_bgr,
        detector_batch_size=args.detector_batch_size,
        wilor_batch_size=args.wilor_batch_size,
        confidence=args.detector_confidence,
        rescale_factor=args.rescale_factor,
    )
    left_detected = raw_pose["left_valid"].copy()
    right_detected = raw_pose["right_valid"].copy()
    left, left_valid, left_filled = stabilize_pose_sequence(
        raw_pose["left"],
        left_detected,
        max_gap=args.pose_max_gap,
        smooth_radius=args.pose_smooth_radius,
    )
    right, right_valid, right_filled = stabilize_pose_sequence(
        raw_pose["right"],
        right_detected,
        max_gap=args.pose_max_gap,
        smooth_radius=args.pose_smooth_radius,
    )
    poses = {
        "left": left,
        "right": right,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "left_filled": left_filled,
        "right_filled": right_filled,
        "left_score": raw_pose["left_score"],
        "right_score": raw_pose["right_score"],
    }
    print(
        "Pose coverage: "
        f"left {left_detected.sum()}/{len(left_detected)} detected, "
        f"{left_valid.sum()}/{len(left_valid)} after stabilization; "
        f"right {right_detected.sum()}/{len(right_detected)} detected, "
        f"{right_valid.sum()}/{len(right_valid)} after stabilization"
    )
    save_pose_jsonl(output_dir / "wilor_poses.jsonl", source_indices, poses)

    sampled_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sampled_bgr])
    sampled_pressure, blend_weight = run_touchanything_clips(
        model=model,
        cfg=cfg,
        frames_rgb=sampled_rgb,
        left_pose=left,
        right_pose=right,
        left_valid=left_valid,
        right_valid=right_valid,
        device=device,
        batch_size=args.batch_size,
        overlap=overlap,
        blend_mode=args.blend_mode,
    )
    sampled_timestamps = source_timestamps[source_indices]
    source_pressure = interpolate_at_positions(
        sampled_pressure,
        sampled_timestamps,
        source_timestamps,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        pressure=source_pressure,
        sampled_pressure=sampled_pressure,
        sampled_source_indices=source_indices,
        source_timestamps=source_timestamps,
        sampled_timestamps=sampled_timestamps,
        blend_weight=blend_weight,
        left_pose=left,
        right_pose=right,
        left_valid=left_valid,
        right_valid=right_valid,
        left_detected=left_detected,
        right_detected=right_detected,
        left_interpolated=left_filled,
        right_interpolated=right_filled,
        source_fps=np.float32(source_fps),
        target_fps=np.float32(effective_target_fps),
        frame_interval=np.int32(frame_interval),
        input_kind=np.asarray(input_kind),
        image_topic=np.asarray(image_topic or ""),
    )
    if not args.no_video:
        if input_kind == "video":
            render_prediction_video(
                input_path,
                output_dir / "prediction.mp4",
                source_pressure,
                fps=source_fps,
                max_frames=args.max_frames,
            )
        else:
            assert image_topic is not None and image_schema is not None
            render_prediction_mcap(
                input_path,
                image_topic,
                image_schema,
                output_dir / "prediction.mp4",
                source_pressure,
                fps=source_fps,
                max_frames=args.max_frames,
            )
    print(f"Done. Results: {output_dir}")


if __name__ == "__main__":
    main()
