"""Full Qwen3.5 multimodal reasoner for dense bimanual tactile prediction.

The existing TouchAnything stack extracts hand-local visual/pose query tokens.
This module projects those tokens into the complete Qwen3.5 text-backbone
width and uses the pretrained 32-layer hybrid language model as a latent
multimodal reasoner. Because Qwen3.5 mixes causal full attention with causal
Gated DeltaNet layers, offline bidirectionality is implemented with shared
weights over forward and time-reversed query streams rather than by changing
the pretrained attention semantics.
"""

from __future__ import annotations

import gc
from typing import Dict, Optional

import torch
import torch.nn as nn

from .clip_memory import CausalClipMemory
from .tactile_queries import roi_supervision_loss
from .touch_anything import TouchAnything


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported Qwen language dtype {name!r}; "
            "expected bf16, fp16, or fp32"
        )
    return mapping[normalized]


class BidirectionalQwenReasoner(nn.Module):
    """Adapt tactile-query embeddings to a pretrained Qwen text backbone."""

    def __init__(
        self,
        language_model: nn.Module,
        prompt_ids: torch.Tensor,
        query_dim: int,
        num_queries_per_hand: int,
        direction_mode: str = "bidirectional_shared",
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if direction_mode not in {"forward", "bidirectional_shared"}:
            raise ValueError(
                f"Unsupported direction_mode={direction_mode!r}; expected "
                "forward or bidirectional_shared"
            )
        self.language_model = language_model
        self.hidden_size = int(language_model.config.hidden_size)
        self.query_dim = int(query_dim)
        self.num_queries_per_hand = int(num_queries_per_hand)
        self.direction_mode = direction_mode

        self.query_in = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, self.hidden_size),
        )
        self.hand_embedding = nn.Parameter(
            torch.randn(1, 1, 2, 1, self.hidden_size) * 0.02
        )
        self.query_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                1,
                num_queries_per_hand,
                self.hidden_size,
            )
            * 0.02
        )
        if direction_mode == "bidirectional_shared":
            self.direction_gate = nn.Linear(
                self.hidden_size * 2,
                self.hidden_size,
            )
        else:
            self.direction_gate = None
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.query_out = nn.Sequential(
            nn.Linear(self.hidden_size, query_dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.register_buffer(
            "prompt_ids",
            prompt_ids.to(dtype=torch.long).reshape(1, -1),
            persistent=True,
        )

    def _language_forward(self, queries: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_hands, num_queries, hidden_size = queries.shape
        query_sequence = queries.reshape(
            batch_size,
            num_frames * num_hands * num_queries,
            hidden_size,
        )
        if self.prompt_ids.numel():
            prompt_ids = self.prompt_ids.expand(batch_size, -1)
            prompt = self.language_model.embed_tokens(prompt_ids)
            sequence = torch.cat([prompt, query_sequence], dim=1)
            prompt_length = prompt.shape[1]
        else:
            sequence = query_sequence
            prompt_length = 0

        attention_mask = torch.ones(
            sequence.shape[:2],
            dtype=torch.long,
            device=sequence.device,
        )
        output = self.language_model(
            inputs_embeds=sequence,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state[:, prompt_length:]
        return output.reshape(
            batch_size,
            num_frames,
            num_hands,
            num_queries,
            hidden_size,
        )

    def forward(self, tactile_queries: torch.Tensor) -> torch.Tensor:
        if tactile_queries.ndim != 5 or tactile_queries.shape[2] != 2:
            raise ValueError(
                "Expected tactile queries with shape (B,T,2,Q,D), got "
                f"{tuple(tactile_queries.shape)}"
            )
        if tactile_queries.shape[3] != self.num_queries_per_hand:
            raise ValueError(
                f"Expected {self.num_queries_per_hand} queries per hand, got "
                f"{tactile_queries.shape[3]}"
            )

        adapted = self.query_in(tactile_queries)
        adapted = adapted + self.hand_embedding + self.query_embedding
        if self.direction_mode == "bidirectional_shared":
            # Run the shared backbone once. Besides avoiding duplicate model
            # dispatch, this is important for DDP: every language parameter is
            # then used once per wrapped forward, so its reduction hook cannot
            # fire before the second directional contribution is accumulated.
            directional_batch = torch.cat(
                [adapted, adapted.flip(dims=[1])],
                dim=0,
            )
            directional_features = self._language_forward(directional_batch)
            forward_features, backward_features = directional_features.chunk(2, dim=0)
            backward_features = backward_features.flip(dims=[1])
            gate = torch.sigmoid(
                self.direction_gate(
                    torch.cat([forward_features, backward_features], dim=-1)
                )
            )
            reasoned = gate * forward_features + (1.0 - gate) * backward_features
        else:
            forward_features = self._language_forward(adapted)
            reasoned = forward_features

        update = self.query_out(self.output_norm(reasoned))
        return tactile_queries + self.residual_scale.to(update.dtype) * update


def build_clip_memory(memory_cfg, query_dim):
    """Build optional memory without shifting shared-module RNG in paired runs."""
    if not bool(memory_cfg.get("enabled", False)):
        return None
    seed = memory_cfg.get("initialization_seed")
    # Memory is initialized on CPU. Default configs preserve the old RNG behavior.
    from contextlib import nullcontext
    context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with context:
        if seed is not None:
            torch.random.default_generator.manual_seed(int(seed))
        return CausalClipMemory(
            embed_dim=query_dim,
            history_clips=int(memory_cfg.get("history_clips", 2)),
            memory_tokens_per_clip=int(memory_cfg.get("memory_tokens_per_clip", 16)),
            num_heads=int(memory_cfg.get("num_heads", 8)),
            num_layers=int(memory_cfg.get("num_layers", 2)),
            mlp_ratio=float(memory_cfg.get("mlp_ratio", 4.0)),
            dropout=float(memory_cfg.get("dropout", 0.1)),
            residual_scale=float(memory_cfg.get("residual_scale", 0.1)),
        )


class Qwen35FullTactileModel(nn.Module):
    """TouchAnything front/back ends with the full Qwen3.5 language reasoner."""

    def __init__(self, config: dict):
        super().__init__()
        reasoner_cfg = config.get("model", {}).get("qwen_full_reasoner", {})
        if not bool(reasoner_cfg.get("enabled", False)):
            raise ValueError("model.qwen_full_reasoner.enabled must be true")
        if config["model"].get("task") != "tactile_prediction":
            raise ValueError("Full Qwen tactile adaptation supports tactile_prediction only")
        if not config["model"].get("tactile_queries", {}).get("enabled", False):
            raise ValueError("Full Qwen tactile adaptation requires tactile_queries")

        self.config = config
        self.task_model = TouchAnything(config)
        # Replaced by the full Qwen reasoner; removing it avoids unused DDP
        # parameters while keeping every other baseline component unchanged.
        self.task_model.temporal_transformer = None
        # Dense tactile queries consume the unmerged 14x14 vision grid. Qwen's
        # original spatial merger is therefore outside this graph; freeze it
        # so DDP does not wait for gradients that can never be produced.
        vision_cfg = config["model"]["vision_encoder"]
        if not bool(vision_cfg.get("use_pooler_output", True)):
            vision_backbone = getattr(self.task_model.vision_encoder, "encoder", None)
            unused_merger = getattr(vision_backbone, "merger", None)
            if unused_merger is not None:
                unused_merger.requires_grad_(False)

        pretrained_path = reasoner_cfg.get(
            "pretrained_path",
            config["model"]["vision_encoder"].get("pretrained_path"),
        )
        if not pretrained_path:
            raise ValueError("qwen_full_reasoner.pretrained_path is required")
        load_dtype = _resolve_dtype(reasoner_cfg.get("load_dtype", "bfloat16"))

        try:
            from transformers import (
                AutoTokenizer,
                Qwen3_5ForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "Full Qwen3.5 tactile adaptation requires transformers>=5.9.0"
            ) from exc

        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            pretrained_path,
            local_files_only=bool(reasoner_cfg.get("local_files_only", True)),
            low_cpu_mem_usage=True,
            dtype=load_dtype,
            attn_implementation=reasoner_cfg.get("attn_implementation", "sdpa"),
        )
        language_model = full_model.model.language_model
        # Drop the duplicate visual module and token LM head. The task keeps
        # the high-resolution vision-only tower and uses text embeddings only
        # for the semantic task prompt.
        full_model.model.language_model = None
        del full_model
        gc.collect()

        freeze_language = bool(reasoner_cfg.get("freeze_language_model", False))
        language_model.requires_grad_(not freeze_language)
        if bool(reasoner_cfg.get("freeze_token_embeddings", True)):
            language_model.embed_tokens.requires_grad_(False)
        if bool(reasoner_cfg.get("gradient_checkpointing", True)):
            language_model.gradient_checkpointing_enable()
            language_model.config.use_cache = False

        prompt_text = str(
            reasoner_cfg.get(
                "prompt",
                "Predict bimanual tactile pressure from multi-view video and hand pose.",
            )
        )
        if bool(reasoner_cfg.get("use_prompt", True)):
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path,
                local_files_only=bool(reasoner_cfg.get("local_files_only", True)),
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
        else:
            prompt_ids = torch.empty(1, 0, dtype=torch.long)

        query_cfg = config["model"]["tactile_queries"]
        query_dim = int(config["model"]["vision_encoder"]["embed_dim"])
        self.reasoner = BidirectionalQwenReasoner(
            language_model=language_model,
            prompt_ids=prompt_ids,
            query_dim=query_dim,
            num_queries_per_hand=int(query_cfg.get("num_queries_per_hand", 64)),
            direction_mode=reasoner_cfg.get(
                "direction_mode",
                "bidirectional_shared",
            ),
            dropout=float(reasoner_cfg.get("dropout", 0.1)),
            residual_scale=float(reasoner_cfg.get("residual_scale", 0.1)),
        )

        self.clip_memory = build_clip_memory(
            config.get("model", {}).get("clip_memory", {}), query_dim
        )

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        language_total = sum(
            parameter.numel()
            for parameter in self.reasoner.language_model.parameters()
        )
        language_trainable = sum(
            parameter.numel()
            for parameter in self.reasoner.language_model.parameters()
            if parameter.requires_grad
        )
        print("\n" + "=" * 60)
        print("Full Qwen3.5 Tactile Model Parameters:")
        print(f"  Total:              {total:,}")
        print(f"  Trainable:          {trainable:,}")
        print(f"  Language total:     {language_total:,}")
        print(f"  Language trainable: {language_trainable:,}")
        print(f"  Direction mode:     {self.reasoner.direction_mode}")
        if self.clip_memory is not None:
            print(
                "  Clip memory:        "
                f"{self.clip_memory.history_clips} clips x "
                f"{self.clip_memory.memory_tokens_per_clip} tokens"
            )
        else:
            print("  Clip memory:        disabled")
        print("=" * 60 + "\n")

    @property
    def language_model(self):
        return self.reasoner.language_model

    def _encode_queries(
        self,
        frames: Optional[torch.Tensor],
        poses: Optional[torch.Tensor],
        views: Optional[Dict[str, torch.Tensor]],
        view_valid: Optional[Dict[str, torch.Tensor]],
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
    ):
        if self.task_model.multi_view_enabled and views is not None:
            view_features = self.task_model.multi_view_encoder.encode_views(
                views,
                view_valid=view_valid,
            )
        elif frames is not None:
            view_features = {"ego": self.task_model.vision_encoder(frames)}
        elif views is not None:
            ego = views.get("ego", next(iter(views.values())))
            view_features = {"ego": self.task_model.vision_encoder(ego)}
        else:
            raise ValueError("Must provide either frames or views")

        if self.task_model.use_pose_input:
            if poses is None:
                raise ValueError("Pose input required when model.use_pose=true")
            relation_kwargs = {}
            if inter_wrist_delta is not None or inter_wrist_valid is not None:
                relation_kwargs = {
                    "inter_wrist_delta": inter_wrist_delta,
                    "inter_wrist_valid": inter_wrist_valid,
                }
            pose_features = self.task_model.pose_encoder(poses, **relation_kwargs)
        else:
            pose_features = None
        return self.task_model.tactile_query_fusion(
            view_features,
            pose_features,
            view_valid=view_valid,
        )

    def _encode_current_and_history(
        self,
        frames: Optional[torch.Tensor],
        poses: Optional[torch.Tensor],
        views: Optional[Dict[str, torch.Tensor]],
        view_valid: Optional[Dict[str, torch.Tensor]],
        history_frames: Optional[torch.Tensor],
        history_poses: Optional[torch.Tensor],
        history_views: Optional[Dict[str, torch.Tensor]],
        history_view_valid: Optional[Dict[str, torch.Tensor]],
        history_valid: torch.Tensor,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
        history_inter_wrist_delta: Optional[torch.Tensor] = None,
        history_inter_wrist_valid: Optional[torch.Tensor] = None,
    ):
        """Encode all clips in one backbone call, then restore the clip axis."""
        batch_size, history_clips = history_valid.shape
        if self.clip_memory is None:
            raise RuntimeError("Historical encoding requires clip_memory")
        if history_clips != self.clip_memory.history_clips:
            raise ValueError(
                f"Expected {self.clip_memory.history_clips} historical clips, "
                f"got {history_clips}"
            )
        history_batch = batch_size * history_clips

        combined_frames = None
        combined_views = None
        combined_view_valid = None
        if views is not None:
            if history_views is None:
                raise ValueError("history_views are required with multi-view input")
            combined_views = {}
            for name, current_view in views.items():
                if name not in history_views:
                    raise ValueError(f"Missing historical view {name!r}")
                historical_view = history_views[name]
                expected = (batch_size, history_clips) + tuple(current_view.shape[1:])
                if tuple(historical_view.shape) != expected:
                    raise ValueError(
                        f"Expected history_views[{name!r}] shape {expected}, got "
                        f"{tuple(historical_view.shape)}"
                    )
                combined_views[name] = torch.cat(
                    [
                        historical_view.reshape(
                            history_batch,
                            *current_view.shape[1:],
                        ),
                        current_view,
                    ],
                    dim=0,
                )

            combined_view_valid = {}
            for name in combined_views:
                current_view = views[name]
                current_valid = (
                    view_valid[name]
                    if view_valid is not None and name in view_valid
                    else torch.ones(
                        batch_size,
                        dtype=torch.bool,
                        device=history_valid.device,
                    )
                ).to(device=current_view.device, dtype=torch.bool)
                historical_valid = (
                    history_view_valid[name]
                    if history_view_valid is not None and name in history_view_valid
                    else torch.ones(
                        batch_size,
                        history_clips,
                        dtype=torch.bool,
                        device=history_valid.device,
                    )
                ).to(device=current_view.device, dtype=torch.bool)
                combined_view_valid[name] = torch.cat(
                    [historical_valid.reshape(history_batch), current_valid],
                    dim=0,
                )
        else:
            if frames is None or history_frames is None:
                raise ValueError("history_frames are required with single-view input")
            expected = (batch_size, history_clips) + tuple(frames.shape[1:])
            if tuple(history_frames.shape) != expected:
                raise ValueError(
                    f"Expected history_frames shape {expected}, got "
                    f"{tuple(history_frames.shape)}"
                )
            combined_frames = torch.cat(
                [history_frames.reshape(history_batch, *frames.shape[1:]), frames],
                dim=0,
            )

        combined_poses = None
        if self.task_model.use_pose_input:
            if poses is None or history_poses is None:
                raise ValueError("history_poses are required when model.use_pose=true")
            expected = (batch_size, history_clips) + tuple(poses.shape[1:])
            if tuple(history_poses.shape) != expected:
                raise ValueError(
                    f"Expected history_poses shape {expected}, got "
                    f"{tuple(history_poses.shape)}"
                )
            combined_poses = torch.cat(
                [history_poses.reshape(history_batch, *poses.shape[1:]), poses],
                dim=0,
            )

        combined_inter_wrist_delta = None
        combined_inter_wrist_valid = None
        if (
            inter_wrist_delta is not None
            or history_inter_wrist_delta is not None
        ):
            if poses is None:
                raise ValueError("Inter-wrist inputs require pose inputs")
            num_frames = poses.shape[1]
            current_delta_shape = (batch_size, num_frames, 3)
            history_delta_shape = (
                batch_size,
                history_clips,
                num_frames,
                3,
            )
            if inter_wrist_delta is None:
                current_delta = poses.new_zeros(current_delta_shape)
                current_relation_valid = torch.zeros(
                    batch_size,
                    num_frames,
                    dtype=torch.bool,
                    device=poses.device,
                )
            else:
                if tuple(inter_wrist_delta.shape) != current_delta_shape:
                    raise ValueError(
                        "Expected inter_wrist_delta shape "
                        f"{current_delta_shape}, got {tuple(inter_wrist_delta.shape)}"
                    )
                current_delta = inter_wrist_delta.to(
                    device=poses.device,
                    dtype=poses.dtype,
                )
                current_relation_valid = (
                    inter_wrist_valid.to(device=poses.device, dtype=torch.bool)
                    if inter_wrist_valid is not None
                    else torch.ones(
                        batch_size,
                        num_frames,
                        dtype=torch.bool,
                        device=poses.device,
                    )
                )

            if history_inter_wrist_delta is None:
                historical_delta = poses.new_zeros(history_delta_shape)
                historical_relation_valid = torch.zeros(
                    batch_size,
                    history_clips,
                    num_frames,
                    dtype=torch.bool,
                    device=poses.device,
                )
            else:
                if tuple(history_inter_wrist_delta.shape) != history_delta_shape:
                    raise ValueError(
                        "Expected history_inter_wrist_delta shape "
                        f"{history_delta_shape}, got "
                        f"{tuple(history_inter_wrist_delta.shape)}"
                    )
                historical_delta = history_inter_wrist_delta.to(
                    device=poses.device,
                    dtype=poses.dtype,
                )
                historical_relation_valid = (
                    history_inter_wrist_valid.to(
                        device=poses.device,
                        dtype=torch.bool,
                    )
                    if history_inter_wrist_valid is not None
                    else torch.ones(
                        batch_size,
                        history_clips,
                        num_frames,
                        dtype=torch.bool,
                        device=poses.device,
                    )
                )
                historical_relation_valid = (
                    historical_relation_valid
                    & history_valid.to(device=poses.device, dtype=torch.bool).unsqueeze(-1)
                )

            combined_inter_wrist_delta = torch.cat(
                [
                    historical_delta.reshape(history_batch, num_frames, 3),
                    current_delta,
                ],
                dim=0,
            )
            combined_inter_wrist_valid = torch.cat(
                [
                    historical_relation_valid.reshape(history_batch, num_frames),
                    current_relation_valid,
                ],
                dim=0,
            )

        combined_output = self._encode_queries(
            combined_frames,
            combined_poses,
            combined_views,
            combined_view_valid,
            combined_inter_wrist_delta,
            combined_inter_wrist_valid,
        )
        history_queries = combined_output["queries"][:history_batch].reshape(
            batch_size,
            history_clips,
            *combined_output["queries"].shape[1:],
        )
        current_output = {
            name: value[history_batch:]
            for name, value in combined_output.items()
        }
        return current_output, history_queries

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
        history_frames: Optional[torch.Tensor] = None,
        history_poses: Optional[torch.Tensor] = None,
        history_inter_wrist_delta: Optional[torch.Tensor] = None,
        history_inter_wrist_valid: Optional[torch.Tensor] = None,
        history_views: Optional[Dict[str, torch.Tensor]] = None,
        history_view_valid: Optional[Dict[str, torch.Tensor]] = None,
        history_valid: Optional[torch.Tensor] = None,
        history_memory: Optional[torch.Tensor] = None,
        roi_masks: Optional[Dict[str, torch.Tensor]] = None,
        roi_valid: Optional[Dict[str, torch.Tensor]] = None,
        tactile_targets: Optional[torch.Tensor] = None,
        tactile_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        historical_queries = None
        if self.clip_memory is not None and (
            history_frames is not None or history_views is not None
        ):
            if history_valid is None:
                raise ValueError("history_valid is required with raw historical clips")
            query_output, historical_queries = self._encode_current_and_history(
                frames,
                poses,
                views,
                view_valid,
                history_frames,
                history_poses,
                history_views,
                history_view_valid,
                history_valid,
                inter_wrist_delta,
                inter_wrist_valid,
                history_inter_wrist_delta,
                history_inter_wrist_valid,
            )
        else:
            query_output = self._encode_queries(
                frames,
                poses,
                views,
                view_valid,
                inter_wrist_delta,
                inter_wrist_valid,
            )

        current_memory = None
        current_queries = query_output["queries"]
        if self.clip_memory is not None:
            if historical_queries is not None and history_memory is not None:
                raise ValueError(
                    "Provide raw historical clips or history_memory, not both"
                )
            if history_valid is None:
                history_valid = torch.zeros(
                    current_queries.shape[0],
                    self.clip_memory.history_clips,
                    dtype=torch.bool,
                    device=current_queries.device,
                )
            else:
                history_valid = history_valid.to(current_queries.device).bool()
            if historical_queries is None and history_memory is None:
                history_memory = current_queries.new_zeros(
                    current_queries.shape[0],
                    self.clip_memory.history_clips,
                    self.clip_memory.memory_tokens_per_clip,
                    self.clip_memory.embed_dim,
                )
            elif history_memory is not None:
                history_memory = history_memory.to(
                    device=current_queries.device,
                    dtype=current_queries.dtype,
                )
            current_queries, current_memory = self.clip_memory(
                current_queries,
                history_valid=history_valid,
                history_queries=historical_queries,
                history_memory=history_memory,
            )

        reasoned_queries = self.reasoner(current_queries)
        decoded = self.task_model.decoder(
            reasoned_queries,
            return_outputs=True,
        )
        outputs = {
            "tactile": decoded["pressure"],
            "pressure_logits": decoded["pressure_logits"],
            "contact": decoded["contact"],
            "contact_logits": decoded["contact_logits"],
            # Backward-compatible alias used by older train loops, where
            # tactile_logits has always fed contact BCE.
            "tactile_logits": decoded["contact_logits"],
            "tactile_queries": reasoned_queries,
        }
        if current_memory is not None:
            # Cache these tokens for the next streaming step. They summarize
            # only this clip, before any historical-memory update.
            outputs["clip_memory"] = current_memory

        if (
            "ego_roi_logits" in query_output
            and roi_masks is not None
            and roi_valid is not None
            and "ego" in roi_masks
            and "ego" in roi_valid
        ):
            outputs["roi_loss"] = roi_supervision_loss(
                query_output["ego_roi_logits"],
                roi_masks["ego"],
                roi_valid["ego"],
            )
            outputs["ego_roi_logits"] = query_output["ego_roi_logits"]

        if self.task_model.alignment_enabled and tactile_targets is not None:
            alignment = self.task_model.alignment_head(
                reasoned_queries,
                query_output["visual_context"],
                query_output.get("pose_context"),
                tactile_targets,
                tactile_mask=tactile_mask,
            )
            outputs["alignment"] = alignment
            outputs["alignment_loss"] = alignment["loss"]
        return outputs


def build_qwen35_full_tactile_model(config: dict) -> Qwen35FullTactileModel:
    return Qwen35FullTactileModel(config)
