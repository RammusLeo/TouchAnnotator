import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Tuple


def patchify_qwen35_video(
    frames: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(B,T,C,H,W)`` clips to the native Qwen3.5 video patch layout.

    This is the tensor-only equivalent of the Qwen video processor after resize
    and normalization. Keeping it in the model lets the existing dataloaders,
    augmentations, multi-view path, and DDP code continue to operate on regular
    video tensors without PIL/CPU round trips.
    """
    if frames.ndim != 5:
        raise ValueError(
            f"Qwen3.5 video input must have shape (B,T,C,H,W), got {tuple(frames.shape)}"
        )

    batch_size, num_frames, channels, height, width = frames.shape
    if channels != 3:
        raise ValueError(f"Qwen3.5 expects RGB video, got {channels} channels")
    if num_frames < 1:
        raise ValueError("Qwen3.5 video input must contain at least one frame")

    spatial_factor = patch_size * spatial_merge_size
    if height % spatial_factor != 0 or width % spatial_factor != 0:
        raise ValueError(
            "Qwen3.5 video height and width must be divisible by "
            f"patch_size * spatial_merge_size ({spatial_factor}); "
            f"got {(height, width)}"
        )

    # Match the official processor: repeat the last frame when the temporal
    # length is not divisible by the temporal tubelet size.
    temporal_pad = (-num_frames) % temporal_patch_size
    if temporal_pad:
        frames = torch.cat(
            [frames, frames[:, -1:].expand(-1, temporal_pad, -1, -1, -1)],
            dim=1,
        )

    grid_t = frames.shape[1] // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = frames.reshape(
        batch_size,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flattened = patches.reshape(
        batch_size * grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    grid_thw = torch.tensor(
        [grid_t, grid_h, grid_w],
        dtype=torch.long,
        device=frames.device,
    ).repeat(batch_size, 1)
    return flattened, grid_thw


class VisionEncoder(nn.Module):
    """
    Video vision encoder using Qwen3.5, DINOv2/v3, or Depth Anything V2.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        patch_size: int = 14,
        embed_dim: int = 384,
        local_files_only: bool = True,
        load_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        use_pooler_output: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.local_files_only = local_files_only
        self.load_dtype = load_dtype
        self.attn_implementation = attn_implementation
        self.use_pooler_output = use_pooler_output

        normalized_name = model_name.lower().replace('-', '_')
        self.is_qwen35 = any(
            marker in normalized_name
            for marker in ('qwen3.5', 'qwen3_5', 'qwen35')
        )
        
        # Load encoder based on model type
        if self.is_qwen35:
            self.encoder = self._load_qwen35(
                model_name=model_name,
                pretrained_path=pretrained_path,
                local_files_only=local_files_only,
                load_dtype=load_dtype,
                attn_implementation=attn_implementation,
            )
            self._configure_qwen35_adapter()
        elif model_name.startswith('depth_anything'):
            self.encoder = self._load_depth_anything_v2(model_name, pretrained_path)
        elif model_name.startswith('dinov3'):
            self.encoder = self._load_dinov3(model_name, pretrained_path)
        else:
            self.encoder = self._load_dinov2(model_name, pretrained_path)
        
        # Freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        print(f"[VisionEncoder] Loaded {model_name}, embed_dim={embed_dim}, freeze={freeze}")
    
    def train(self, mode: bool = True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype:
        dtype_map = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        try:
            return dtype_map[str(dtype_name).lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Qwen3.5 load_dtype={dtype_name!r}; "
                f"choose one of {sorted(dtype_map)}"
            ) from exc

    def _load_qwen35(
        self,
        model_name: str,
        pretrained_path: Optional[str],
        local_files_only: bool,
        load_dtype: str,
        attn_implementation: str,
    ):
        """Load Qwen3.5 and retain its native video vision tower only."""
        if pretrained_path is None:
            raise ValueError(
                f"Cannot load {model_name}: pretrained_path must be a local "
                "Qwen3.5 directory or a Hugging Face model id."
            )
        if local_files_only and not os.path.isdir(pretrained_path):
            raise ValueError(
                f"Cannot load {model_name}: local Qwen3.5 directory does not exist: "
                f"{pretrained_path}"
            )

        try:
            from transformers import (
                AutoConfig,
                Qwen3_5ForConditionalGeneration,
                Qwen3_5VisionModel,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3.5 video encoding requires transformers>=5.9.0. "
                "Install the project environment before building this model."
            ) from exc

        requested_dtype = self._resolve_dtype(load_dtype)
        vision_only_path = (
            os.path.join(pretrained_path, 'vision_encoder.safetensors')
            if os.path.isdir(pretrained_path)
            else None
        )
        if vision_only_path is not None and os.path.isfile(vision_only_path):
            try:
                from safetensors.torch import load_file

                full_config = AutoConfig.from_pretrained(
                    pretrained_path,
                    local_files_only=True,
                )
                vision_config = full_config.vision_config
                vision_config._attn_implementation = attn_implementation
                encoder = Qwen3_5VisionModel(vision_config)
                state_dict = load_file(vision_only_path, device='cpu')
                encoder.load_state_dict(state_dict, strict=True)
                encoder.to(dtype=requested_dtype)
            except Exception as exc:
                raise ValueError(
                    f"Cannot load Qwen3.5 vision-only checkpoint "
                    f"{vision_only_path}: {exc}"
                ) from exc
            print(
                f"[VisionEncoder] Loaded Qwen3.5 vision-only checkpoint "
                f"from {vision_only_path}"
            )
            return encoder

        load_kwargs = {
            'local_files_only': local_files_only,
            'low_cpu_mem_usage': True,
            'dtype': requested_dtype,
            'attn_implementation': attn_implementation,
        }
        try:
            full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                pretrained_path,
                **load_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot load Qwen3.5 model {model_name} from {pretrained_path}: {exc}"
            ) from exc

        # Loading the complete checkpoint is the most version-stable way to
        # resolve sharded/safetensors prefixes. The language model is released
        # immediately; only the pretrained video tower remains registered.
        encoder = full_model.model.visual
        del full_model
        print(f"[VisionEncoder] Loaded Qwen3.5 video tower from {pretrained_path}")
        return encoder

    def _configure_qwen35_adapter(self):
        vision_config = self.encoder.config
        actual_patch_size = int(vision_config.patch_size)
        if actual_patch_size != self.patch_size:
            raise ValueError(
                f"Qwen3.5 patch_size ({actual_patch_size}) does not match configured "
                f"patch_size ({self.patch_size})."
            )

        self.temporal_patch_size = int(vision_config.temporal_patch_size)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        if self.use_pooler_output:
            native_dim = int(vision_config.out_hidden_size)
        else:
            native_dim = int(vision_config.hidden_size)
        self.native_embed_dim = native_dim

        projection_layers = []
        if native_dim != self.embed_dim:
            projection_layers.append(nn.Linear(native_dim, self.embed_dim))
        projection_layers.append(nn.LayerNorm(self.embed_dim))
        self.output_projection = nn.Sequential(*projection_layers)

        print(
            "[VisionEncoder] Qwen3.5 video config: "
            f"temporal_patch={self.temporal_patch_size}, "
            f"spatial_merge={self.spatial_merge_size}, "
            f"native_dim={native_dim}, output_dim={self.embed_dim}"
        )
    
    def _load_dinov2(self, model_name: str, pretrained_path: Optional[str] = None):
        """Build DINOv2 offline and load an explicit local checkpoint.

        Historical checkpoints in this repository were built with either the
        official torch.hub implementation or timm.  Prefer an existing hub
        checkout when available and otherwise use timm's architecture-only
        constructor; neither path is allowed to download weights.
        """
        hub_dir = torch.hub.get_dir()
        local_repo_candidates = [
            os.path.join(hub_dir, 'facebookresearch_dinov2_main'),
            os.path.join(hub_dir, 'facebookresearch_dinov2_master'),
        ]
        local_repo = next((p for p in local_repo_candidates if os.path.isdir(p)), None)

        if local_repo is not None:
            try:
                encoder = torch.hub.load(
                    local_repo, model_name, source='local', pretrained=False
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot build local DINOv2 model {model_name} from "
                    f"{local_repo}: {e}"
                ) from e
        else:
            timm_names = {
                'dinov2_vits14': 'vit_small_patch14_dinov2',
                'dinov2_vitb14': 'vit_base_patch14_dinov2',
                'dinov2_vitl14': 'vit_large_patch14_dinov2',
            }
            timm_name = timm_names.get(model_name)
            if timm_name is None:
                raise ValueError(f"Unsupported DINOv2 model: {model_name}")
            try:
                import timm
                encoder = timm.create_model(
                    timm_name,
                    pretrained=False,
                    dynamic_img_size=True,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot build offline timm DINOv2 model {model_name}: {e}"
                ) from e
            print(
                f"[VisionEncoder] Built {model_name} with offline timm fallback; "
                f"torch.hub cache was absent under {hub_dir}"
            )

        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"[VisionEncoder] Loading from local checkpoint: {pretrained_path}")
            state_dict = torch.load(
                pretrained_path, map_location='cpu', weights_only=True
            )
            if isinstance(state_dict, dict):
                state_dict = state_dict.get(
                    'state_dict', state_dict.get('model', state_dict)
                )
            # The official DINOv2 pretraining checkpoint contains a mask token
            # that timm omits because downstream feature extraction never uses
            # masked-image pretraining.
            if local_repo is None:
                state_dict.pop('mask_token', None)
                incompatible = encoder.load_state_dict(state_dict, strict=False)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        f"DINOv2 checkpoint mismatch: missing="
                        f"{incompatible.missing_keys}, unexpected="
                        f"{incompatible.unexpected_keys}"
                    )
            else:
                encoder.load_state_dict(state_dict, strict=True)
            print(f"[VisionEncoder] Successfully loaded {model_name} from local checkpoint")
            return encoder

        print(f"[VisionEncoder] Loaded {model_name} from local hub cache: {local_repo}")
        return encoder

    def _load_dinov3(self, model_name: str, pretrained_path: Optional[str] = None):
        """Load a DINOv3 Hugging Face checkpoint from a local directory."""
        if pretrained_path is None:
            raise ValueError(
                f"Cannot load {model_name}: pretrained_path must point to a local "
                "DINOv3 Hugging Face model directory."
            )
        if not os.path.isdir(pretrained_path):
            raise ValueError(
                f"Cannot load {model_name}: DINOv3 model directory does not exist: "
                f"{pretrained_path}"
            )

        try:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(
                pretrained_path,
                local_files_only=True,
            )
        except Exception as e:
            raise ValueError(
                f"Cannot load local DINOv3 model {model_name} from {pretrained_path}: {e}"
            ) from e

        hidden_size = getattr(encoder.config, 'hidden_size', None)
        patch_size = getattr(encoder.config, 'patch_size', None)
        if hidden_size != self.embed_dim:
            raise ValueError(
                f"DINOv3 hidden_size ({hidden_size}) does not match configured "
                f"embed_dim ({self.embed_dim})."
            )
        if patch_size != self.patch_size:
            raise ValueError(
                f"DINOv3 patch_size ({patch_size}) does not match configured "
                f"patch_size ({self.patch_size})."
            )

        print(f"[VisionEncoder] Successfully loaded {model_name} from {pretrained_path}")
        return encoder
    
    def _load_depth_anything_v2(self, model_name: str, pretrained_path: Optional[str] = None):
        """Load Depth Anything V2 encoder (fine-tuned DINOv2)."""
        # Map model_name to DA-V2 encoder variant
        variant_map = {
            'depth_anything_v2_vits': 'vits',
            'depth_anything_v2_vitb': 'vitb',
            'depth_anything_v2_vitl': 'vitl',
        }
        variant = variant_map.get(model_name)
        if variant is None:
            raise ValueError(f"Unknown DA-V2 model: {model_name}. Available: {list(variant_map.keys())}")
        
        # Import DA-V2's own DinoVisionTransformer (handles pos_embed for img_size=518)
        da_v2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ref', 'Depth-Anything-V2')
        if da_v2_path not in sys.path:
            sys.path.insert(0, da_v2_path)
        from depth_anything_v2.dinov2 import DINOv2 as DA_DINOv2
        
        # Build architecture using DA-V2's factory (img_size=518, patch_size=14)
        encoder = DA_DINOv2(model_name=variant)
        
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"[VisionEncoder] Loading DA-V2 from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Extract only encoder weights (pretrained.*) and strip prefix
            encoder_state = {}
            for k, v in state_dict.items():
                if k.startswith('pretrained.'):
                    encoder_state[k.replace('pretrained.', '')] = v
            encoder.load_state_dict(encoder_state, strict=True)
            print(f"[VisionEncoder] Successfully loaded DA-V2 {variant} encoder ({len(encoder_state)} params)")
        else:
            print(f"[VisionEncoder] WARNING: No pretrained weights for DA-V2, using random init")
        
        return encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, H, W) - batch of video clips
        
        Returns:
            features: (B, T, N, D) - frame-wise features
                N = number of patches (e.g., 16x16=256 for 224x224 image with patch_size=14)
                D = embed_dim
        """
        if self.is_qwen35:
            return self._forward_qwen35_video(x)

        B, T, C, H, W = x.shape
        
        # Reshape to (B*T, 3, H, W) for batch processing
        x = x.view(B * T, C, H, W)
        
        # Extract features. Hugging Face DINOv3 returns all tokens in
        # last_hidden_state; DINOv2/Depth Anything use forward_features().
        if self.model_name.startswith('dinov3'):
            if self.freeze:
                with torch.no_grad():
                    output = self.encoder(pixel_values=x)
            else:
                output = self.encoder(pixel_values=x)

            num_register_tokens = getattr(self.encoder.config, 'num_register_tokens', 0)
            patch_token_start = 1 + num_register_tokens  # CLS + register tokens
            features = output.last_hidden_state[:, patch_token_start:, :]
        else:
            if self.freeze:
                with torch.no_grad():
                    output = self.encoder.forward_features(x)
            else:
                output = self.encoder.forward_features(x)

            # DINOv2 returns a dict, extract patch tokens (without CLS token)
            if isinstance(output, dict):
                features = output['x_norm_patchtokens']  # (B*T, N, D)
            else:
                # Fallback: if it's a tensor, remove CLS token manually
                features = output[:, 1:, :]  # (B*T, N, D)
        
        # Reshape back to (B, T, N, D)
        N = features.shape[1]
        D = features.shape[2]
        features = features.view(B, T, N, D)
        
        return features

    def _forward_qwen35_video(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a video clip with Qwen3.5 and align tubelets to each frame."""
        batch_size, num_frames, _, height, width = x.shape
        flattened, grid_thw = patchify_qwen35_video(
            x,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
        )
        # The frozen tower is normally stored in bf16 while dataloaders emit
        # fp32 tensors. Do not require global autocast merely to satisfy the
        # patch embedding dtype contract.
        first_encoder_parameter = next(self.encoder.parameters(), None)
        encoder_dtype = (
            first_encoder_parameter.dtype
            if first_encoder_parameter is not None
            else flattened.dtype
        )
        flattened = flattened.to(dtype=encoder_dtype)

        if self.freeze:
            with torch.no_grad():
                output = self.encoder(
                    flattened,
                    grid_thw=grid_thw,
                    return_dict=True,
                )
        else:
            output = self.encoder(
                flattened,
                grid_thw=grid_thw,
                return_dict=True,
            )

        if self.use_pooler_output:
            features = output.pooler_output
            spatial_tokens = (
                (height // self.patch_size // self.spatial_merge_size)
                * (width // self.patch_size // self.spatial_merge_size)
            )
        else:
            features = output.last_hidden_state
            spatial_tokens = (
                (height // self.patch_size) * (width // self.patch_size)
            )

        grid_t = (num_frames + self.temporal_patch_size - 1) // self.temporal_patch_size
        expected_tokens = batch_size * grid_t * spatial_tokens
        if features.shape[0] != expected_tokens:
            raise RuntimeError(
                "Unexpected Qwen3.5 video token count: "
                f"got {features.shape[0]}, expected {expected_tokens} "
                f"for grid={(grid_t, height, width)}"
            )

        features = features.reshape(
            batch_size,
            grid_t,
            spatial_tokens,
            self.native_embed_dim,
        )
        # The frozen tower commonly runs in bf16 while the trainable adapter is
        # fp32. Explicit conversion also keeps non-autocast validation working.
        projection_dtype = next(self.output_projection.parameters()).dtype
        features = self.output_projection(features.to(projection_dtype))

        # Qwen3.5 represents consecutive frames with one temporal tubelet.
        # Repeat each tubelet feature so downstream pose fusion and the tactile
        # decoder retain the original per-frame output contract.
        features = features.repeat_interleave(self.temporal_patch_size, dim=1)
        return features[:, :num_frames]
    
    def get_num_patches(self, image_size: int) -> int:
        """Calculate number of patches for given image size."""
        if self.is_qwen35 and self.use_pooler_output:
            merged_patch = self.patch_size * self.spatial_merge_size
            return (image_size // merged_patch) ** 2
        return (image_size // self.patch_size) ** 2


class DINOv2Config:
    """Configuration for supported vision transformer models."""
    
    CONFIGS = {
        'dinov2_vits14': {'embed_dim': 384, 'num_heads': 6, 'depth': 12},
        'dinov2_vitb14': {'embed_dim': 768, 'num_heads': 12, 'depth': 12},
        'dinov2_vitl14': {'embed_dim': 1024, 'num_heads': 16, 'depth': 24},
        'dinov3_vitb16': {'embed_dim': 768, 'num_heads': 12, 'depth': 12},
        'qwen3.5_2b': {'embed_dim': 768, 'num_heads': 16, 'depth': 24},
        'qwen3.5_4b': {'embed_dim': 768, 'num_heads': 16, 'depth': 24},
        'depth_anything_v2_vits': {'embed_dim': 384, 'num_heads': 6, 'depth': 12},
        'depth_anything_v2_vitb': {'embed_dim': 768, 'num_heads': 12, 'depth': 12},
        'depth_anything_v2_vitl': {'embed_dim': 1024, 'num_heads': 16, 'depth': 24},
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> dict:
        if model_name not in cls.CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[model_name]
