import torch
import torch.nn as nn
from typing import Dict, Optional

from .vision_encoder import VisionEncoder
from .temporal_transformer import TemporalTransformer
from .pose_decoder import PoseDecoder, TactileDecoder, JointLevelTactileDecoder, build_tactile_decoder
from .pose_encoder import PoseEncoder, TransformerPoseEncoder, build_pose_encoder
from .fusion import ConcatFusion, CrossAttentionFusion, build_fusion
from .multi_view_encoder import MultiViewEncoder, build_multi_view_encoder
from .tactile_queries import (
    TactileQueryDecoder,
    TactileQueryFusion,
    TriModalAlignmentHead,
    roi_supervision_loss,
)


class TouchAnything(nn.Module):
    """
    Touch Anything: Vision-based Hand Pose and Tactile Prediction Model.
    
    Architecture:
        Single-view:
            Video -> VisionEncoder -> TemporalTransformer -> Decoder
        Multi-view:
            {ego, wrist_L, wrist_R} -> SharedEncoder + ViewEmbed + CrossViewAttn -> TemporalTransformer -> Decoder
        
        Pose prediction:    Visual features -> PoseDecoder -> (B,T,J,3)
        Tactile prediction: Visual features + Pose -> Fusion -> TactileDecoder -> (B,T,2,16,16)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.task = config['model']['task']
        self.use_pose_input = bool(config['model'].get('use_pose', True))
        embed_dim = config['model']['vision_encoder']['embed_dim']
        
        # Video Vision Encoder (Qwen3.5, DINOv2/v3, or Depth Anything V2)
        vision_cfg = config['model']['vision_encoder']
        self.vision_encoder = VisionEncoder(
            model_name=vision_cfg['model_name'],
            pretrained_path=vision_cfg.get('pretrained_path'),
            freeze=vision_cfg['freeze'],
            patch_size=vision_cfg['patch_size'],
            embed_dim=embed_dim,
            local_files_only=vision_cfg.get('local_files_only', True),
            load_dtype=vision_cfg.get('load_dtype', 'bfloat16'),
            attn_implementation=vision_cfg.get('attn_implementation', 'sdpa'),
            use_pooler_output=vision_cfg.get('use_pooler_output', True),
        )
        
        # Multi-view wrapper (optional)
        mv_cfg = config['model'].get('multi_view', {})
        self.multi_view_enabled = mv_cfg.get('enabled', False)
        if self.multi_view_enabled:
            self.multi_view_encoder = build_multi_view_encoder(
                self.vision_encoder, mv_cfg, embed_dim
            )
        
        # Temporal Transformer
        temporal_cfg = config['model']['temporal_transformer']
        self.temporal_transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_layers=temporal_cfg['num_layers'],
            num_heads=temporal_cfg['num_heads'],
            mlp_ratio=temporal_cfg['mlp_ratio'],
            dropout=temporal_cfg['dropout'],
            window_size=temporal_cfg['window_size'],
        )
        
        # Task-specific decoder
        if self.task == 'pose_prediction':
            pose_cfg = config['model']['pose_decoder']
            self.decoder = PoseDecoder(
                input_dim=embed_dim,
                hidden_dims=pose_cfg['hidden_dims'],
                output_dim=pose_cfg['output_dim'],
                dropout=pose_cfg['dropout'],
                decoder_type=pose_cfg['type'],
                num_query_layers=pose_cfg.get('num_query_layers', 2),
                num_query_heads=pose_cfg.get('num_query_heads', 8),
            )
        elif self.task == 'tactile_prediction':
            tactile_cfg = config['model'].get('tactile_decoder', {})
            num_joints = config['model'].get('num_input_joints', 48)
            
            # --- Pose Encoder ---
            if self.use_pose_input:
                pose_enc_cfg = config['model'].get('pose_encoder', {})
                # Backward compat: if no pose_encoder section, use old MLP defaults
                if not pose_enc_cfg or 'type' not in pose_enc_cfg:
                    pose_enc_cfg = {
                        'type': 'mlp',
                        'num_joints': num_joints,
                        'hidden_dims': tactile_cfg.get('pose_encoder_dims', [512, 512]),
                        'dropout': tactile_cfg.get('dropout', 0.1),
                    }
                pose_enc_cfg.setdefault('num_joints', num_joints)
                self.pose_encoder = build_pose_encoder(pose_enc_cfg, embed_dim)
            
            query_cfg = config['model'].get('tactile_queries', {})
            self.query_fusion_enabled = query_cfg.get('enabled', False)
            if self.query_fusion_enabled:
                num_queries = int(query_cfg.get('num_queries_per_hand', 64))
                self.tactile_query_fusion = TactileQueryFusion(
                    embed_dim=embed_dim,
                    num_queries_per_hand=num_queries,
                    num_heads=query_cfg.get('num_heads', 8),
                    num_layers=query_cfg.get('num_layers', 2),
                    mlp_ratio=query_cfg.get('mlp_ratio', 4.0),
                    dropout=query_cfg.get(
                        'dropout',
                        tactile_cfg.get('dropout', 0.1),
                    ),
                    roi_floor=query_cfg.get('roi_floor', 0.05),
                    use_pose=self.use_pose_input,
                )
                self.decoder = TactileQueryDecoder(
                    embed_dim=embed_dim,
                    num_queries_per_hand=num_queries,
                    hidden_channels=tactile_cfg.get('conv_channels', 128),
                    tactile_size=tactile_cfg.get('tactile_size', 21),
                    dropout=tactile_cfg.get('dropout', 0.1),
                )
                alignment_cfg = query_cfg.get('alignment', {})
                self.alignment_enabled = alignment_cfg.get('enabled', False)
                if self.alignment_enabled:
                    self.alignment_head = TriModalAlignmentHead(
                        embed_dim=embed_dim,
                        alignment_dim=alignment_cfg.get('dim', 256),
                        num_queries_per_hand=num_queries,
                        contact_threshold=alignment_cfg.get(
                            'contact_threshold',
                            config.get('training', {}).get(
                                'loss',
                                {},
                            ).get('pressure_threshold', 0.1),
                        ),
                        visual_touch_weight=alignment_cfg.get(
                            'visual_touch_weight',
                            0.25,
                        ),
                        pose_touch_weight=alignment_cfg.get(
                            'pose_touch_weight',
                            0.25,
                        ),
                        visual_pose_weight=alignment_cfg.get(
                            'visual_pose_weight',
                            0.1,
                        ),
                        reconstruction_weight=alignment_cfg.get(
                            'reconstruction_weight',
                            0.25,
                        ),
                        use_pose=self.use_pose_input,
                    )
                self.fusion_output_type = 'tactile_queries'
            else:
                if not self.use_pose_input:
                    raise ValueError(
                        "model.use_pose=false currently requires "
                        "model.tactile_queries.enabled=true"
                    )
                self.alignment_enabled = False
                # --- Legacy fusion path ---
                fusion_cfg = config['model'].get('fusion', {})
                if not fusion_cfg or 'type' not in fusion_cfg:
                    fusion_cfg = {'type': 'concat'}
                fusion_cfg.setdefault('dropout', tactile_cfg.get('dropout', 0.1))
                self.fusion = build_fusion(fusion_cfg, embed_dim)
                self.fusion_output_type = self.fusion.output_type
                if self.fusion_output_type == 'global':
                    self.fusion_proj = nn.Sequential(
                        nn.Linear(embed_dim * 2, embed_dim),
                        nn.LayerNorm(embed_dim),
                        nn.GELU(),
                    )
                dec_cfg = dict(tactile_cfg)
                if 'type' not in dec_cfg:
                    dec_cfg['type'] = 'conv'
                dec_cfg.setdefault('num_joints', num_joints)
                self.decoder = build_tactile_decoder(dec_cfg, embed_dim)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        print(f"[TouchAnything] Initialized with task={self.task}")
    
    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        inter_wrist_delta: Optional[torch.Tensor] = None,
        inter_wrist_valid: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
        view_valid: Optional[Dict[str, torch.Tensor]] = None,
        roi_masks: Optional[Dict[str, torch.Tensor]] = None,
        roi_valid: Optional[Dict[str, torch.Tensor]] = None,
        tactile_targets: Optional[torch.Tensor] = None,
        tactile_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W) - single-view video (backward compatible)
            poses: (B, T, J, 3) - hand poses (required for tactile_prediction)
            inter_wrist_delta: optional (B, T, 3) right-minus-left wrist position
            inter_wrist_valid: optional (B, T) validity mask for that relation
            views: dict of {view_name: (B, T, 3, H, W)} - multi-view input
                   view_name in ['ego', 'wrist_left', 'wrist_right']
                   Any subset can be provided at inference time.
            view_valid: optional dict of per-sample view availability masks (B,)
            roi_masks: optional egocentric left/right masks for training-time ROI supervision
            roi_valid: optional per-frame, per-hand ROI label availability
            tactile_targets: optional training target used only by the alignment teacher
            tactile_mask: optional physical-sensor mask for tactile teacher reconstruction
        
        Returns:
            outputs: dict containing predictions
                - 'poses': (B, T, J, 3) for pose prediction
                - 'tactile': (B, T, 2, S, S) continuous pressure prediction
                - 'pressure_logits': pre-sigmoid pressure output
                - 'contact': (B, T, 2, S, S) binary-contact probability
                - 'contact_logits': pre-sigmoid binary-contact output
                - 'roi_loss': optional hand-ROI auxiliary loss
                - 'alignment_loss': optional tri-modal SigLIP-style auxiliary loss
        """
        use_tactile_queries = (
            self.task == 'tactile_prediction'
            and getattr(self, 'query_fusion_enabled', False)
        )
        view_features = None
        if use_tactile_queries and self.multi_view_enabled and views is not None:
            view_features = self.multi_view_encoder.encode_views(
                views,
                view_valid=view_valid,
            )
            visual_features = None
        elif use_tactile_queries and frames is not None:
            view_features = {'ego': self.vision_encoder(frames)}
            visual_features = None
        elif use_tactile_queries and views is not None:
            ego_frames = views.get('ego', next(iter(views.values())))
            view_features = {'ego': self.vision_encoder(ego_frames)}
            visual_features = None
        elif self.multi_view_enabled and views is not None:
            visual_features = self.multi_view_encoder(
                views,
                view_valid=view_valid,
            )
        elif frames is not None:
            visual_features = self.vision_encoder(frames)  # (B, T, N, D)
        elif views is not None:
            # Multi-view input but multi_view not enabled: use ego view
            if 'ego' in views:
                visual_features = self.vision_encoder(views['ego'])
            else:
                visual_features = self.vision_encoder(next(iter(views.values())))
        else:
            raise ValueError("Must provide either 'frames' or 'views'")
        
        if self.task == 'pose_prediction':
            # Temporal modeling
            temporal_features = self.temporal_transformer(visual_features)  # (B, T, N, D)
            poses_out = self.decoder(temporal_features)  # (B, T, J, 3)
            return {'poses': poses_out}
        
        elif self.task == 'tactile_prediction':
            if self.use_pose_input:
                if poses is None:
                    raise ValueError("Pose input required when model.use_pose=true")
                pose_features = self.pose_encoder(
                    poses,
                    inter_wrist_delta=inter_wrist_delta,
                    inter_wrist_valid=inter_wrist_valid,
                )  # (B,T,D) or (B,T,J,D)
            else:
                pose_features = None

            if self.fusion_output_type == 'tactile_queries':
                query_output = self.tactile_query_fusion(
                    view_features,
                    pose_features,
                    view_valid=view_valid,
                )
                queries = query_output['queries']
                batch_size, num_frames, num_hands, num_queries, width = queries.shape
                temporal_queries = self.temporal_transformer(
                    queries.reshape(
                        batch_size,
                        num_frames,
                        num_hands * num_queries,
                        width,
                    )
                ).reshape(
                    batch_size,
                    num_frames,
                    num_hands,
                    num_queries,
                    width,
                )
                decoded = self.decoder(
                    temporal_queries,
                    return_outputs=True,
                )
                outputs = {
                    'tactile': decoded['pressure'],
                    'pressure_logits': decoded['pressure_logits'],
                    'contact': decoded['contact'],
                    'contact_logits': decoded['contact_logits'],
                    # Backward-compatible alias used by older train loops,
                    # where tactile_logits has always fed contact BCE.
                    'tactile_logits': decoded['contact_logits'],
                    'tactile_queries': temporal_queries,
                }

                if (
                    'ego_roi_logits' in query_output
                    and roi_masks is not None
                    and roi_valid is not None
                    and 'ego' in roi_masks
                    and 'ego' in roi_valid
                ):
                    outputs['roi_loss'] = roi_supervision_loss(
                        query_output['ego_roi_logits'],
                        roi_masks['ego'],
                        roi_valid['ego'],
                    )
                    outputs['ego_roi_logits'] = query_output['ego_roi_logits']

                if self.alignment_enabled and tactile_targets is not None:
                    alignment = self.alignment_head(
                        temporal_queries,
                        query_output['visual_context'],
                        query_output.get('pose_context'),
                        tactile_targets,
                        tactile_mask=tactile_mask,
                    )
                    outputs['alignment'] = alignment
                    outputs['alignment_loss'] = alignment['loss']
                return outputs

            elif self.fusion_output_type == 'global':
                # --- Original path: global fusion + temporal transformer ---
                # Pool pose features if per-joint
                if pose_features.dim() == 4:
                    pose_pooled = pose_features.mean(dim=2)  # (B, T, D)
                else:
                    pose_pooled = pose_features  # (B, T, D)
                
                visual_pooled = visual_features.mean(dim=2)  # (B, T, D)
                fused = torch.cat([visual_pooled, pose_pooled], dim=-1)  # (B, T, 2D)
                fused = self.fusion_proj(fused)  # (B, T, D)
                
                # Add fused token for temporal modeling
                fused_token = fused.unsqueeze(2)  # (B, T, 1, D)
                combined = torch.cat([visual_features, fused_token], dim=2)  # (B, T, N+1, D)
                temporal_features = self.temporal_transformer(combined)  # (B, T, N+1, D)
                tactile_features = temporal_features[:, :, -1, :]  # (B, T, D)
                
                tactile = self.decoder(tactile_features)  # (B, T, 2, 16, 16)
                return {'tactile': tactile}
            
            else:
                # --- New path: cross-attention fusion (per-joint) ---
                # Temporal modeling on visual features first
                temporal_visual = self.temporal_transformer(visual_features)  # (B, T, N, D)
                
                # Cross-attention: per-joint features query temporal visual features
                fused_joints = self.fusion(temporal_visual, pose_features)  # (B, T, J, D)
                
                # Decode from per-joint features
                tactile = self.decoder(fused_joints)  # (B, T, 2, 16, 16)
                return {'tactile': tactile}
    
    def freeze_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        print("[TouchAnything] Vision encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        self.vision_encoder.train()
        print("[TouchAnything] Vision encoder unfrozen")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters for each component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        temporal_params = sum(p.numel() for p in self.temporal_transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'temporal_transformer': temporal_params,
            'decoder': decoder_params,
        }


def build_model(config: dict) -> TouchAnything:
    """Build TouchAnything model from config."""
    model = TouchAnything(config)
    
    # Print model info
    param_counts = model.get_num_parameters()
    print("\n" + "="*60)
    print("Model Parameters:")
    print(f"  Total:      {param_counts['total']:,}")
    print(f"  Trainable:  {param_counts['trainable']:,}")
    print(f"  Vision:     {param_counts['vision_encoder']:,}")
    print(f"  Temporal:   {param_counts['temporal_transformer']:,}")
    print(f"  Decoder:    {param_counts['decoder']:,}")
    print("="*60 + "\n")
    
    return model
