import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from typing import List, Tuple


class VideoTransform:
    """
    Transform for video clips.
    Ensures temporal consistency: all frames in a clip share the same
    random augmentation parameters.
    Optimized: uses batch tensor operations instead of per-frame transforms.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
        use_augmentation: bool = True,
        random_crop: bool = False,
        color_jitter: float = 0.0,
        glove_augmentation=None,
    ):
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.random_crop = random_crop and use_augmentation
        self.color_jitter = color_jitter if use_augmentation else 0.0
        self.glove_augmentation = glove_augmentation
        
        # Resize size: larger than target if using random crop
        if self.random_crop:
            self.resize_size = (int(image_size[0] * 1.15), int(image_size[1] * 1.15))
        else:
            self.resize_size = image_size
        
        # Pre-compute normalize tensors for batch operation
        self.mean = torch.tensor(normalize_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(normalize_std).view(1, 3, 1, 1)
    
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W, 3) numpy array, range [0, 255]
        
        Returns:
            transformed_frames: (T, 3, H, W) tensor, normalized
        """
        # Convert to tensor: (T, H, W, 3) -> (T, 3, H, W), [0, 1]
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        
        # Resize (skip if already target size, e.g. decord pre-resized)
        current_size = (frames_tensor.shape[2], frames_tensor.shape[3])
        if current_size != self.resize_size:
            frames_tensor = TF.resize(frames_tensor, self.resize_size)
        
        # Random crop - batch slice (same crop for all frames)
        if self.random_crop:
            h, w = frames_tensor.shape[2], frames_tensor.shape[3]
            th, tw = self.image_size
            crop_i = random.randint(0, h - th)
            crop_j = random.randint(0, w - tw)
            frames_tensor = frames_tensor[:, :, crop_i:crop_i+th, crop_j:crop_j+tw]
        
        # Color jitter - batch operations (same params for all frames)
        if self.color_jitter > 0:
            jitter_params = T.ColorJitter.get_params(
                brightness=(max(0, 1 - self.color_jitter), 1 + self.color_jitter),
                contrast=(max(0, 1 - self.color_jitter), 1 + self.color_jitter),
                saturation=(max(0, 1 - self.color_jitter), 1 + self.color_jitter),
                hue=(-self.color_jitter / 2, self.color_jitter / 2),
            )
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_params
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    # brightness: multiply all pixels
                    frames_tensor = frames_tensor * brightness_factor
                elif fn_id == 1 and contrast_factor is not None:
                    # contrast: blend with per-frame gray mean
                    gray = frames_tensor.mean(dim=1, keepdim=True)  # (T, 1, H, W)
                    gray_mean = gray.mean(dim=(2, 3), keepdim=True)  # (T, 1, 1, 1)
                    frames_tensor = contrast_factor * frames_tensor + (1 - contrast_factor) * gray_mean
                elif fn_id == 2 and saturation_factor is not None:
                    # saturation: blend with grayscale
                    gray = frames_tensor.mean(dim=1, keepdim=True).expand_as(frames_tensor)
                    frames_tensor = saturation_factor * frames_tensor + (1 - saturation_factor) * gray
                elif fn_id == 3 and hue_factor is not None and abs(hue_factor) > 1e-6:
                    # hue: apply per-frame (unavoidable, but rare with small hue range)
                    frames_tensor = torch.stack([
                        TF.adjust_hue(frames_tensor[t], hue_factor) for t in range(frames_tensor.shape[0])
                    ])
            frames_tensor = frames_tensor.clamp(0.0, 1.0)
        
        # Normalize - batch operation
        frames_tensor = (frames_tensor - self.mean) / self.std
        
        return frames_tensor  # (T, 3, H, W)


def get_transforms(config: dict, is_training: bool = True):
    """Get transforms based on config."""
    # Get glove augmentation if enabled
    from .glove_augmentation import get_glove_augmentation
    glove_aug = get_glove_augmentation(config, is_training)
    
    return VideoTransform(
        image_size=tuple(config['data']['image_size']),
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std'],
        use_augmentation=is_training and config['data']['use_augmentation'],
        random_crop=is_training and config['data']['random_crop'],
        color_jitter=config['data']['color_jitter'] if is_training else 0.0,
        glove_augmentation=glove_aug,
    )
