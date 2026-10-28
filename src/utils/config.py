import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configs, with override_config taking precedence.
    Recursively merges nested dictionaries.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def _expand_config_values(value):
    """Recursively expand environment variables and '~' in config strings."""
    if isinstance(value, dict):
        return {k: _expand_config_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_config_with_base(config_path: str, base_config_path: str = None) -> Dict[str, Any]:
    """
    Load config and optionally merge with base config.
    
    Example:
        config = load_config_with_base(
            'configs/hamer_pose_training_330_full.yaml',
            'configs/base.yaml',
        )
    """
    # Prefer an explicitly declared base.  This makes small experiment configs
    # composable while preserving the legacy neighboring-base fallback.
    specific_config = load_config(config_path)
    declared_base = specific_config.pop('_base_', None)
    if declared_base is not None and base_config_path is not None:
        raise ValueError(
            f"Config {config_path} declares _base_ while base_config_path was also provided"
        )

    if declared_base is not None:
        declared_path = Path(os.path.expandvars(os.path.expanduser(str(declared_base))))
        if not declared_path.is_absolute():
            declared_path = Path(config_path).parent / declared_path
        base_config = load_config_with_base(str(declared_path))
    elif base_config_path is None:
        config_dir = Path(config_path).parent
        default_base = config_dir / 'base.yaml'
        fallback_base = config_dir.parent / 'base.yaml'
        if default_base.exists() and default_base.resolve() != Path(config_path).resolve():
            base_config_path = default_base
        elif fallback_base.exists() and fallback_base.resolve() != Path(config_path).resolve():
            base_config_path = fallback_base
        else:
            base_config_path = None

        # Checkpoint directories contain an already-resolved config.yaml but do
        # not contain a neighboring base.yaml. Treat those as standalone.
        if base_config_path is not None and Path(base_config_path).exists():
            base_config = load_config_with_base(str(base_config_path))
        else:
            base_config = {}
    else:
        base_path = Path(base_config_path)
        base_config = load_config_with_base(str(base_path)) if base_path.exists() else {}
    
    # Merge configs
    merged_config = merge_configs(base_config, specific_config)
    merged_config = _expand_config_values(merged_config)

    return merged_config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
