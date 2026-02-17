"""Utilities for YAML-driven CLI configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            'PyYAML is required for --config support. Install with: pip install pyyaml'
        ) from exc

    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML config must be a mapping at root: {path}')
    return data


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested config dict by taking leaf keys.

    Example:
      {'data': {'processed_dir': './x'}, 'training': {'epochs': 10}}
      -> {'processed_dir': './x', 'epochs': 10}
    """
    out: Dict[str, Any] = {}

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    _walk(value)
                else:
                    out[key] = value

    _walk(config)
    return out


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Iterable[str]] = None,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse args with optional YAML config defaults.

    Priority (lowest -> highest):
      parser defaults < YAML config values < explicit CLI flags
    """
    dests = {a.dest for a in parser._actions if getattr(a, 'dest', None)}

    if 'config' not in dests:
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Path to YAML config file',
        )

    preload = argparse.ArgumentParser(add_help=False)
    preload.add_argument('--config', type=str, default=None)
    pre_args, _ = preload.parse_known_args(argv)

    loaded_config: Dict[str, Any] = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f'Config file not found: {cfg_path}')

        loaded_config = _load_yaml(cfg_path)
        flat = _flatten_config(loaded_config)
        valid = {k: v for k, v in flat.items() if k in {a.dest for a in parser._actions}}
        if valid:
            parser.set_defaults(**valid)

    args = parser.parse_args(argv)
    return args, loaded_config
