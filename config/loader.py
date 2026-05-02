"""
config/loader.py
----------------
Loads and validates the YAML config. Provides a typed Settings object
accessible from anywhere in the system.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load YAML config from *path* (falls back to the bundled default).
    Environment variables override YAML values when prefixed with AKIS_.
    E.g.  AKIS_GENERATOR_PROVIDER=gemini  overrides generator.provider.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # Apply env-var overrides (flat key: AKIS_SECTION_KEY)
    for env_key, env_val in os.environ.items():
        if env_key.startswith("AKIS_"):
            parts = env_key[5:].lower().split("_", 1)
            if len(parts) == 2:
                section, key = parts
                if section in cfg and isinstance(cfg[section], dict):
                    cfg[section][key] = env_val

    return cfg


# Singleton — loaded once at import time
CONFIG: Dict[str, Any] = load_config()


def get(section: str, key: str, default: Any = None) -> Any:
    """Convenience accessor: get(section, key)."""
    return CONFIG.get(section, {}).get(key, default)
