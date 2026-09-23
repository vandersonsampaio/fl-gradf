"""Loads config/config.yaml and config/hyperparameters.yaml."""

import os
from typing import Dict, Optional

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")


def load_config(path: Optional[str] = None) -> Dict:
    """Loads config/config.yaml (FL parameters, datasets, GRADF thresholds)."""
    path = path or os.path.join(_CONFIG_DIR, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_hyperparameters(path: Optional[str] = None) -> Dict:
    """Loads config/hyperparameters.yaml (network/DQN hyperparameters)."""
    path = path or os.path.join(_CONFIG_DIR, "hyperparameters.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2, ensure_ascii=False))
    print(json.dumps(load_hyperparameters(), indent=2, ensure_ascii=False))
