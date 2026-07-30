""" config.py

- This file loads all configurations from config/config.yaml and resolve repo-relative paths
- Single source of truth for both the sim and the real robot, so that the values stay consistent
"""

import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3] # __file__ is this config file, parents[3] means ../../.. (3 levels up) back to project root (FYP)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml" # path of config wrt to project root, i.e. FYP


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


_CONFIG = None  # fake private variable only used for get_config

def get_config(path=DEFAULT_CONFIG_PATH) -> dict:
    global _CONFIG  # calling the global variable
    if _CONFIG is None:
        _CONFIG = load_config(path)
    return _CONFIG


def resolve(rel_path) -> Path:
    return PROJECT_ROOT / rel_path
