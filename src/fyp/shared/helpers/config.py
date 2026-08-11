""" config.py

- This file loads all configurations from config/config.yaml and resolve repo-relative paths
- Single source of truth for both the sim and the real robot, so that the values stay consistent
"""

import yaml
from pathlib import Path


def _find_project_root() -> Path:
    """
    It takes no arguments and gives you the repo root, found by walking up from
    this file until it sees pyproject.toml.

    This used to be a hardcoded `parents[3]`, which silently broke the moment
    the module moved one level deeper during the architecture_a/b/shared split.
    Every path in config.yaml is resolved relative to this, so a wrong answer
    here is a wrong answer everywhere, and it fails as a confusing missing-file
    error rather than as a path bug. Anchoring on a real marker survives moves.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"no pyproject.toml above {here}, cannot locate the project root")


PROJECT_ROOT = _find_project_root()
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
