"""Load config/config.yaml and resolve repo-relative paths.

Single source of truth for both the sim and the real robot, so a value can
never drift between them.
"""

import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


_CONFIG = None


def get_config(path=DEFAULT_CONFIG_PATH) -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config(path)
    return _CONFIG


def resolve(rel_path) -> Path:
    return PROJECT_ROOT / rel_path
