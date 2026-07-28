"""Load config/config.yaml and resolve repo-relative paths.

Single source of truth for both the sim and the real robot, so a value can
never drift between them.
"""

import yaml
from pathlib import Path

# src/fyp/helpers/config.py -> parents: [0] helpers, [1] fyp, [2] src, [3] repo root.
# This index MUST track the file's depth. It was parents[2] when this module
# lived at src/fyp/config.py; moving it one level deeper made it parents[3].
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """load the project config from YAML into a dict"""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


_CONFIG = None


def get_config(path=DEFAULT_CONFIG_PATH) -> dict:
    global _CONFIG  # refers to the global _CONFIG variable above
    if _CONFIG is None:
        _CONFIG = load_config(path)
    return _CONFIG


def resolve(rel_path) -> Path:      # config-relative -> absolute
    return PROJECT_ROOT / rel_path
