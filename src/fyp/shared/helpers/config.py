""" config.py

- This file loads all configurations from config/config.yaml and resolve repo-relative paths
- Single source of truth for both architectures, so the values cannot drift apart
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


# Cached per resolved path, not globally. The previous version kept a single
# module-level dict and returned it for every later call regardless of `path`,
# so passing a different config file appeared to work and silently did nothing.
# URController.__init__ takes a path argument, which made it look configurable.
_CACHE: dict[Path, dict] = {}


def get_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    It takes a config path and gives you the parsed config, reading each distinct
    file at most once.

    Caching is per path so two different configs can coexist in one process,
    which is what the evaluation harness needs when it runs a trial against
    modified workspace bounds.
    """
    key = Path(path).resolve()
    if key not in _CACHE:
        _CACHE[key] = load_config(key)
    return _CACHE[key]


def clear_config_cache() -> None:
    """It takes nothing and gives you a cleared cache, so tests can reload from disk."""
    _CACHE.clear()


def resolve(rel_path) -> Path:
    return PROJECT_ROOT / rel_path
