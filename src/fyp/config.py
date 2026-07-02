import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """load the project config from YAML into a dict"""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
