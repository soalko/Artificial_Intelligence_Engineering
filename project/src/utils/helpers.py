import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_config(config_name="config.yaml"):
    path = PROJECT_ROOT / "configs" / config_name
    with open(path, "r") as f:
        return yaml.safe_load(f)