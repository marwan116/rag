from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: Optional[str] = None):
    config_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).parent / "config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
