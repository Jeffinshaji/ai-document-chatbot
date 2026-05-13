"""
config_loader.py — Loads config.yaml and exposes it as a typed dict.

Usage in any module:
    from src.config_loader import config
    model = config["llm"]["model_name"]
"""

import os
import yaml

def load_config() -> dict:
    """
    Locates and loads config.yaml from the project root.
    Raises FileNotFoundError if config.yaml is not found.
    """
    # Works regardless of where the script is run from
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(base_dir, "..", "config.yaml"))

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at expected path: {config_path}\n"
            "Make sure config.yaml is in the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ── Singleton — loaded once on import ───────────────────────────
config = load_config()
