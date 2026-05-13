"""
logger.py — Centralized logging setup for the RAG project.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import yaml

# ── Load config once at module level ────────────────────────────
def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config_path = os.path.normpath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config     = _load_config()
_log_cfg    = _config.get("logging", {})
_path_cfg   = _config.get("paths", {})

LOG_LEVEL    = getattr(logging, _log_cfg.get("level", "INFO").upper(), logging.INFO)
LOG_FILE     = _log_cfg.get("log_file", "rag_app.log")
LOGS_DIR     = _path_cfg.get("logs_dir", "./logs")
MAX_BYTES    = _log_cfg.get("max_bytes", 5 * 1024 * 1024)   # 5 MB default
BACKUP_COUNT = _log_cfg.get("backup_count", 3)
LOG_FORMAT   = _log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
DATE_FORMAT  = _log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")


def _setup_root_logger() -> None:
    """
    Configure the root logger ONCE with:
      - RotatingFileHandler  → logs/<log_file>
      - StreamHandler        → console (stdout)
    Subsequent calls to get_logger() reuse this setup.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers on re-import
    if root.handlers:
        return

    root.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── File handler (rotating) ──────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, LOG_FILE)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    # ── Console handler ──────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


# Setup once on import
_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger inheriting the root configuration.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)
