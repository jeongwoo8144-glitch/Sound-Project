"""Shared configuration and logging utilities.

Single source of truth for config loading and logger setup — all pipeline
modules import from here instead of duplicating the same two functions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml (str or Path).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict, name: str | None = None) -> logging.Logger:
    """Configure root logging from config and return a named logger.

    Args:
        cfg: Full parsed config dictionary (reads ``cfg["logging"]``).
        name: Logger name. If ``None``, uses the calling module's ``__name__``.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s – %(message)s")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = log_cfg.get("file")
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger(name or __name__)


def get_class_map(cfg: dict) -> dict[int, str]:
    """Return ``{class_id: class_name}`` for ALL classes (양성 + background).

    양성 클래스는 ID 오름차순, background(99)는 맨 마지막에 배치됩니다.
    background_classes가 config에 있으면 'background' 항목을 자동 추가합니다.

    Args:
        cfg: Full parsed config dictionary.

    Returns:
        Ordered dict: e.g. {1: 'car_horn', 8: 'siren', 99: 'background'}
    """
    BACKGROUND_ID = 99
    ds = cfg["dataset"]

    target = ds["target_classes"]
    result = dict(sorted({int(cid): name for name, cid in target.items()}.items()))

    if ds.get("background_classes"):
        result[BACKGROUND_ID] = "background"

    return result


def get_class_names(cfg: dict) -> list[str]:
    """Return class names sorted by class ID (dense-label order).

    Args:
        cfg: Full parsed config dictionary.

    Returns:
        List of class name strings, e.g. ``["car_horn", "siren"]``.
    """
    return list(get_class_map(cfg).values())


def get_num_classes(cfg: dict) -> int:
    """Return the number of target classes defined in config.

    Args:
        cfg: Full parsed config dictionary.

    Returns:
        Integer count of target classes.
    """
    return len(cfg["dataset"]["target_classes"])
