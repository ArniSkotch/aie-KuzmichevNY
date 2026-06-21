"""Загрузка общей конфигурации проекта из configs/config.yaml.

Использование:
    from src.config import load_config
    cfg = load_config()
    print(cfg["paths"]["audio_dir"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Читает YAML-конфиг и возвращает словарь с настройками проекта."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {config_path}. "
            "Ожидается configs/config.yaml в корне проекта."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str, root: Path = PROJECT_ROOT) -> Path:
    """Преобразует относительный путь из конфига в абсолютный путь от корня проекта."""
    return (root / relative_path).resolve()
