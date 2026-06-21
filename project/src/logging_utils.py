"""Настройка логирования для всего проекта."""
from __future__ import annotations

import logging


def setup_logger(name: str = "voice_model", level: int = logging.INFO) -> logging.Logger:
    """Настраивает и возвращает логгер с единым форматом для всего проекта."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
