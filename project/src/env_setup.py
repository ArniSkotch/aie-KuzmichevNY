"""Настройка переменных окружения, необходимых до импорта torchaudio/transformers.

Важно: HF_ENDPOINT и TORCHAUDIO_BACKEND должны быть выставлены ДО первого
импорта соответствующих библиотек, поэтому apply_environment() стоит вызывать
в самом начале точки входа (ноутбука или скрипта).
"""
from __future__ import annotations

import os
from typing import Any, Dict


def apply_environment(cfg: Dict[str, Any], logger=None) -> None:
    """Применяет переменные окружения из секции `env` конфига.

    Параметры:
        cfg: словарь конфигурации (см. src.config.load_config()).
        logger: опциональный логгер для вывода сообщений.
    """
    env_cfg = cfg.get("env", {})

    hf_endpoint = env_cfg.get("hf_endpoint")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        _log(logger, "Зеркало Hugging Face успешно настроено!")

    torchaudio_backend = env_cfg.get("torchaudio_backend")
    if torchaudio_backend:
        # Переменная должна быть установлена ДО первого импорта torchaudio.
        os.environ["TORCHAUDIO_BACKEND"] = torchaudio_backend
        _log(logger, f"TORCHAUDIO_BACKEND -> {torchaudio_backend}")

    espeak_path = env_cfg.get("espeak_path")
    if espeak_path:
        espeak_lib = os.path.join(espeak_path, "libespeak-ng.dll")
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + espeak_path
        _log(logger, "Путь к espeak настроен!")


def configure_torchaudio_backend(backend: str = "soundfile", logger=None) -> None:
    """Принудительно выбирает бэкенд torchaudio, если используется старый API."""
    import torchaudio

    try:
        torchaudio.set_audio_backend(backend)
        _log(logger, f"Backend: {backend} (legacy API)")
    except (AttributeError, RuntimeError):
        # torchaudio >= 2.1 управляет бэкендом через переменную TORCHAUDIO_BACKEND.
        _log(
            logger,
            "torchaudio >= 2.1: бэкенд управляется через переменную окружения TORCHAUDIO_BACKEND",
        )


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)
