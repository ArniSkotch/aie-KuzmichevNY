"""Скачивание и распаковка датасета TIMIT с Kaggle.

Перед использованием:
1. Скопируйте configs/.env.example -> configs/.env
2. Заполните KAGGLE_USERNAME и KAGGLE_KEY (https://www.kaggle.com/settings -> API)
3. НЕ коммитьте configs/.env в репозиторий (он уже добавлен в .gitignore).
"""
from __future__ import annotations

import os
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

DATASET_ID = "mfekadu/darpa-timit-acousticphonetic-continuous-speech"


def load_kaggle_credentials(env_path: str = "configs/.env", logger=None) -> tuple[str, str]:
    """Загружает KAGGLE_USERNAME/KAGGLE_KEY из configs/.env или окружения."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path)
        _log(logger, f"Конфиг загружен из {env_path}")
    except ImportError:
        _log(logger, "python-dotenv не установлен, пробуем переменные окружения напрямую", warn=True)

    kaggle_username = os.environ.get("KAGGLE_USERNAME", "")
    kaggle_key = os.environ.get("KAGGLE_KEY", "")

    if not kaggle_username or not kaggle_key:
        raise EnvironmentError(
            "Не найдены KAGGLE_USERNAME и/или KAGGLE_KEY.\n"
            f"Создайте файл {env_path} по образцу configs/.env.example."
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    _log(logger, "Kaggle credentials загружены из окружения.")
    return kaggle_username, kaggle_key


def download_and_extract_timit(dataset_root: Path | str, logger=None) -> Path:
    """Скачивает архив TIMIT с Kaggle (если его ещё нет) и распаковывает.

    Возвращает путь к распакованной директории.
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    # 1. Устанавливаем kaggle при необходимости.
    try:
        import kaggle  # noqa: F401

        _log(logger, "kaggle уже установлен")
    except ImportError:
        _log(logger, "Установка kaggle...")
        subprocess.run(["pip", "install", "kaggle", "-q"], check=True)
        import kaggle  # noqa: F401

        _log(logger, "kaggle установлен")

    # 2. Скачивание датасета.
    archive_path = dataset_root / "timit.zip"
    if not archive_path.exists():
        _log(logger, f"Скачивание {DATASET_ID}...")
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_ID, path=str(dataset_root), quiet=False, unzip=False)
        downloaded = list(dataset_root.glob("*.zip"))
        if downloaded:
            downloaded[0].rename(archive_path)
        _log(logger, "Датасет скачан")
    else:
        _log(logger, f"Архив уже существует: {archive_path}")

    # 3. Распаковка.
    extract_root = dataset_root / "extracted"
    if not extract_root.exists():
        _log(logger, "Распаковка архива...")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_root)
        _log(logger, f"Распаковано в {extract_root}")
    else:
        _log(logger, f"Уже распаковано: {extract_root}")

    _fix_permissions(extract_root)
    _log(logger, "Права на файлы исправлены")
    return extract_root


def _fix_permissions(extract_root: Path) -> None:
    """Исправляет права доступа после распаковки (zip может выставить read-only)."""
    for root, dirs, files in os.walk(extract_root):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IRWXU)
        for fname in files:
            os.chmod(os.path.join(root, fname), stat.S_IRUSR | stat.S_IWUSR)


def _log(logger, message: str, warn: bool = False) -> None:
    if logger is not None:
        logger.warning(message) if warn else logger.info(message)
    else:
        print(message)
