"""Утилиты для работы с директориями данных и поиска аудиофайлов."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def ensure_dirs(*dirs: Path | str) -> None:
    """Создаёт директории, если они не существуют."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def list_audio_files(root_dir: Path | str, valid_extensions: Iterable[str]) -> List[str]:
    """Рекурсивно ищет аудиофайлы с заданными расширениями в root_dir.

    Возвращает отсортированный список путей (строками), как в исходном ноутбуке.
    """
    root_dir = str(root_dir)
    found: List[str] = []
    if not os.path.exists(root_dir):
        return found
    valid_extensions = tuple(ext.lower() for ext in valid_extensions)
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                found.append(os.path.join(dirpath, filename))
    return sorted(found)
