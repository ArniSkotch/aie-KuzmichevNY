# audio_utils.py

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

# build_dataset.py


"""Парсинг файлов TIMIT (.PHN/.TXT) и сборка унифицированного датасета
вида input_audio/audio/*.wav + input_audio/annotations/*.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import librosa
import soundfile as sf

TIMIT_SR = 16000  # TIMIT всегда записан с частотой 16 кГц


def parse_phn_file(phn_path: Path, sample_rate: int = TIMIT_SR, logger=None) -> List[Dict[str, Any]]:
    """Читает TIMIT .PHN файл -> список сегментов с таймингами.

    Формат строк: start_sample end_sample phoneme, например: ``0 3050 h#``
    """
    segments: List[Dict[str, Any]] = []
    try:
        with open(phn_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start_s, end_s, phoneme = parts
                    segments.append({
                        "phoneme": phoneme.lower(),
                        "start_time": round(int(start_s) / sample_rate, 4),
                        "end_time": round(int(end_s) / sample_rate, 4),
                    })
    except Exception as e:
        if logger:
            logger.error(f"Ошибка чтения {phn_path}: {e}")
    return segments


def parse_txt_file(txt_path: Path, logger=None) -> str:
    """Читает TIMIT .TXT файл -> строка транскрипции.

    Формат: start_sample end_sample СЛОВО1 СЛОВО2 ...
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) > 2:
            return " ".join(parts[2:]).lower()
    except Exception as e:
        if logger:
            logger.error(f"Ошибка чтения {txt_path}: {e}")
    return ""


def build_dataset(
    extract_root: Path,
    audio_dir: Path,
    annotation_dir: Path,
    max_files: int = 300,
    random_sample: bool = True,
    random_seed: int = 42,
    logger=None,
) -> Dict[str, int]:
    """Конвертирует распакованный TIMIT в audio_dir/annotation_dir.

    Параметры:
        max_files     — сколько файлов взять (200–500 для репрезентативного EDA).
        random_sample — если True, перемешивает список перед нарезкой;
                        гарантирует покрытие всех 8 диалектов TIMIT.
        random_seed   — зерно для воспроизводимости.

    Возвращает {"processed": N, "skipped": M}.
    """
    import random as _random

    audio_dir = Path(audio_dir)
    annotation_dir = Path(annotation_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    all_wav = sorted(extract_root.rglob("*.WAV")) + sorted(extract_root.rglob("*.wav"))
    seen_stems = set()
    wav_files = []
    for w in all_wav:
        if w.stem.upper() not in seen_stems:
            seen_stems.add(w.stem.upper())
            wav_files.append(w)

    if logger:
        logger.info(f"Найдено WAV файлов: {len(wav_files)}")

    if random_sample:
        rng = _random.Random(random_seed)
        rng.shuffle(wav_files)
        if logger:
            logger.info(
                f"Случайная выборка: seed={random_seed}, берём первые {max_files} "
                f"из {len(wav_files)} (после перемешивания)"
            )

    processed = 0
    skipped = 0

    for wav_path in wav_files[:max_files]:
        try:
            stem = wav_path.stem
            speaker = wav_path.parent.name
            dialect = wav_path.parent.parent.name
            split = wav_path.parent.parent.parent.name

            phn_path = wav_path.with_suffix(".PHN")
            if not phn_path.exists():
                phn_path = wav_path.with_suffix(".phn")

            txt_path = wav_path.with_suffix(".TXT")
            if not txt_path.exists():
                txt_path = wav_path.with_suffix(".txt")

            segments = parse_phn_file(phn_path, logger=logger) if phn_path.exists() else []
            transcript = parse_txt_file(txt_path, logger=logger) if txt_path.exists() else ""

            if not segments:
                skipped += 1
                if logger:
                    logger.warning(f"Нет .PHN для {wav_path.name} - пропуск")
                continue

            try:
                y, sr = librosa.load(str(wav_path), sr=None)
            except Exception:
                y, sr = sf.read(str(wav_path))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y = y.astype("float32")

            file_id = f"{split}_{dialect}_{speaker}_{stem}"
            audio_out = audio_dir / f"{file_id}.wav"
            ann_out = annotation_dir / f"{file_id}.json"

            sf.write(str(audio_out), y, sr)

            annotation = {
                "file_id": file_id,
                "transcript": transcript,
                "sample_rate": sr,
                "speaker_id": speaker,
                "dialect": dialect,
                "split": split,
                "utterance": stem,
                "audio_path": str(audio_out),
                "segments": segments,
            }

            with open(ann_out, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            processed += 1
            if logger and processed % 5 == 0:
                logger.info(f"Обработано: {processed} файлов")

        except Exception as e:
            if logger:
                logger.error(f"Ошибка при обработке {wav_path}: {e}")

    if logger:
        logger.info(f"Итого обработано: {processed} файлов | пропущено: {skipped}")
        logger.info(f"Аудио: {audio_dir} | Аннотации: {annotation_dir}")

    return {"processed": processed, "skipped": skipped}

# download_timit.py

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
