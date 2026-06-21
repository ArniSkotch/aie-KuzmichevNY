
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
