"""Загрузка и нормализация эталонных аннотаций (reference) для сравнения
с предсказаниями модели. Поддерживает форматы .json, .csv, .txt.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional


def normalize_label(label: Any) -> str:
    if label is None:
        return ""
    label = str(label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    return label


def segment_to_label(segment: Any) -> str:
    if isinstance(segment, dict):
        return normalize_label(
            segment.get("phoneme")
            or segment.get("label")
            or segment.get("symbol")
            or segment.get("token")
        )
    return normalize_label(segment)


def segment_has_timing(segment: Any) -> bool:
    return (
        isinstance(segment, dict)
        and segment.get("start_time") is not None
        and segment.get("end_time") is not None
    )


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transcript_to_phoneme_tokens(text: Optional[str]) -> List[str]:
    """Преобразует текст в список фонем через внешний espeak (если установлен)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        result = subprocess.run(
            ["espeak", "-q", "-x", text],
            capture_output=True,
            text=True,
            check=True,
        )
        return [tok for tok in result.stdout.strip().split() if tok.strip()]
    except Exception as exc:
        print(f"Не удалось преобразовать transcript в фонемы через espeak: {exc}")
        return []


def parse_segments_from_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    segments = []
    for row in rows:
        if isinstance(row, dict):
            phoneme = row.get("phoneme") or row.get("label") or row.get("symbol") or row.get("token")
            start_time = row.get("start_time", row.get("start"))
            end_time = row.get("end_time", row.get("end"))
        else:
            phoneme = str(row)
            start_time = None
            end_time = None

        seg: Dict[str, Any] = {"phoneme": normalize_label(phoneme)}
        if start_time is not None and end_time is not None:
            seg["start_time"] = float(start_time)
            seg["end_time"] = float(end_time)
        segments.append(seg)
    return segments


def find_annotation_file(
    audio_path: str,
    audio_dir: str,
    annotations_dir: str,
    valid_annotation_extensions: Iterable[str],
) -> Optional[str]:
    rel_path = os.path.relpath(audio_path, audio_dir)
    rel_dir = os.path.dirname(rel_path)
    stem = os.path.splitext(os.path.basename(rel_path))[0]

    candidates = []
    for ext in valid_annotation_extensions:
        candidates.append(os.path.join(annotations_dir, rel_dir, f"{stem}{ext}"))

    parts = stem.split("-")
    chapter_id = "-".join(parts[:-1]) if len(parts) >= 3 else None
    if chapter_id:
        candidates.append(os.path.join(annotations_dir, rel_dir, f"{chapter_id}.trans.txt"))
        candidates.append(os.path.join(annotations_dir, rel_dir, f"{chapter_id}.txt"))

    exact_names = {f"{stem}{ext}" for ext in valid_annotation_extensions}
    if chapter_id:
        exact_names.update({f"{chapter_id}.trans.txt", f"{chapter_id}.txt"})

    for dirpath, _, filenames in os.walk(annotations_dir):
        for filename in filenames:
            if filename in exact_names:
                candidates.append(os.path.join(dirpath, filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def load_reference_annotation(
    audio_path: str,
    audio_dir: str,
    annotations_dir: str,
    valid_annotation_extensions: Iterable[str],
) -> Dict[str, Any]:
    """Загружает эталонную разметку для аудиофайла, если она существует."""
    annotation_path = find_annotation_file(audio_path, audio_dir, annotations_dir, valid_annotation_extensions)

    if annotation_path is None:
        return {
            "annotation_path": None,
            "format": "missing",
            "transcript": None,
            "segments": [],
            "has_timing": False,
        }

    ext = annotation_path.lower()

    if ext.endswith(".json"):
        return _load_json_annotation(annotation_path)

    if ext.endswith(".csv") or ext.endswith(".txt"):
        return _load_csv_or_txt_annotation(annotation_path, audio_path)

    return {
        "annotation_path": annotation_path,
        "format": "unknown",
        "transcript": None,
        "segments": [],
        "has_timing": False,
    }


def _load_json_annotation(annotation_path: str) -> Dict[str, Any]:
    payload = load_json_file(annotation_path)
    transcript = None
    segments: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        transcript = payload.get("transcript") or payload.get("text")
        if isinstance(payload.get("segments"), list):
            segments = parse_segments_from_rows(payload["segments"])
        elif isinstance(payload.get("phonemes"), list):
            segments = parse_segments_from_rows(payload["phonemes"])
        elif isinstance(payload.get("labels"), list):
            segments = parse_segments_from_rows(payload["labels"])
    elif isinstance(payload, list):
        segments = parse_segments_from_rows(payload)

    has_timing = bool(segments) and all(segment_has_timing(s) for s in segments)
    if not segments and transcript:
        segments = [{"phoneme": p} for p in transcript_to_phoneme_tokens(transcript)]

    return {
        "annotation_path": annotation_path,
        "format": "json",
        "transcript": transcript,
        "segments": segments,
        "has_timing": has_timing,
    }


def _load_csv_or_txt_annotation(annotation_path: str, audio_path: str) -> Dict[str, Any]:
    transcript = None
    segments: List[Dict[str, Any]] = []

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    chapter_id = "-".join(base_name.split("-")[:-1]) if base_name.count("-") >= 2 else None

    structured_rows: List[Dict[str, Any]] = []
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames and any(
                key in [name.lower() for name in reader.fieldnames]
                for key in ["phoneme", "start_time", "end_time", "label", "symbol", "token", "transcript", "text"]
            ):
                for row in reader:
                    structured_rows.append(row)
    except Exception:
        structured_rows = []

    if structured_rows:
        lowered_fields = {k.lower(): k for k in structured_rows[0].keys()}
        if "transcript" in lowered_fields or "text" in lowered_fields:
            key = lowered_fields.get("transcript") or lowered_fields.get("text")
            transcript = " ".join(
                str(r.get(key, "")).strip() for r in structured_rows if str(r.get(key, "")).strip()
            )
        else:
            segments = parse_segments_from_rows(structured_rows)

    if not structured_rows or (not segments and not transcript):
        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        found_line = None
        if chapter_id:
            for line in lines:
                if line.startswith(base_name + " "):
                    found_line = line
                    break

        if found_line:
            transcript = found_line.split(" ", 1)[1].strip() if " " in found_line else ""
        elif len(lines) == 1:
            transcript = lines[0].strip()
        else:
            transcript = " ".join(line.strip() for line in lines)

    if not segments and transcript:
        segments = [{"phoneme": tok} for tok in transcript_to_phoneme_tokens(transcript)]

    has_timing = bool(segments) and all(segment_has_timing(s) for s in segments)

    return {
        "annotation_path": annotation_path,
        "format": "txt",
        "transcript": transcript,
        "segments": segments,
        "has_timing": has_timing,
    }
