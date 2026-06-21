"""Инференс Wav2Vec2 по одному аудиофайлу: получение фонем с таймингами,
акустическая коррекция границ и сравнение с эталонной аннотацией.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import librosa
import soundfile as sf
import torch

from src.features.annotation_loader import load_reference_annotation


def extract_and_save_phonemes(
    filepath: str,
    output_subfolder: str,
    model,
    processor,
    pad_token_id: int,
    audio_dir: str,
    annotations_dir: str,
    valid_annotation_extensions,
    frame_duration: float = 0.02,
    top_db: int = 35,
) -> Dict[str, Any]:
    """Прогоняет аудиофайл через модель, извлекает фонемы с таймингами,
    сравнивает с эталонной разметкой (если есть) и сохраняет результат
    в output_subfolder/metadata.json + отдельные .wav-нарезки по фонемам.
    """
    os.makedirs(output_subfolder, exist_ok=True)

    y_orig, sr_orig = librosa.load(filepath, sr=None)
    y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=16000)

    inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()

    segments = []
    current_token = None
    start_frame = 0

    for i, token_id in enumerate(predicted_ids):
        if token_id != current_token:
            if current_token is not None and current_token != pad_token_id:
                phoneme_symbol = processor.decode([current_token]).strip()
                segments.append({
                    "phoneme": phoneme_symbol,
                    "start_time": start_frame * frame_duration,
                    "end_time": i * frame_duration,
                })
            current_token = token_id
            start_frame = i

    if current_token is not None and current_token != pad_token_id:
        phoneme_symbol = processor.decode([current_token]).strip()
        segments.append({
            "phoneme": phoneme_symbol,
            "start_time": start_frame * frame_duration,
            "end_time": len(predicted_ids) * frame_duration,
        })

    if not segments:
        return {
            "source_file": os.path.basename(filepath),
            "audio_path": filepath,
            "prediction": {"segments": [], "count": 0},
            "reference": None,
            "metrics": None,
        }

    corrected_segments = correct_segment_boundaries(segments, y_orig, sr_orig, frame_duration, top_db)

    source_filename = os.path.basename(filepath)
    reference_info = load_reference_annotation(filepath, audio_dir, annotations_dir, valid_annotation_extensions)
    annotation_path = reference_info["annotation_path"]

    metadata: Dict[str, Any] = {
        "version": 2,
        "dataset": "TIMIT",
        "source_file": source_filename,
        "audio_path": filepath,
        "annotation_path": annotation_path,
        "prediction": {
            "sample_rate": sr_orig,
            "frame_duration": frame_duration,
            "segments": corrected_segments,
            "count": len(corrected_segments),
        },
        "reference": {
            "annotation_path": reference_info.get("annotation_path"),
            "format": reference_info.get("format"),
            "transcript": reference_info.get("transcript"),
            "segments": reference_info.get("segments", []),
            "count": len(reference_info.get("segments", [])),
            "has_timing": reference_info.get("has_timing", False),
        },
        "metrics": None,
    }

    metadata["phonemes"] = metadata["prediction"]["segments"]

    _save_phoneme_slices(corrected_segments, y_orig, sr_orig, output_subfolder)

    with open(os.path.join(output_subfolder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return metadata


def correct_segment_boundaries(segments, y_orig, sr_orig, frame_duration, top_db):
    """Уточняет границы фонем по акустическим (не тихим) интервалам сигнала."""
    non_mute_intervals = librosa.effects.split(y_orig, top_db=top_db)
    acoustic_intervals = [{"start": s / sr_orig, "end": e / sr_orig} for s, e in non_mute_intervals]

    _, trim_indices = librosa.effects.trim(y_orig, top_db=top_db)
    audio_content_end = trim_indices[1] / sr_orig

    def find_acoustic_interval(phoneme_start_time):
        tolerance = 0.05
        for interval in acoustic_intervals:
            if interval["start"] - tolerance <= phoneme_start_time <= interval["end"] + tolerance:
                return interval
        return None

    corrected_segments = []
    for i, seg in enumerate(segments):
        start_time = seg["start_time"]
        end_time = seg["end_time"]

        current_interval = find_acoustic_interval(start_time)

        if i < len(segments) - 1:
            next_start = segments[i + 1]["start_time"]
            next_in_same_interval = (
                current_interval is not None and find_acoustic_interval(next_start) is current_interval
            )
            if next_in_same_interval:
                end_time = next_start
            elif current_interval is not None:
                end_time = current_interval["end"]
            else:
                end_time = start_time + 0.04
        elif current_interval is not None:
            end_time = current_interval["end"]

        end_time = min(end_time, audio_content_end)
        end_time = max(end_time, start_time + frame_duration)

        corrected_segments.append({
            "phoneme": seg["phoneme"],
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
        })

    return corrected_segments


def _save_phoneme_slices(segments, y_orig, sr_orig, output_subfolder) -> None:
    """Сохраняет аудио-нарезки по каждой найденной фонеме (для ручной проверки)."""
    for idx, seg in enumerate(segments):
        start_sample = int(seg["start_time"] * sr_orig)
        end_sample = int(seg["end_time"] * sr_orig)

        audio_slice = y_orig[start_sample:end_sample]

        safe_phoneme = "".join(c for c in seg["phoneme"] if c.isalnum() or c in ("_", "-"))
        if not safe_phoneme:
            safe_phoneme = "unk"

        filename = f"{idx:04d}_{safe_phoneme}.wav"
        save_path = os.path.join(output_subfolder, filename)

        sf.write(save_path, audio_slice, sr_orig)
        seg["file"] = filename
