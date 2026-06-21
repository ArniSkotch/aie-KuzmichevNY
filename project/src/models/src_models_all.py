# inference.py

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
        "dataset": "LibriSpeech-like",
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

# metrics.py

"""Метрики качества фонемного выравнивания: PER, precision/recall/F1,
ошибки границ сегментов (MAE/RMSE), IoU, точность в пределах допуска.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.features.annotation_loader import segment_has_timing, segment_to_label


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def safe_median(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def safe_rmse(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def align_sequences(reference_labels: List[str], predicted_labels: List[str]) -> Tuple[list, int]:
    """Выравнивание Левенштейна между эталонными и предсказанными метками.

    Возвращает (alignment, edit_distance), где alignment - список троек
    (op, ref_idx, pred_idx) с op в {"match", "sub", "ins", "del"}.
    """
    n = len(reference_labels)
    m = len(predicted_labels)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.empty((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i, 0] = "del"
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0, j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if reference_labels[i - 1] == predicted_labels[j - 1] else 1
            choices = [
                (dp[i - 1, j] + 1, "del"),
                (dp[i, j - 1] + 1, "ins"),
                (dp[i - 1, j - 1] + sub_cost, "match" if sub_cost == 0 else "sub"),
            ]
            best_cost, best_op = min(choices, key=lambda x: (x[0], 0 if x[1] == "match" else 1))
            dp[i, j] = best_cost
            back[i, j] = best_op

    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i, j]
        if op in ("match", "sub"):
            alignment.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "del":
            alignment.append((op, i - 1, None))
            i -= 1
        else:
            alignment.append((op, None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment, int(dp[n, m])


def compute_metrics(
    predicted_segments: List[Dict[str, Any]],
    reference_segments: List[Dict[str, Any]],
    tolerance_ms: Tuple[int, ...] = (10, 20, 50),
) -> Dict[str, Any]:
    """Считает полный набор метрик фонемного выравнивания для одного файла."""
    pred_labels = [segment_to_label(seg) for seg in predicted_segments]
    ref_labels = [segment_to_label(seg) for seg in reference_segments]

    alignment, edit_distance = align_sequences(ref_labels, pred_labels)

    matches = subs = ins = dels = 0
    timing_available_pairs = []

    for op, ref_idx, pred_idx in alignment:
        if op == "match":
            matches += 1
            ref_seg = reference_segments[ref_idx]
            pred_seg = predicted_segments[pred_idx]
            if segment_has_timing(ref_seg) and segment_has_timing(pred_seg):
                timing_available_pairs.append((ref_seg, pred_seg))
        elif op == "sub":
            subs += 1
        elif op == "ins":
            ins += 1
        elif op == "del":
            dels += 1

    ref_count = len(ref_labels)
    pred_count = len(pred_labels)

    precision = matches / pred_count if pred_count else None
    recall = matches / ref_count if ref_count else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    per = (subs + ins + dels) / ref_count if ref_count else None
    accuracy = matches / ref_count if ref_count else None
    seq_exact_match = int(ref_labels == pred_labels)

    boundary_start_errors, boundary_end_errors = [], []
    boundary_center_errors, boundary_min_errors = [], []
    duration_errors, duration_rel_errors, iou_scores = [], [], []

    for ref_seg, pred_seg in timing_available_pairs:
        ref_start = float(ref_seg["start_time"])
        ref_end = float(ref_seg["end_time"])
        pred_start = float(pred_seg["start_time"])
        pred_end = float(pred_seg["end_time"])

        start_err = abs(pred_start - ref_start)
        end_err = abs(pred_end - ref_end)
        center_err = abs(((pred_start + pred_end) / 2) - ((ref_start + ref_end) / 2))
        min_boundary_err = min(start_err, end_err)

        ref_dur = max(0.0, ref_end - ref_start)
        pred_dur = max(0.0, pred_end - pred_start)

        intersection = max(0.0, min(ref_end, pred_end) - max(ref_start, pred_start))
        union = max(ref_end, pred_end) - min(ref_start, pred_start)
        iou = (intersection / union) if union > 0 else None

        boundary_start_errors.append(start_err)
        boundary_end_errors.append(end_err)
        boundary_center_errors.append(center_err)
        boundary_min_errors.append(min_boundary_err)
        duration_errors.append(abs(pred_dur - ref_dur))
        duration_rel_errors.append((abs(pred_dur - ref_dur) / ref_dur) if ref_dur > 0 else None)
        iou_scores.append(iou)

    metrics: Dict[str, Any] = {
        "reference_count": ref_count,
        "predicted_count": pred_count,
        "edit_distance": edit_distance,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "matches": matches,
        "per": per,
        "phoneme_accuracy": accuracy,
        "phoneme_precision": precision,
        "phoneme_recall": recall,
        "phoneme_f1": f1,
        "sequence_exact_match": seq_exact_match,
        "matched_pairs_with_timing": len(timing_available_pairs),
        "reference_segments_with_timing": sum(1 for s in reference_segments if segment_has_timing(s)),
        "predicted_segments_with_timing": sum(1 for s in predicted_segments if segment_has_timing(s)),
        "boundary_start_mae": safe_mean(boundary_start_errors),
        "boundary_end_mae": safe_mean(boundary_end_errors),
        "boundary_center_mae": safe_mean(boundary_center_errors),
        "boundary_start_rmse": safe_rmse(boundary_start_errors),
        "boundary_end_rmse": safe_rmse(boundary_end_errors),
        "boundary_center_rmse": safe_rmse(boundary_center_errors),
        "boundary_start_median_ae": safe_median(boundary_start_errors),
        "boundary_end_median_ae": safe_median(boundary_end_errors),
        "boundary_center_median_ae": safe_median(boundary_center_errors),
        "minimal_boundary_error_mean": safe_mean(boundary_min_errors),
        "minimal_boundary_error_median": safe_median(boundary_min_errors),
        "minimal_boundary_error_max": max(boundary_min_errors) if boundary_min_errors else None,
        "duration_mae": safe_mean(duration_errors),
        "duration_rmse": safe_rmse(duration_errors),
        "duration_relative_mae": safe_mean(duration_rel_errors),
        "segment_iou_mean": safe_mean(iou_scores),
        "segment_iou_median": safe_median(iou_scores),
        "segment_iou_min": min(iou_scores) if iou_scores else None,
    }

    for tol in tolerance_ms:
        tol_sec = tol / 1000.0
        start_hits = [
            1 if abs(pred_seg["start_time"] - ref_seg["start_time"]) <= tol_sec else 0
            for ref_seg, pred_seg in timing_available_pairs
        ]
        end_hits = [
            1 if abs(pred_seg["end_time"] - ref_seg["end_time"]) <= tol_sec else 0
            for ref_seg, pred_seg in timing_available_pairs
        ]
        both_hits = [
            1
            if (
                abs(pred_seg["start_time"] - ref_seg["start_time"]) <= tol_sec
                and abs(pred_seg["end_time"] - ref_seg["end_time"]) <= tol_sec
            )
            else 0
            for ref_seg, pred_seg in timing_available_pairs
        ]

        metrics[f"boundary_start_accuracy_{tol}ms"] = safe_mean(start_hits)
        metrics[f"boundary_end_accuracy_{tol}ms"] = safe_mean(end_hits)
        metrics[f"boundary_exact_accuracy_{tol}ms"] = safe_mean(both_hits)

    return metrics


def summarize_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует метрики по всем обработанным файлам в единую сводку."""
    if not rows:
        return {}

    summary: Dict[str, Any] = {
        "files_processed": len(rows),
        "files_with_timing": sum(1 for r in rows if r.get("matched_pairs_with_timing", 0) > 0),
    }

    total_ref = sum(r.get("reference_count", 0) or 0 for r in rows)
    total_errors = sum(
        (r.get("substitutions", 0) or 0) + (r.get("insertions", 0) or 0) + (r.get("deletions", 0) or 0)
        for r in rows
    )
    summary["overall_per_weighted"] = (total_errors / total_ref) if total_ref else None

    for key in [
        "per",
        "phoneme_accuracy",
        "phoneme_precision",
        "phoneme_recall",
        "phoneme_f1",
        "boundary_start_mae",
        "boundary_end_mae",
        "boundary_center_mae",
        "boundary_start_rmse",
        "boundary_end_rmse",
        "boundary_center_rmse",
        "minimal_boundary_error_mean",
        "minimal_boundary_error_median",
        "duration_mae",
        "duration_rmse",
        "duration_relative_mae",
        "segment_iou_mean",
        "segment_iou_median",
        "sequence_exact_match",
        "boundary_exact_accuracy_20ms",
        "boundary_exact_accuracy_50ms",
    ]:
        values = [r.get(key) for r in rows if r.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None

    return summary


# model_loader.py

"""Загрузка модели Wav2Vec2 для фонемного распознавания."""
from __future__ import annotations

from typing import Tuple

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_model(model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft", logger=None) -> Tuple:
    """Загружает процессор и модель Wav2Vec2 в режиме инференса.

    Возвращает (processor, model, pad_token_id).
    """
    if logger:
        logger.info("Загрузка модели (это может занять пару минут при первом запуске)...")
    else:
        print("Загрузка модели (это может занять пару минут при первом запуске)...")

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()

    pad_token_id = processor.tokenizer.pad_token_id

    if logger:
        logger.info("Модель успешно загружена!")

    return processor, model, pad_token_id
