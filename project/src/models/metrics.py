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
