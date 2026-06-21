"""Сравнение baseline и улучшенной модели — критерий 4 чек-листа.

Вариант B: одна модель, два режима обработки.

  Baseline:   facebook/wav2vec2-lv-60-espeak-cv-ft, БЕЗ коррекции границ
              (сырые CTC-предсказания — стандартное поведение модели)
  Улучшенная: facebook/wav2vec2-lv-60-espeak-cv-ft, С коррекцией границ
              (акустическая доработка через librosa.effects.split/trim)

Запуск:
    python -m src.compare_models

Результаты сохраняются в:
    data/processed/output_phonemes/baseline/
    data/processed/output_phonemes/improved/
    data/processed/output_phonemes/model_comparison.csv
    data/processed/output_phonemes/model_comparison.json
    data/processed/output_phonemes/model_comparison_plot.png
    data/processed/output_phonemes/boundary_accuracy_by_dialect.png
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve_path
from src.data.audio_utils import ensure_dirs, list_audio_files
from src.env_setup import apply_environment
from src.features.annotation_loader import load_reference_annotation
from src.logging_utils import setup_logger
from src.models.inference import correct_segment_boundaries
from src.models.metrics import compute_metrics, summarize_metric_rows
from src.models.model_loader import load_model

# ─────────────────────────────────────────────────────────────
# Параметры сравнения
# ─────────────────────────────────────────────────────────────

BASELINE_LABEL = "baseline"
IMPROVED_LABEL = "improved"

BASELINE_DESC = (
    "Wav2Vec2-LV60 (CTC без постобработки) — "
    "сырые границы фонем напрямую из argmax CTC-логитов"
)
IMPROVED_DESC = (
    "Wav2Vec2-LV60 + акустическая коррекция границ — "
    "librosa.effects.split/trim уточняет начало/конец каждого сегмента"
)

# ─────────────────────────────────────────────────────────────
# Утилита: извлечь диалект из имени файла
# ─────────────────────────────────────────────────────────────

def _dialect_from_filename(source_file: str) -> str:
    """TRAIN_DR2_MCPM0_SI552.wav → 'DR2'; если формат другой → 'unknown'."""
    stem = Path(source_file).stem          # убираем .wav
    parts = stem.split("_")
    for part in parts:
        if part.upper().startswith("DR") and len(part) == 3:
            return part.upper()
    return "unknown"


# ─────────────────────────────────────────────────────────────
# Инференс с двумя вариантами за один проход
# ─────────────────────────────────────────────────────────────

def _ctc_decode(
    filepath: str,
    processor,
    model,
    pad_token_id: int,
    frame_duration: float,
) -> Tuple[List[Dict[str, Any]], Any, int]:
    """Загружает аудио, прогоняет через CTC, возвращает
    (raw_segments, y_orig, sr_orig) — до любой коррекции."""
    y_orig, sr_orig = librosa.load(filepath, sr=None)
    y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=16000)

    inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()

    raw_segments: List[Dict[str, Any]] = []
    current_token = None
    start_frame = 0

    for i, token_id in enumerate(predicted_ids):
        if token_id != current_token:
            if current_token is not None and current_token != pad_token_id:
                raw_segments.append({
                    "phoneme":    processor.decode([current_token]).strip(),
                    "start_time": round(start_frame * frame_duration, 3),
                    "end_time":   round(i * frame_duration, 3),
                })
            current_token = token_id
            start_frame = i

    if current_token is not None and current_token != pad_token_id:
        raw_segments.append({
            "phoneme":    processor.decode([current_token]).strip(),
            "start_time": round(start_frame * frame_duration, 3),
            "end_time":   round(len(predicted_ids) * frame_duration, 3),
        })

    return raw_segments, y_orig, sr_orig


def run_both_variants(
    audio_files: List[str],
    output_root: Path,
    cfg: Dict[str, Any],
    logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Прогоняет все файлы через модель один раз, возвращает
    (baseline_rows, improved_rows) — метрики для обоих вариантов."""

    model_id        = cfg["model"]["model_id"]
    frame_duration  = cfg["model"]["frame_duration"]
    top_db          = cfg["model"]["top_db"]
    audio_dir       = resolve_path(cfg["paths"]["audio_dir"])
    annotations_dir = resolve_path(cfg["paths"]["annotations_dir"])
    valid_ann_ext   = cfg["audio"]["valid_annotation_extensions"]
    tolerance_ms    = tuple(cfg["processing"]["tolerance_ms"])

    baseline_dir = output_root / BASELINE_LABEL
    improved_dir = output_root / IMPROVED_LABEL
    baseline_dir.mkdir(parents=True, exist_ok=True)
    improved_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Модель: {model_id}")
    logger.info(f"  [{BASELINE_LABEL}] {BASELINE_DESC}")
    logger.info(f"  [{IMPROVED_LABEL}] {IMPROVED_DESC}")
    logger.info(f"Файлов: {len(audio_files)}")

    processor, model, pad_token_id = load_model(model_id, logger=logger)

    baseline_rows: List[Dict[str, Any]] = []
    improved_rows: List[Dict[str, Any]] = []

    for filepath in audio_files:
        rel_path  = os.path.relpath(filepath, str(audio_dir))
        base_name = os.path.splitext(rel_path)[0].replace(os.sep, "__")
        logger.info(f"  Обработка: {rel_path}")

        try:
            raw_segs, y_orig, sr_orig = _ctc_decode(
                filepath, processor, model, pad_token_id, frame_duration
            )

            if not raw_segs:
                logger.warning("    Нет сегментов — пропуск")
                continue

            baseline_segs = raw_segs
            improved_segs = correct_segment_boundaries(
                raw_segs, y_orig, sr_orig, frame_duration, top_db
            )

            ref_info = load_reference_annotation(
                filepath, str(audio_dir), str(annotations_dir), valid_ann_ext
            )
            ref_segs = ref_info.get("segments", [])

            source_file = os.path.basename(filepath)
            dialect     = _dialect_from_filename(source_file)

            for label, segs, out_dir in [
                (BASELINE_LABEL, baseline_segs, baseline_dir),
                (IMPROVED_LABEL, improved_segs, improved_dir),
            ]:
                metrics = compute_metrics(segs, ref_segs, tolerance_ms=tolerance_ms)
                meta = {
                    "source_file": source_file,
                    "variant":     label,
                    "model_id":    model_id,
                    "dialect":     dialect,
                    "prediction":  {"segments": segs, "count": len(segs)},
                    "reference":   {"segments": ref_segs, "count": len(ref_segs)},
                    "metrics":     metrics,
                }
                sub = out_dir / base_name
                sub.mkdir(parents=True, exist_ok=True)
                with open(sub / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=4)

            b_metrics = compute_metrics(baseline_segs, ref_segs, tolerance_ms=tolerance_ms)
            i_metrics = compute_metrics(improved_segs, ref_segs, tolerance_ms=tolerance_ms)

            baseline_rows.append({
                "source_file": source_file, "model": BASELINE_LABEL,
                "dialect": dialect, **b_metrics
            })
            improved_rows.append({
                "source_file": source_file, "model": IMPROVED_LABEL,
                "dialect": dialect, **i_metrics
            })

            b_per = b_metrics.get("per")
            i_per = i_metrics.get("per")
            b_per_str = f"{b_per:.4f}" if b_per is not None else "N/A"
            i_per_str = f"{i_per:.4f}" if i_per is not None else "N/A"
            logger.info(
                f"    baseline PER={b_per_str}  "
                f"improved PER={i_per_str}  "
                f"dialect={dialect}"
            )

        except Exception as exc:
            logger.error(f"  Ошибка при обработке {rel_path}: {exc}", exc_info=True)

    for label, rows, out_dir in [
        (BASELINE_LABEL, baseline_rows, baseline_dir),
        (IMPROVED_LABEL, improved_rows, improved_dir),
    ]:
        summary = summarize_metric_rows(rows)
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump({"variant": label, "per_file": rows, "summary": summary},
                      f, ensure_ascii=False, indent=4)
        logger.info(f"\n[{label.upper()}] mean_per={summary.get('mean_per')}  "
                    f"mean_f1={summary.get('mean_phoneme_f1')}")

    return baseline_rows, improved_rows


# ─────────────────────────────────────────────────────────────
# Таблица сравнения
# ─────────────────────────────────────────────────────────────

KEY_METRICS = [
    "per",
    "phoneme_f1",
    "boundary_start_mae",
    "boundary_end_mae",
    "segment_iou_mean",
    "sequence_exact_match",
    "boundary_start_accuracy_20ms",
    "boundary_end_accuracy_20ms",
    "boundary_exact_accuracy_20ms",
    "boundary_exact_accuracy_50ms",
]


def build_comparison_table(
    baseline_rows: List[Dict[str, Any]],
    improved_rows: List[Dict[str, Any]],
    output_dir: Path,
    logger,
) -> pd.DataFrame:
    def aggregate(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        s = summarize_metric_rows(rows)
        row: Dict[str, Any] = {"model": label, "files_processed": s.get("files_processed", 0)}
        for k in KEY_METRICS:
            row[k] = s.get(f"mean_{k}")
        return row

    b_agg = aggregate(baseline_rows, BASELINE_LABEL)
    i_agg = aggregate(improved_rows, IMPROVED_LABEL)

    cmp_df = pd.DataFrame([b_agg, i_agg]).set_index("model")

    delta: Dict[str, Any] = {"model": "delta (improved − baseline)"}
    for col in KEY_METRICS:
        if col not in cmp_df.columns:
            continue
        bv = cmp_df.loc[BASELINE_LABEL, col]
        iv = cmp_df.loc[IMPROVED_LABEL, col]
        try:
            delta[col] = float(iv) - float(bv)
        except (TypeError, ValueError):
            delta[col] = None

    full_df = pd.concat([cmp_df, pd.DataFrame([delta]).set_index("model")])

    csv_path  = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"
    full_df.to_csv(csv_path)
    full_df.reset_index().to_json(json_path, orient="records", force_ascii=False, indent=2)

    logger.info(f"Таблица сохранена: {csv_path}")
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СРАВНЕНИЕ (Baseline vs Improved)")
    print("=" * 70)
    print(full_df.round(4).to_string())
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")

    return full_df


# ─────────────────────────────────────────────────────────────
# График 1: сводное сравнение метрик (2×2)
# ─────────────────────────────────────────────────────────────

def plot_comparison(comparison_df: pd.DataFrame, output_dir: Path, logger) -> None:
    """2×2 бар-чарт: PER, Boundary Exact Accuracy @20ms/@50ms, Segment IoU.

    Метрики выбраны так, чтобы акустическая коррекция границ давала
    видимый эффект: PER не меняется (последовательность фонем одинакова),
    зато точность попадания границ в допуск должна расти.
    """
    models = [m for m in comparison_df.index if "delta" not in m]
    colors = ["#5B8DB8", "#E07B54"]

    # (ключ в df, заголовок, lower_better)
    PLOTS = [
        ("per",                          "PER (↓ лучше)",                        True),
        ("boundary_exact_accuracy_20ms", "Boundary Exact Accuracy @20 ms (↑)",   False),
        ("boundary_exact_accuracy_50ms", "Boundary Exact Accuracy @50 ms (↑)",   False),
        ("segment_iou_mean",             "Segment IoU mean (↑ лучше)",           False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Baseline (CTC без постобработки) vs Improved (+ коррекция границ)",
        fontsize=13, fontweight="bold",
    )
    axes = axes.flatten()

    for ax, (metric, title, lower_better) in zip(axes, PLOTS):
        vals = []
        for m in models:
            v = comparison_df.loc[m, metric] if metric in comparison_df.columns else None
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)

        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
        y_max = max(vals, default=1)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9,
            )

        best_idx = int(np.argmin(vals) if lower_better else np.argmax(vals))
        bars[best_idx].set_edgecolor("#2ECC71")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, y_max * 1.22)
        ax.set_ylabel("Значение метрики")
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = output_dir / "model_comparison_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"График сравнения: {path}")
    print(f"График: {path}")


# ─────────────────────────────────────────────────────────────
# График 2: Boundary Exact Accuracy @20ms и @50ms по диалектам
# ─────────────────────────────────────────────────────────────

def _group_by_dialect(
    rows: List[Dict[str, Any]],
    metric: str,
) -> Dict[str, List[float]]:
    """Группирует значения метрики по диалектам, пропуская None."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        dialect = r.get("dialect", "unknown")
        v = r.get(metric)
        if v is not None:
            try:
                groups[dialect].append(float(v))
            except (TypeError, ValueError):
                pass
    return dict(groups)


def plot_boundary_accuracy_by_dialect(
    baseline_rows: List[Dict[str, Any]],
    improved_rows: List[Dict[str, Any]],
    output_dir: Path,
    logger,
    metric_20: str = "boundary_exact_accuracy_20ms",
    metric_50: str = "boundary_exact_accuracy_50ms",
) -> None:
    """Grouped bar chart: Boundary Exact Accuracy по диалектам, два допуска.

    Для каждого диалекта — 4 бара: baseline@20ms, improved@20ms,
    baseline@50ms, improved@50ms.  Хорошо показывает, в каких диалектах
    коррекция границ помогает больше всего.
    """
    b20 = _group_by_dialect(baseline_rows, metric_20)
    i20 = _group_by_dialect(improved_rows, metric_20)
    b50 = _group_by_dialect(baseline_rows, metric_50)
    i50 = _group_by_dialect(improved_rows, metric_50)

    dialects = sorted(set(b20) | set(i20) | set(b50) | set(i50))
    if not dialects:
        logger.warning("Нет данных по диалектам — график не построен.")
        return

    def mean_or_zero(d: Dict[str, List[float]], key: str) -> float:
        vals = d.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    x = np.arange(len(dialects))
    w = 0.19  # ширина одного бара

    fig, ax = plt.subplots(figsize=(max(10, len(dialects) * 1.6), 6))

    bars_b20 = ax.bar(x - 1.5 * w, [mean_or_zero(b20, d) for d in dialects],
                      w, label="Baseline @20 ms", color="#5B8DB8", edgecolor="white")
    bars_i20 = ax.bar(x - 0.5 * w, [mean_or_zero(i20, d) for d in dialects],
                      w, label="Improved @20 ms", color="#2980B9", edgecolor="white")
    bars_b50 = ax.bar(x + 0.5 * w, [mean_or_zero(b50, d) for d in dialects],
                      w, label="Baseline @50 ms", color="#E07B54", edgecolor="white")
    bars_i50 = ax.bar(x + 1.5 * w, [mean_or_zero(i50, d) for d in dialects],
                      w, label="Improved @50 ms", color="#C0392B", edgecolor="white")

    # Подписи значений над барами
    for bar_group in [bars_b20, bars_i20, bars_b50, bars_i50]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(dialects, fontsize=10)
    ax.set_xlabel("Диалект TIMIT")
    ax.set_ylabel("Boundary Exact Accuracy (доля файлов в допуске)")
    ax.set_title(
        "Boundary Exact Accuracy по диалектам\n"
        "Baseline vs Improved · допуск 20 мс и 50 мс",
        fontsize=12,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Горизонтальные ориентиры
    for lvl, ls in [(0.5, ":"), (0.75, "--")]:
        ax.axhline(lvl, color="grey", linewidth=0.8, linestyle=ls, alpha=0.6)

    plt.tight_layout()
    path = output_dir / "boundary_accuracy_by_dialect.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"График по диалектам: {path}")
    print(f"График: {path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logger = setup_logger("voice_model.compare")
    cfg    = load_config()
    apply_environment(cfg, logger=logger)

    audio_dir  = resolve_path(cfg["paths"]["audio_dir"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    ensure_dirs(audio_dir, output_dir)

    audio_files = list_audio_files(audio_dir, cfg["audio"]["valid_audio_extensions"])
    if not audio_files:
        logger.error(
            f"Аудиофайлы не найдены в {audio_dir}. "
            "Сначала выполните подготовку данных (ноутбук, шаг 1)."
        )
        return

    logger.info(f"Найдено аудиофайлов: {len(audio_files)}")

    baseline_rows, improved_rows = run_both_variants(
        audio_files, output_dir, cfg, logger
    )

    comparison_df = build_comparison_table(baseline_rows, improved_rows, output_dir, logger)
    plot_comparison(comparison_df, output_dir, logger)
    plot_boundary_accuracy_by_dialect(baseline_rows, improved_rows, output_dir, logger)

    logger.info("Сравнение завершено. Артефакты: " + str(output_dir))


if __name__ == "__main__":
    main()
