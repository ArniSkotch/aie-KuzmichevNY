"""Разведочный анализ данных (EDA) — критерий 3 чек-листа.

Функции модуля:
  compute_dataset_stats()         — агрегирует статистику по всему датасету
  plot_duration_distribution()    — гистограмма длительностей аудио
  plot_phoneme_frequency()        — топ-N самых частых фонем
  plot_dialect_distribution()     — распределение по диалектам
  plot_per_histogram()            — гистограмма PER по файлам
  plot_boundary_errors_boxplot()  — boxplot ошибок границ
  print_worst_files_table()       — топ-N худших файлов по PER
  print_eda_summary()             — текстовый отчёт по датасету

Использование (в ноутбуке или скрипте):
    from src.eda import compute_dataset_stats, plot_duration_distribution, ...
    stats = compute_dataset_stats(audio_dir, annotations_dir)
    plot_duration_distribution(stats)
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.audio_utils import list_audio_files
from src.features.annotation_loader import load_reference_annotation


# ─────────────────────────────────────────────────────────────────────
# Сбор статистики
# ─────────────────────────────────────────────────────────────────────

def compute_dataset_stats(
    audio_dir: Path | str,
    annotations_dir: Path | str,
    valid_audio_ext: Optional[List[str]] = None,
    valid_annotation_ext: Optional[List[str]] = None,
    logger=None,
) -> Dict[str, Any]:
    """Обходит все аудиофайлы и их аннотации, возвращает сводную статистику.

    Возвращает словарь:
      total_files, files_with_annotations,
      durations_sec, mean_duration_sec, median_duration_sec,
      phoneme_counts_per_file, phoneme_freq (Counter),
      unique_phonemes, dialects (Counter), speakers (Counter)
    """
    audio_dir = Path(audio_dir)
    annotations_dir = Path(annotations_dir)

    if valid_audio_ext is None:
        valid_audio_ext = [".wav", ".flac", ".ogg", ".mp3"]
    if valid_annotation_ext is None:
        valid_annotation_ext = [".json", ".csv", ".txt"]

    audio_files = list_audio_files(audio_dir, valid_audio_ext)
    if logger:
        logger.info(f"EDA: найдено аудиофайлов: {len(audio_files)}")

    durations: List[float] = []
    phoneme_counts: List[int] = []
    phoneme_freq: Counter = Counter()
    dialects: Counter = Counter()
    speakers: Counter = Counter()
    files_with_annotations = 0

    for filepath in audio_files:
        # Длительность
        try:
            duration = librosa.get_duration(path=filepath)
            durations.append(duration)
        except Exception:
            try:
                y, sr = librosa.load(filepath, sr=None)
                durations.append(len(y) / sr)
            except Exception:
                durations.append(0.0)

        # Аннотация
        ref = load_reference_annotation(
            filepath, str(audio_dir), str(annotations_dir), valid_annotation_ext
        )
        segments = ref.get("segments", [])
        if segments:
            files_with_annotations += 1
            phoneme_counts.append(len(segments))
            for seg in segments:
                ph = (seg.get("phoneme") or seg.get("label") or "").strip().lower()
                if ph:
                    phoneme_freq[ph] += 1

        # Диалект и спикер из пути или имени файла
        name = Path(filepath).stem  # например: TRAIN_DR1_FCJF0_SA1
        parts = name.split("_")
        if len(parts) >= 3:
            dialects[parts[1]] += 1
            speakers[parts[2]] += 1
        else:
            dialects["unknown"] += 1

    stats: Dict[str, Any] = {
        "total_files": len(audio_files),
        "files_with_annotations": files_with_annotations,
        "durations_sec": durations,
        "mean_duration_sec": float(np.mean(durations)) if durations else 0.0,
        "median_duration_sec": float(np.median(durations)) if durations else 0.0,
        "min_duration_sec": float(np.min(durations)) if durations else 0.0,
        "max_duration_sec": float(np.max(durations)) if durations else 0.0,
        "phoneme_counts_per_file": phoneme_counts,
        "mean_phonemes_per_file": float(np.mean(phoneme_counts)) if phoneme_counts else 0.0,
        "phoneme_freq": phoneme_freq,
        "unique_phonemes": len(phoneme_freq),
        "total_phoneme_instances": sum(phoneme_freq.values()),
        "dialects": dialects,
        "speakers": speakers,
    }

    if logger:
        logger.info(
            f"EDA: средняя длительность={stats['mean_duration_sec']:.2f}с, "
            f"уникальных фонем={stats['unique_phonemes']}, "
            f"диалектов={len(dialects)}"
        )

    return stats


# ─────────────────────────────────────────────────────────────────────
# Графики EDA
# ─────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: Optional[Path | str]) -> None:
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_duration_distribution(
    stats: Dict[str, Any],
    save_path: Optional[Path | str] = None,
) -> None:
    """Гистограмма длительностей аудиофайлов (в секундах)."""
    durations = stats.get("durations_sec", [])
    if not durations:
        print("Нет данных о длительностях.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(durations, bins=20, color="#5B8DB8", edgecolor="white", linewidth=0.6)
    ax.axvline(stats["mean_duration_sec"], color="#E07B54", linestyle="--",
               linewidth=1.5, label=f"Среднее: {stats['mean_duration_sec']:.2f} с")
    ax.axvline(stats["median_duration_sec"], color="#4CAF50", linestyle=":",
               linewidth=1.5, label=f"Медиана: {stats['median_duration_sec']:.2f} с")
    ax.set_xlabel("Длительность (с)")
    ax.set_ylabel("Количество файлов")
    ax.set_title("Распределение длительностей аудиофайлов")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_phoneme_frequency(
    stats: Dict[str, Any],
    top_n: int = 20,
    save_path: Optional[Path | str] = None,
) -> None:
    """Горизонтальный бар-чарт топ-N самых частых фонем."""
    freq: Counter = stats.get("phoneme_freq", Counter())
    if not freq:
        print("Нет данных о фонемах.")
        return

    top = freq.most_common(top_n)
    phonemes = [p for p, _ in reversed(top)]
    counts = [c for _, c in reversed(top)]
    total = sum(freq.values())

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bars = ax.barh(phonemes, counts, color="#5B8DB8", edgecolor="white", linewidth=0.4)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({cnt/total*100:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("Количество вхождений")
    ax.set_title(f"Топ-{top_n} самых частых фонем (всего уникальных: {len(freq)})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_dialect_distribution(
    stats: Dict[str, Any],
    save_path: Optional[Path | str] = None,
) -> None:
    """Два subplot: распределение по диалектам и количество фонем на файл."""
    dialects: Counter = stats.get("dialects", Counter())
    phoneme_counts: List[int] = stats.get("phoneme_counts_per_file", [])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Левый: диалекты
    if dialects:
        labels = list(dialects.keys())
        values = list(dialects.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))  # type: ignore
        axes[0].bar(labels, values, color=colors, edgecolor="white")
        axes[0].set_xlabel("Диалект")
        axes[0].set_ylabel("Количество файлов")
        axes[0].set_title("Распределение файлов по диалектам")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].spines[["top", "right"]].set_visible(False)
    else:
        axes[0].text(0.5, 0.5, "Нет данных о диалектах", ha="center", va="center",
                     transform=axes[0].transAxes)

    # Правый: фонем на файл
    if phoneme_counts:
        axes[1].hist(phoneme_counts, bins=15, color="#9C7BB5", edgecolor="white", linewidth=0.5)
        axes[1].axvline(np.mean(phoneme_counts), color="#E07B54", linestyle="--",
                        linewidth=1.5, label=f"Среднее: {np.mean(phoneme_counts):.1f}")
        axes[1].set_xlabel("Количество фонем на файл")
        axes[1].set_ylabel("Количество файлов")
        axes[1].set_title("Распределение количества фонем на файл")
        axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)
    else:
        axes[1].text(0.5, 0.5, "Нет данных об аннотациях", ha="center", va="center",
                     transform=axes[1].transAxes)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────
# Графики метрик (по результатам run_pipeline)
# ─────────────────────────────────────────────────────────────────────

def plot_per_histogram(
    per_file_rows: List[Dict[str, Any]],
    save_path: Optional[Path | str] = None,
) -> None:
    """Гистограмма PER по всем обработанным файлам."""
    per_values = [r.get("per") for r in per_file_rows if r.get("per") is not None]
    if not per_values:
        print("Нет данных PER для построения гистограммы.")
        return

    per_arr = [float(v) for v in per_values]
    mean_per = np.mean(per_arr)
    median_per = np.median(per_arr)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(per_arr, bins=max(8, len(per_arr) // 2), color="#5B8DB8",
            edgecolor="white", linewidth=0.6)
    ax.axvline(mean_per, color="#E07B54", linestyle="--", linewidth=1.5,
               label=f"Среднее PER: {mean_per:.3f}")
    ax.axvline(median_per, color="#4CAF50", linestyle=":", linewidth=1.5,
               label=f"Медиана PER: {median_per:.3f}")
    ax.set_xlabel("PER (Phoneme Error Rate)")
    ax.set_ylabel("Количество файлов")
    ax.set_title(f"Распределение PER по файлам датасета (n={len(per_arr)})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_boundary_errors_boxplot(
    per_file_rows: List[Dict[str, Any]],
    save_path: Optional[Path | str] = None,
) -> None:
    """Boxplot ошибок границ фонем (start MAE, end MAE, center MAE)."""
    keys = {
        "Start MAE": "boundary_start_mae",
        "End MAE": "boundary_end_mae",
        "Center MAE": "boundary_center_mae",
    }

    data = []
    labels = []
    for label, key in keys.items():
        vals = [float(r[key]) for r in per_file_rows if r.get(key) is not None]
        if vals:
            data.append(vals)
            labels.append(label)

    if not data:
        print("Нет данных об ошибках границ. Убедитесь, что в датасете есть аннотации с таймингами.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})
    colors = ["#5B8DB8", "#E07B54", "#9C7BB5"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Ошибка границы (с)")
    ax.set_title("Ошибки границ фонем: MAE по всем файлам")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def print_worst_files_table(
    per_file_rows: List[Dict[str, Any]],
    top_n: int = 5,
) -> None:
    """Выводит таблицу топ-N худших файлов по PER."""
    rows_with_per = [r for r in per_file_rows if r.get("per") is not None]
    if not rows_with_per:
        print("Нет данных PER.")
        return

    sorted_rows = sorted(rows_with_per, key=lambda r: float(r["per"]), reverse=True)
    top = sorted_rows[:top_n]

    header = f"{'Файл':<40} {'PER':>8} {'F1':>8} {'Фонем (pred)':>14} {'Фонем (ref)':>13}"
    print(f"\nТоп-{top_n} худших файлов по PER:")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for r in top:
        fname = str(r.get("source_file", r.get("file_id", "?")))[:38]
        per = r.get("per")
        f1 = r.get("phoneme_f1")
        pred_c = r.get("predicted_count", "?")
        ref_c = r.get("reference_count", "?")
        per_s = f"{float(per):.4f}" if per is not None else "N/A"
        f1_s = f"{float(f1):.4f}" if f1 is not None else "N/A"
        print(f"{fname:<40} {per_s:>8} {f1_s:>8} {str(pred_c):>14} {str(ref_c):>13}")
    print("─" * len(header))


# ─────────────────────────────────────────────────────────────────────
# Текстовый отчёт
# ─────────────────────────────────────────────────────────────────────

def print_eda_summary(stats: Dict[str, Any]) -> None:
    """Печатает краткие текстовые выводы по EDA."""
    freq: Counter = stats.get("phoneme_freq", Counter())
    durations = stats.get("durations_sec", [])
    dialects: Counter = stats.get("dialects", Counter())

    top3 = [p for p, _ in freq.most_common(3)]
    rare3 = [p for p, _ in freq.most_common()[:-4:-1]] if len(freq) >= 3 else []
    most_dialect = dialects.most_common(1)[0][0] if dialects else "—"

    short_files = sum(1 for d in durations if d < 4.0)
    pct_short = short_files / len(durations) * 100 if durations else 0

    print("=" * 60)
    print("ВЫВОДЫ ПО EDA ДАТАСЕТА TIMIT")
    print("=" * 60)
    print(f"\n1. Размер датасета: {stats['total_files']} аудиофайлов,")
    print(f"   из них {stats['files_with_annotations']} имеют аннотации фонем.")
    print(
        f"\n2. Длительность записей: большинство файлов ({pct_short:.0f}%) короче 4 с. "
        f"Средняя длительность — {stats['mean_duration_sec']:.2f} с, "
        f"медиана — {stats['median_duration_sec']:.2f} с."
    )
    print(
        f"\n3. Фонемный состав: всего {stats['unique_phonemes']} уникальных фонем "
        f"({stats['total_phoneme_instances']} вхождений). "
        f"Самые частые: {', '.join(top3)}. "
        + (f"Самые редкие: {', '.join(rare3)}." if rare3 else "")
    )
    if dialects:
        print(
            f"\n4. Диалекты: {len(dialects)} групп. "
            f"Наибольшее представление — '{most_dialect}' "
            f"({dialects[most_dialect]} файлов)."
        )
    ph_counts = stats.get("phoneme_counts_per_file", [])
    if ph_counts:
        print(
            f"\n5. Количество фонем на файл: в среднем {np.mean(ph_counts):.1f} "
            f"(мин. {np.min(ph_counts)}, макс. {np.max(ph_counts)})."
        )
    print("=" * 60)
