"""Точка входа пайплайна фонемного выравнивания.

Запуск:
    python -m src.run_pipeline

Шаги:
1. Настраивает окружение и логирование.
2. Загружает модель Wav2Vec2.
3. Прогоняет все аудиофайлы из configs/config.yaml -> paths.audio_dir.
4. Сохраняет metadata.json + нарезки по фонемам в output_dir.
5. Считает метрики (PER, F1, ошибки границ, IoU) и сохраняет сводку.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.audio_utils import ensure_dirs, list_audio_files
from src.env_setup import apply_environment
from src.logging_utils import setup_logger
from src.models.inference import extract_and_save_phonemes
from src.models.metrics import compute_metrics, summarize_metric_rows
from src.models.model_loader import load_model


def main() -> None:
    logger = setup_logger("voice_model.pipeline")
    cfg = load_config()
    apply_environment(cfg, logger=logger)

    audio_dir = resolve_path(cfg["paths"]["audio_dir"])
    annotations_dir = resolve_path(cfg["paths"]["annotations_dir"])
    output_dir = resolve_path(cfg["paths"]["output_dir"])
    ensure_dirs(audio_dir, annotations_dir, output_dir)

    valid_audio_ext = cfg["audio"]["valid_audio_extensions"]
    valid_annotation_ext = cfg["audio"]["valid_annotation_extensions"]
    frame_duration = cfg["model"]["frame_duration"]
    top_db = cfg["model"]["top_db"]
    tolerance_ms = tuple(cfg["processing"]["tolerance_ms"])

    audio_files = list_audio_files(audio_dir, valid_audio_ext)
    if not audio_files:
        logger.warning(f"Файлы с расширениями {valid_audio_ext} не найдены в папке {audio_dir}.")
        return

    processor, model, pad_token_id = load_model(cfg["model"]["model_id"], logger=logger)

    logger.info(f"Найдено аудиофайлов: {len(audio_files)}. Начинаем обработку...")
    all_metrics = []

    for filepath in audio_files:
        rel_path = os.path.relpath(filepath, audio_dir)
        base_name = os.path.splitext(rel_path)[0].replace(os.sep, "__")
        output_subfolder = os.path.join(output_dir, base_name)

        logger.info(f"Обработка {rel_path}...")
        try:
            metadata = extract_and_save_phonemes(
                filepath,
                output_subfolder,
                model,
                processor,
                pad_token_id,
                audio_dir=str(audio_dir),
                annotations_dir=str(annotations_dir),
                valid_annotation_extensions=valid_annotation_ext,
                frame_duration=frame_duration,
                top_db=top_db,
            )

            predicted_segments = metadata["prediction"]["segments"]
            reference_segments = metadata["reference"]["segments"]
            metrics = compute_metrics(predicted_segments, reference_segments, tolerance_ms=tolerance_ms)
            metadata["metrics"] = metrics

            with open(os.path.join(output_subfolder, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            summary_row = {"source_file": metadata["source_file"], **metrics}
            all_metrics.append(summary_row)

            print(
                f"  -> Извлечено фонем: {metadata['prediction']['count']}. "
                f"PER: {metrics.get('per')}, MBE: {metrics.get('minimal_boundary_error_mean')}"
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {rel_path}: {e}")

    metrics_summary = summarize_metric_rows(all_metrics)
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_file": all_metrics, "summary": metrics_summary}, f, ensure_ascii=False, indent=4)

    logger.info(f"Готово! Проверьте папку {output_dir}.")
    logger.info("Сводка метрик сохранена в metrics_summary.json")

    if metrics_summary:
        print("\nИтоговые метрики по набору:")
        for key, value in metrics_summary.items():
            print(f"  {key}: {value}")

    build_metrics_table(output_dir, logger=logger)


def build_metrics_table(output_dir: Path, logger=None) -> None:
    """Собирает таблицу метрик по всем metadata.json в output_dir
    (включая строку AVERAGE) и сохраняет её как .csv и .json."""
    output_dir = Path(output_dir)
    metrics_results = []

    for metadata_file in sorted(output_dir.rglob("metadata.json")):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Не удалось прочитать {metadata_file}: {e}")
            continue

        file_id = meta.get("source_file", metadata_file.parent.name)
        file_metrics = meta.get("metrics")

        if file_metrics is None:
            predicted_segments = meta.get("prediction", {}).get("segments", [])
            reference_segments = meta.get("reference", {}).get("segments", [])
            file_metrics = compute_metrics(predicted_segments, reference_segments)

        metrics_results.append({"file_id": file_id, **file_metrics})

    if not metrics_results:
        print("Нет данных для таблицы. Сначала запустите обработку аудиофайлов.")
        return

    metrics_df = pd.DataFrame(metrics_results)
    numeric_cols = metrics_df.select_dtypes(include="number").columns.tolist()

    average_row = {"file_id": "AVERAGE"}
    for col in numeric_cols:
        average_row[col] = metrics_df[col].mean()

    metrics_df = pd.concat([metrics_df, pd.DataFrame([average_row])], ignore_index=True)

    csv_path = output_dir / "metrics_comparison.csv"
    json_path = output_dir / "metrics_summary.json"

    metrics_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(metrics_df.to_string())
    print()
    print(f"CSV сохранён:  {csv_path}")
    print(f"JSON сохранён: {json_path}")


if __name__ == "__main__":
    main()
