"""FastAPI-сервис фонемного выравнивания речи.

Эндпоинты:
- GET  /health  - проверка работоспособности сервиса
- POST /predict - принимает .wav/.flac/.ogg/.mp3-файл, возвращает список фонем с таймингами

Запуск как обычный процесс:
    python -m src.service.api

Запуск в фоне внутри ноутбука (без блокировки остальных ячеек):
    from src.service.api import run_in_background
    run_in_background()
"""
from __future__ import annotations

import tempfile

import librosa
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import load_config, resolve_path
from src.env_setup import apply_environment
from src.logging_utils import setup_logger
from src.models.inference import correct_segment_boundaries
from src.models.model_loader import load_model

logger = setup_logger("voice_model.service")

cfg = load_config()
apply_environment(cfg, logger=logger)
MODEL_ID = cfg["model"]["model_id"]
FRAME_DURATION = cfg["model"]["frame_duration"]
TOP_DB = cfg["model"]["top_db"]
SUPPORTED_EXTENSIONS = tuple(cfg["audio"]["valid_audio_extensions"])

app = FastAPI(
    title="Voice Phoneme Alignment API",
    description=(
        "Сервис фонемного выравнивания речи на базе facebook/wav2vec2-lv-60-espeak-cv-ft.\n\n"
        "Принимает аудиофайл (.wav), возвращает список фонем с временными метками."
    ),
    version="1.0.0",
)

# Модель загружается один раз при импорте модуля.
processor, model, pad_token_id = load_model(MODEL_ID, logger=logger)


@app.get("/health", summary="Проверка работоспособности сервиса")
def health_check():
    """Возвращает статус сервиса и факт загрузки модели."""
    logger.info("GET /health -> ok")
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
    }


@app.post("/predict", summary="Фонемное выравнивание аудиофайла")
async def predict(file: UploadFile = File(..., description="Аудиофайл в формате WAV")):
    """Принимает аудиофайл и возвращает список фонем с временными метками.

    Формат ответа:
        {
          "source_file": "example.wav",
          "phoneme_count": 42,
          "segments": [{"phoneme": "h", "start_time": 0.0, "end_time": 0.08}, ...]
        }
    """
    if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        logger.warning(f"POST /predict - неподдерживаемый формат: {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат файла. Ожидается один из: {SUPPORTED_EXTENSIONS}",
        )

    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"POST /predict - обработка файла: {file.filename} ({len(contents)} байт)")

        segments = _predict_segments(tmp_path)

        logger.info(f"POST /predict - {file.filename}: извлечено {len(segments)} фонем")

        return JSONResponse(content={
            "source_file": file.filename,
            "phoneme_count": len(segments),
            "segments": segments,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"POST /predict - ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")
    finally:
        if tmp_path:
            try:
                import os as _os

                _os.unlink(tmp_path)
            except Exception:
                pass


def _predict_segments(tmp_path: str):
    """Прогоняет файл через модель и возвращает скорректированные сегменты фонем."""
    y_orig, sr_orig = librosa.load(tmp_path, sr=None)
    y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=16000)

    inputs = processor(y_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()

    raw_segments = []
    current_token = None
    start_frame = 0

    for i, token_id in enumerate(predicted_ids):
        if token_id != current_token:
            if current_token is not None and current_token != pad_token_id:
                phoneme_symbol = processor.decode([current_token]).strip()
                raw_segments.append({
                    "phoneme": phoneme_symbol,
                    "start_time": start_frame * FRAME_DURATION,
                    "end_time": i * FRAME_DURATION,
                })
            current_token = token_id
            start_frame = i

    if current_token is not None and current_token != pad_token_id:
        phoneme_symbol = processor.decode([current_token]).strip()
        raw_segments.append({
            "phoneme": phoneme_symbol,
            "start_time": start_frame * FRAME_DURATION,
            "end_time": len(predicted_ids) * FRAME_DURATION,
        })

    if not raw_segments:
        return []

    return correct_segment_boundaries(raw_segments, y_orig, sr_orig, FRAME_DURATION, TOP_DB)


def run_in_background(host: str | None = None, port: int | None = None) -> None:
    """Запускает сервис в фоновом потоке - удобно вызывать из ноутбука."""
    import threading

    import uvicorn

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        import subprocess

        subprocess.run(["pip", "install", "nest_asyncio", "-q"], check=True)
        import nest_asyncio

        nest_asyncio.apply()

    service_host = host or cfg["service"]["host"]
    service_port = port or cfg["service"]["port"]

    def _run_server():
        uvicorn.run(app, host=service_host, port=service_port, log_level="warning")

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    logger.info(f"Сервис запущен: http://localhost:{service_port}")
    logger.info(f"Swagger UI:     http://localhost:{service_port}/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg["service"]["host"], port=cfg["service"]["port"])
