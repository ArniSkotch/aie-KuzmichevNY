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
