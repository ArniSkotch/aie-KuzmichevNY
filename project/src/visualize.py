"""Визуализация аудиосигналов: осциллограмма, спектр (БПФ), мел-спектрограмма."""
from __future__ import annotations

import os
import random
from typing import Iterable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(audio_files: Iterable[str], audio_dir: str, num_samples: int = 3) -> None:
    """Строит три графика (волна/спектр/мел-спектрограмма) для случайных файлов."""
    audio_files = list(audio_files)
    if not audio_files:
        print(f"Файлы для визуализации не найдены в папке {audio_dir}.")
        return

    num_samples = min(num_samples, len(audio_files))
    sample_files = random.sample(audio_files, num_samples)

    for filepath in sample_files:
        filename = os.path.relpath(filepath, audio_dir)

        y, sr = librosa.load(filepath, sr=None)

        plt.figure(figsize=(14, 12))
        plt.suptitle(f"Анализ аудиосигнала: {filename}", fontsize=16)

        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr, color="#1f77b4")
        plt.title("Осциллограмма (Амплитуда во времени)")
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")

        plt.subplot(3, 1, 2)
        n = len(y)
        yf = np.abs(np.fft.rfft(y))
        xf = np.fft.rfftfreq(n, 1 / sr)

        plt.plot(xf, yf, color="#2ca02c")
        plt.title("Спектр (Быстрое преобразование Фурье - БПФ)")
        plt.xlabel("Частота (Гц)")
        plt.ylabel("Магнитуда")
        plt.xlim(0, min(8000, sr / 2))

        plt.subplot(3, 1, 3)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=sr / 2, cmap="magma")
        plt.colorbar(img, format="%+2.0f dB")
        plt.title("Мел-спектрограмма")
        plt.xlabel("Время (с)")
        plt.ylabel("Частота (Гц)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
