"""Тесты для src.data.audio_utils."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.audio_utils import ensure_dirs, list_audio_files


def test_list_audio_files_finds_valid_extensions(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"fake")
    (audio_dir / "b.mp3").write_bytes(b"fake")
    (audio_dir / "c.txt").write_bytes(b"not audio")

    sub_dir = audio_dir / "nested"
    sub_dir.mkdir()
    (sub_dir / "d.flac").write_bytes(b"fake")

    found = list_audio_files(audio_dir, valid_extensions=[".wav", ".mp3", ".flac"])

    assert len(found) == 3
    assert all(f.endswith((".wav", ".mp3", ".flac")) for f in found)


def test_list_audio_files_missing_dir_returns_empty(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    found = list_audio_files(missing_dir, valid_extensions=[".wav"])
    assert found == []


def test_ensure_dirs_creates_missing_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    ensure_dirs(target)
    assert target.exists()
