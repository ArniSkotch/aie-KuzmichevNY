"""Тесты для src.features.annotation_loader."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.annotation_loader import (
    normalize_label,
    parse_segments_from_rows,
    segment_has_timing,
    segment_to_label,
)


def test_normalize_label_lowercases_and_trims():
    assert normalize_label("  Hello   World  ") == "hello world"


def test_normalize_label_none_returns_empty_string():
    assert normalize_label(None) == ""


def test_segment_to_label_from_dict():
    assert segment_to_label({"phoneme": "AA"}) == "aa"
    assert segment_to_label({"label": "BB"}) == "bb"


def test_segment_to_label_from_plain_value():
    assert segment_to_label("CC") == "cc"


def test_segment_has_timing():
    assert segment_has_timing({"start_time": 0.0, "end_time": 1.0}) is True
    assert segment_has_timing({"start_time": 0.0}) is False
    assert segment_has_timing("not_a_dict") is False


def test_parse_segments_from_rows_with_timing():
    rows = [{"phoneme": "h", "start_time": "0.0", "end_time": "0.1"}]
    segments = parse_segments_from_rows(rows)
    assert segments == [{"phoneme": "h", "start_time": 0.0, "end_time": 0.1}]


def test_parse_segments_from_rows_without_timing():
    rows = ["h", "e", "l", "l", "o"]
    segments = parse_segments_from_rows(rows)
    assert [s["phoneme"] for s in segments] == ["h", "e", "l", "l", "o"]
    assert all("start_time" not in s for s in segments)
