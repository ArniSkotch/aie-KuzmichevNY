"""Тесты для src.models.metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.metrics import align_sequences, compute_metrics, safe_mean, safe_median, safe_rmse


def test_safe_mean_with_none_values():
    assert safe_mean([1.0, None, 3.0]) == 2.0


def test_safe_mean_empty():
    assert safe_mean([]) is None
    assert safe_mean([None, None]) is None


def test_safe_median():
    assert safe_median([1.0, 2.0, 3.0]) == 2.0


def test_safe_rmse():
    assert safe_rmse([3.0, 4.0]) == 3.5355339059327378


def test_align_sequences_identical():
    alignment, edit_distance = align_sequences(["a", "b", "c"], ["a", "b", "c"])
    assert edit_distance == 0
    assert all(op == "match" for op, _, _ in alignment)


def test_align_sequences_with_substitution():
    alignment, edit_distance = align_sequences(["a", "b", "c"], ["a", "x", "c"])
    assert edit_distance == 1
    ops = [op for op, _, _ in alignment]
    assert ops == ["match", "sub", "match"]


def test_compute_metrics_perfect_match():
    segments = [
        {"phoneme": "h", "start_time": 0.0, "end_time": 0.1},
        {"phoneme": "e", "start_time": 0.1, "end_time": 0.2},
    ]
    metrics = compute_metrics(segments, segments)
    assert metrics["per"] == 0.0
    assert metrics["phoneme_f1"] == 1.0
    assert metrics["sequence_exact_match"] == 1


def test_compute_metrics_with_errors():
    reference = [
        {"phoneme": "h", "start_time": 0.0, "end_time": 0.1},
        {"phoneme": "e", "start_time": 0.1, "end_time": 0.2},
        {"phoneme": "l", "start_time": 0.2, "end_time": 0.3},
    ]
    predicted = [
        {"phoneme": "h", "start_time": 0.0, "end_time": 0.1},
        {"phoneme": "o", "start_time": 0.1, "end_time": 0.2},
    ]
    metrics = compute_metrics(predicted, reference)
    assert metrics["matches"] == 1
    assert metrics["reference_count"] == 3
    assert metrics["predicted_count"] == 2
    assert metrics["sequence_exact_match"] == 0
