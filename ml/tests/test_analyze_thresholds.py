"""
PhishGuard AI - Classification Threshold Analysis Tests

Tests:
- Threshold evaluation metrics
- Confusion matrix calculations
- False positive rate
- Threshold boundary behavior
- Threshold monotonicity
- No-positive-prediction handling

All tests use synthetic data.
No models are trained or saved.
"""

import sys
from pathlib import Path

import numpy as np
import pytest


# ============================================================
# IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import analyze_thresholds as thresholds


# ============================================================
# 1. THRESHOLD METRICS
# ============================================================

def test_threshold_metrics(monkeypatch):

    monkeypatch.setattr(
        thresholds,
        "THRESHOLDS",
        [0.50, 0.90],
    )

    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.85,
        0.65,
        0.95,
    ])

    results = thresholds.evaluate_thresholds(
        y_true,
        probabilities,
    )

    assert len(results) == 2

    low = results.iloc[0]
    high = results.iloc[1]

    # Threshold 0.50
    assert low["true_positives"] == 2
    assert low["true_negatives"] == 1
    assert low["false_positives"] == 1
    assert low["false_negatives"] == 0

    assert low["precision"] == pytest.approx(2 / 3)
    assert low["recall"] == pytest.approx(1.0)
    assert low["f1"] == pytest.approx(0.8)
    assert low["fpr"] == pytest.approx(0.5)

    # Threshold 0.90
    assert high["true_positives"] == 1
    assert high["true_negatives"] == 2
    assert high["false_positives"] == 0
    assert high["false_negatives"] == 1

    assert high["precision"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(0.5)
    assert high["fpr"] == pytest.approx(0.0)


# ============================================================
# 2. THRESHOLD BOUNDARY
# ============================================================

def test_threshold_boundary(monkeypatch):

    monkeypatch.setattr(
        thresholds,
        "THRESHOLDS",
        [0.50],
    )

    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.50,
        0.49,
        0.50,
        0.91,
    ])

    results = thresholds.evaluate_thresholds(
        y_true,
        probabilities,
    )

    result = results.iloc[0]

    # Scores equal to the threshold are classified
    # as positive.

    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 0


# ============================================================
# 3. FALSE POSITIVE RATE
# ============================================================

def test_false_positive_rate(monkeypatch):

    monkeypatch.setattr(
        thresholds,
        "THRESHOLDS",
        [0.50],
    )

    y_true = np.array([0, 0, 0, 1])

    probabilities = np.array([
        0.90,
        0.80,
        0.10,
        0.95,
    ])

    results = thresholds.evaluate_thresholds(
        y_true,
        probabilities,
    )

    result = results.iloc[0]

    assert result["false_positives"] == 2

    assert result["true_negatives"] == 1

    assert result["fpr"] == pytest.approx(2 / 3)


# ============================================================
# 4. THRESHOLD MONOTONICITY
# ============================================================

def test_threshold_monotonicity(monkeypatch):

    monkeypatch.setattr(
        thresholds,
        "THRESHOLDS",
        [0.20, 0.40, 0.60, 0.90],
    )

    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.60,
        0.50,
        0.90,
    ])

    results = thresholds.evaluate_thresholds(
        y_true,
        probabilities,
    )

    # Increasing the threshold cannot increase
    # the number of predicted positives.

    assert results["false_positives"].is_monotonic_decreasing

    assert results["true_positives"].is_monotonic_decreasing

    # False negatives cannot decrease.

    assert results["false_negatives"].is_monotonic_increasing


# ============================================================
# 5. NO POSITIVE PREDICTIONS
# ============================================================

def test_no_positive_predictions(monkeypatch):

    monkeypatch.setattr(
        thresholds,
        "THRESHOLDS",
        [0.99],
    )

    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.20,
        0.70,
        0.80,
    ])

    results = thresholds.evaluate_thresholds(
        y_true,
        probabilities,
    )

    result = results.iloc[0]

    assert result["true_positives"] == 0

    assert result["false_positives"] == 0

    assert result["false_negatives"] == 2

    assert result["precision"] == 0.0

    assert result["recall"] == 0.0

    assert result["f1"] == 0.0