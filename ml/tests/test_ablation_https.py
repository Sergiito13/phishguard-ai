"""
PhishGuard AI - HTTPS Ablation Unit Tests

Tests:
- Experiment feature configuration
- Exact dataset source matching
- Training and validation filtering
- Group leakage prevention
- Model evaluation metrics

Tests use synthetic data.
No real datasets or model artifacts are modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import ablation_https as ablation

from extract_features import FEATURE_COLUMNS


# ============================================================
# SYNTHETIC DATA
# ============================================================

def make_record(label, sources, group_key):

    record = {
        "label": label,
        "sources": sources,
        "group_key": group_key,
    }

    for feature in FEATURE_COLUMNS:
        record[feature] = 1.0

    return record


# ============================================================
# TEST EXPERIMENT CONFIGURATION
# ============================================================

def test_experiment_configuration():

    baseline = ablation.EXPERIMENTS[
        "Baseline (20 features)"
    ]

    without_https = ablation.EXPERIMENTS[
        "Without HTTPS (19 features)"
    ]

    assert baseline == FEATURE_COLUMNS

    assert len(baseline) == 20

    assert len(without_https) == 19

    assert "is_https" in baseline

    assert "is_https" not in without_https

    assert without_https == [
        feature
        for feature in FEATURE_COLUMNS
        if feature != "is_https"
    ]


# ============================================================
# TEST SOURCE DETECTION
# ============================================================

def test_belongs_to_source():

    df = pd.DataFrame({
        "sources": [
            "url_phish",
            "phiusiil|url_phish",
            "legitphish",
            "phishbd",
            "url_phishing",
        ]
    })

    mask = ablation.belongs_to_source(
        df,
        "url_phish",
    )

    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


# ============================================================
# TEST DATASET FILTERING
# ============================================================

def test_load_data_filters_sources(
    tmp_path,
    monkeypatch,
):

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(1, "phiusiil|url_phish", "train-b"),
        make_record(1, "phishbd", "train-c"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "val-a"),
        make_record(1, "legitphish|url_phish", "val-b"),
        make_record(0, "legitphish", "val-c"),
    ])

    train_file = tmp_path / "train.csv"
    validation_file = tmp_path / "validation.csv"

    train_df.to_csv(train_file, index=False)

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        ablation,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        ablation,
        "VALIDATION_FILE",
        validation_file,
    )

    filtered_train, filtered_validation = (
        ablation.load_data()
    )

    assert len(filtered_train) == 2

    assert len(filtered_validation) == 2

    assert set(filtered_train["group_key"]) == {
        "train-a",
        "train-c",
    }

    assert set(filtered_validation["group_key"]) == {
        "val-a",
        "val-b",
    }


# ============================================================
# TEST GROUP LEAKAGE PREVENTION
# ============================================================

def test_load_data_detects_group_leakage(
    tmp_path,
    monkeypatch,
):

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "shared-group"),
    ])

    validation_df = pd.DataFrame([
        make_record(1, "url_phish", "shared-group"),
    ])

    train_file = tmp_path / "train.csv"
    validation_file = tmp_path / "validation.csv"

    train_df.to_csv(train_file, index=False)

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        ablation,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        ablation,
        "VALIDATION_FILE",
        validation_file,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):
        ablation.load_data()


# ============================================================
# TEST MODEL EVALUATION
# ============================================================

def test_run_experiment_metrics(monkeypatch):

    class FakeModel:

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([0, 1])

        def predict_proba(self, X):
            return np.array([
                [0.9, 0.1],
                [0.2, 0.8],
            ])

    monkeypatch.setattr(
        ablation,
        "create_models",
        lambda: {
            "HistGradientBoosting": FakeModel()
        },
    )

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(1, "phishbd", "train-b"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "val-a"),
        make_record(1, "url_phish", "val-b"),
    ])

    metrics = ablation.run_experiment(
        "Test experiment",
        list(FEATURE_COLUMNS),
        train_df,
        validation_df,
    )

    assert metrics["features"] == 20

    assert metrics["accuracy"] == 1.0

    assert metrics["precision"] == 1.0

    assert metrics["recall"] == 1.0

    assert metrics["f1"] == 1.0

    assert metrics["roc_auc"] == 1.0

    assert metrics["false_positives"] == 0

    assert metrics["false_negatives"] == 0