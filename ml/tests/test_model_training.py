"""
PhishGuard AI - Model Training Unit Tests

Tests:
- Model initialization
- Training and prediction
- Probability outputs
- Dataset loading
- Invalid labels
- Missing datasets
- Invalid feature values
- Evaluation metrics
- Dataset source identification
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

from train_model import load_dataset
from compare_models import create_models, calculate_metrics
from evaluate_cross_source import belongs_to_source
from extract_features import FEATURE_COLUMNS


# ============================================================
# SYNTHETIC DATASET
# ============================================================

@pytest.fixture
def sample_dataset():

    records = []

    for i in range(80):

        record = {
            "label": int(i >= 40),
        }

        for feature in FEATURE_COLUMNS:
            record[feature] = float(i)

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# TEST MODEL CREATION
# ============================================================

def test_models_are_created():

    models = create_models()

    assert "Logistic Regression" in models
    assert "HistGradientBoosting" in models

    assert len(models) == 2


# ============================================================
# TEST MODEL TRAINING
# ============================================================

@pytest.mark.parametrize(
    "model_name",
    [
        "Logistic Regression",
        "HistGradientBoosting",
    ],
)
def test_model_training_and_prediction(
    sample_dataset,
    model_name,
):

    X = sample_dataset[FEATURE_COLUMNS]

    y = sample_dataset["label"]

    model = create_models()[model_name]

    model.fit(X, y)

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    assert len(predictions) == len(X)

    assert probabilities.shape == (len(X), 2)

    assert np.isfinite(probabilities).all()

    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


# ============================================================
# TEST DATASET LOADING
# ============================================================

def test_load_dataset(
    sample_dataset,
    tmp_path,
):

    file_path = tmp_path / "dataset.csv"

    sample_dataset.to_csv(
        file_path,
        index=False,
    )

    X, y = load_dataset(file_path)

    assert X.shape == (
        len(sample_dataset),
        len(FEATURE_COLUMNS),
    )

    assert len(y) == len(sample_dataset)

    assert set(np.unique(y)) == {0, 1}


# ============================================================
# TEST INVALID LABELS
# ============================================================

def test_invalid_labels(
    sample_dataset,
    tmp_path,
):

    sample_dataset.loc[0, "label"] = 2

    file_path = tmp_path / "invalid_labels.csv"

    sample_dataset.to_csv(
        file_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Invalid labels",
    ):
        load_dataset(file_path)


# ============================================================
# TEST MISSING DATASET
# ============================================================

def test_missing_dataset(tmp_path):

    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(file_path)


# ============================================================
# TEST INVALID FEATURE VALUES
# ============================================================

def test_invalid_features(
    sample_dataset,
    tmp_path,
):

    sample_dataset.loc[
        0,
        "url_length",
    ] = np.inf

    file_path = tmp_path / "invalid_features.csv"

    sample_dataset.to_csv(
        file_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="invalid numerical values",
    ):
        load_dataset(file_path)


# ============================================================
# TEST EVALUATION METRICS
# ============================================================

def test_calculate_metrics():

    y_true = np.array([0, 0, 1, 1])

    predictions = np.array([0, 1, 0, 1])

    probabilities = np.array([
        0.1,
        0.7,
        0.4,
        0.8,
    ])

    metrics = calculate_metrics(
        y_true,
        predictions,
        probabilities,
    )

    assert metrics["accuracy"] == 0.5

    assert metrics["precision"] == 0.5

    assert metrics["recall"] == 0.5

    assert metrics["f1"] == 0.5

    assert metrics["roc_auc"] == 0.75

    assert metrics["false_positives"] == 1

    assert metrics["false_negatives"] == 1


# ============================================================
# TEST DATASET SOURCE DETECTION
# ============================================================

def test_belongs_to_source():

    df = pd.DataFrame(
        {
            "sources": [
                "legitphish",
                "phiusiil|url_phish",
                "url_phish",
                "phishbd",
            ],
        }
    )

    mask = belongs_to_source(
        df,
        "url_phish",
    )

    assert mask.tolist() == [
        False,
        True,
        True,
        False,
    ]