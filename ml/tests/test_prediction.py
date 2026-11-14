"""
PhishGuard AI - Prediction Unit Tests

Tests the experimental URL prediction pipeline.

No real model training or network access is required.
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

import predict_url as predictor

from extract_features import FEATURE_COLUMNS


# ============================================================
# MOCK MODEL
# ============================================================

class DummyModel:
    """Simulates a trained ML classifier."""

    def __init__(self, phishing_score):
        self.phishing_score = phishing_score

    def predict_proba(self, X):

        assert X.shape == (
            1,
            len(FEATURE_COLUMNS),
        )

        return np.array([
            [
                1 - self.phishing_score,
                self.phishing_score,
            ]
        ])


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model_package():

    return {
        "model": DummyModel(0.10),
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": 0.50,
        "calibrated": False,
        "model_version": "0.1.0-experimental",
    }


# ============================================================
# LEGITIMATE CLASSIFICATION
# ============================================================

def test_legitimate_prediction(model_package):

    result = predictor.predict_url(
        "https://www.python.org",
        model_package,
    )

    assert result["prediction"] == "not_flagged"

    assert result["phishing_score"] == 0.10

    assert result["calibrated"] is False

    assert result["features_analyzed"] == 20


# ============================================================
# PHISHING CLASSIFICATION
# ============================================================

def test_phishing_prediction(model_package):

    model_package["model"] = DummyModel(0.95)

    result = predictor.predict_url(
        "https://example.com/login",
        model_package,
    )

    assert result["prediction"] == "potential_phishing"

    assert result["phishing_score"] == 0.95


# ============================================================
# INVALID URL INPUTS
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "",
        "https:///login",
        "ftp://example.com",
    ],
)
def test_invalid_url(model_package, url):

    with pytest.raises(ValueError):
        predictor.predict_url(
            url,
            model_package,
        )


# ============================================================
# MISSING MODEL
# ============================================================

def test_missing_model(monkeypatch, tmp_path):

    missing_file = tmp_path / "missing.joblib"

    monkeypatch.setattr(
        predictor,
        "MODEL_FILE",
        missing_file,
    )

    with pytest.raises(FileNotFoundError):
        predictor.load_model()


# ============================================================
# FEATURE VERSION MISMATCH
# ============================================================

def test_feature_mismatch(
    monkeypatch,
    tmp_path,
    model_package,
):

    model_file = tmp_path / "model.joblib"

    model_file.touch()

    model_package["feature_columns"] = [
        "unknown_feature",
    ]

    monkeypatch.setattr(
        predictor,
        "MODEL_FILE",
        model_file,
    )

    monkeypatch.setattr(
        predictor.joblib,
        "load",
        lambda _: model_package,
    )

    with pytest.raises(
        ValueError,
        match="Feature mismatch",
    ):
        predictor.load_model()