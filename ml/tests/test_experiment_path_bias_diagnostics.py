"""Synthetic, offline tests for path-bias diagnostics (no real datasets)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_path_bias_diagnostics as diag


@pytest.fixture
def frame():
    rows = []
    for si, source in enumerate(diag.SOURCES):
        for label in (0, 1):
            for i in range(6):
                row = {
                    "url": f"https://{source}-{label}-{i}.example/path",
                    "sources": source,
                    "group_key": f"domain:{source}-{label}-{i}.example",
                    "label": label,
                    "is_https": i % 2,
                    **{f: float(si + label + i / 10) for f in diag.FEATURES},
                }
                row["v2_path_length"] = float(0 if i == 0 else 5)
                row["v2_path_depth"] = float(0 if i == 0 else (3 if i > 3 else 1))
                rows.append(row)
    return pd.DataFrame(rows)


class FakeModel:
    classes_ = np.array([0, 1])
    fit_calls = []

    def fit(self, X, y, sample_weight=None):
        assert X.shape[1] == 31
        assert len(y) == len(X)
        assert set(y) == {0, 1}
        assert sample_weight is not None
        assert len(sample_weight) == len(X)
        assert np.isfinite(sample_weight).all()
        self.__class__.fit_calls.append(len(y))
        return self

    def predict_proba(self, X):
        # Deliberately depend on path length for synthetic discrimination.
        values = np.asarray(X)[:, diag.FEATURES.index("v2_path_length")]
        p = np.where(values > 0, 0.85, 0.15).astype(float)
        return np.column_stack([1.0 - p, p])


@pytest.fixture
def reports(frame, monkeypatch):
    FakeModel.fit_calls = []
    monkeypatch.setattr(diag, "create_model", lambda params: FakeModel())
    reports = diag.run_experiment(frame, {diag.FEATURE_SET: {}})
    return reports


def test_configuration():
    assert diag.FEATURE_SET == "extended_v2"
    assert len(diag.FEATURES) == 31
    assert len(diag.SOURCES) == 4
    assert len(diag.REPORT_FILES) == 3


def test_valid_data(frame):
    diag.validate_input(frame, {diag.FEATURE_SET: {}})


def test_missing_parameters(frame):
    with pytest.raises(ValueError, match="optimized"):
        diag.validate_input(frame, {})


def test_missing_feature(frame):
    with pytest.raises(ValueError, match="Missing columns"):
        diag.validate_input(frame.drop(columns=["v2_path_depth"]), {diag.FEATURE_SET: {}})


def test_bad_labels(frame):
    broken = frame.copy()
    broken.loc[0, "label"] = 3
    with pytest.raises(ValueError, match="binary labels"):
        diag.validate_input(broken, {diag.FEATURE_SET: {}})


def test_invalid_sources(frame):
    broken = frame.copy()
    broken.loc[0, "sources"] = "url_phish|unknown"
    with pytest.raises(ValueError, match="source membership"):
        diag.validate_input(broken, {diag.FEATURE_SET: {}})


def test_nonfinite_features(frame):
    broken = frame.copy()
    broken.loc[0, "v2_path_length"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        diag.validate_input(broken, {diag.FEATURE_SET: {}})


def test_path_groups():
    masks = diag.path_masks([0, 5, 6, 6], [0, 1, 2, 3])
    assert masks["root"].tolist() == [True, False, False, False]
    assert masks["nonempty"].tolist() == [False, True, True, True]
    assert masks["shallow_1_2"].tolist() == [False, True, True, False]
    assert masks["deep_3plus"].tolist() == [False, False, False, True]


def test_invalid_path_arrays():
    with pytest.raises(ValueError, match="equal-length"):
        diag.path_masks([0, 2], [0])
    with pytest.raises(ValueError, match="Negative"):
        diag.path_masks([0, -1], [0, 1])


def test_conditional_fpr_and_recall():
    detail = diag.summarize_predictions(
        "url_phish", labels=[0, 0, 1, 1],
        scores=[0.9, 0.1, 0.1, 0.9],
        lengths=[0, 5, 0, 5], depths=[0, 1, 0, 1],
    )
    def rate(label, group):
        return detail.loc[
            (detail["label"] == label) & (detail["path_group"] == group),
            "metric_value",
        ].iloc[0]
    assert rate(0, "root") == 1.0   # false positive
    assert rate(0, "nonempty") == 0.0
    assert rate(1, "root") == 0.0   # false negative
    assert rate(1, "nonempty") == 1.0


def test_high_confidence_errors():
    detail = diag.summarize_predictions(
        "phishbd", [0, 1], [0.95, 0.05], [0, 0], [0, 0]
    )
    assert (detail["high_confidence_errors"] == 1).all()
    assert (detail["high_confidence_error_fraction"] == 1).all()


def test_malformed_prediction_scores():
    with pytest.raises(ValueError, match="probabilities"):
        diag.summarize_predictions("x", [0, 1], [np.inf, 0.5], [0, 0], [0, 0])


def test_four_group_isolated_fits(reports):
    assert len(FakeModel.fit_calls) == 4
    assert set(reports) == set(diag.REPORT_FILES)
    assert len(reports["contrasts"]) == 8
    assert len(reports["macro"]) == 2


def test_contrasts_consistent(reports):
    c = reports["contrasts"]
    assert np.allclose(c["nonempty_minus_root"], c["nonempty_rate"] - c["root_rate"])
    assert (c["root_samples"] + c["nonempty_samples"] == c["all_samples"]).all()


def test_no_private_columns(reports):
    forbidden = {"url", "urls", "group_key", "domain", "hostname", "sources"}
    for result in reports.values():
        assert forbidden.isdisjoint(result.columns)


def test_save_reports(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "REPORT_DIR", tmp_path)
    diag.save_reports(reports)
    assert len(list(tmp_path.glob("path_bias_*.csv"))) == 3


def test_reject_sensitive_report(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "REPORT_DIR", tmp_path)
    modified = {name: result.copy() for name, result in reports.items()}
    modified["contrasts"]["url"] = "https://private.example"
    with pytest.raises(ValueError, match="sensitive"):
        diag.save_reports(modified)
    assert not list(tmp_path.glob("*.csv"))
