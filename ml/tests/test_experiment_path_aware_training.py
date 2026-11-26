"""Synthetic, offline tests for path-aware sample weighting in PhishGuard AI."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_path_aware_training as exp


@pytest.fixture
def training_data():
    rows = []
    for source_id, source in enumerate(exp.SOURCES):
        for label in (0, 1):
            for i in range(8):
                row = {
                    "sources": source,
                    "group_key": f"domain:{source}-{label}-{i}.example",
                    "label": label,
                    "is_https": i % 2,
                    **{name: float(source_id * 0.3 + label * 0.2 + i * 0.05 + j * 0.003)
                       for j, name in enumerate(exp.FEATURES)},
                }
                row["v2_path_length"] = 0.0 if i < 4 else 12.0
                row["v2_path_depth"] = 0.0 if i < 4 else 2.0
                rows.append(row)
    return pd.DataFrame(rows)


class FakeModel:
    classes_ = np.array([0, 1])
    fits = []

    def fit(self, X, y, sample_weight=None):
        assert set(y) == {0, 1}
        assert sample_weight is not None
        assert np.isfinite(sample_weight).all()
        assert (sample_weight > 0).all()
        assert len(sample_weight) == len(y)
        type(self).fits.append(X.shape[1])
        return self

    def predict_proba(self, X):
        values = np.clip(0.1 + X[:, 0] * 0.25 + (X.shape[1] == 27) * 0.05,
                         0.001, 0.999)
        return np.column_stack((1 - values, values))


@pytest.fixture
def reports(training_data, monkeypatch):
    FakeModel.fits = []
    monkeypatch.setattr(exp, "create_model", lambda params: FakeModel())
    return exp.run_experiment(training_data, {"extended_v2": {}})


def test_configuration():
    variants = exp.feature_variants()
    assert list(variants) == [
        "v2_full", "v2_27_reference", "v2_27_path_x2", "v2_27_path_x4",
    ]
    assert [len(f) for f, _ in variants.values()] == [31, 27, 27, 27]
    assert [v for _, v in variants.values()] == [1.0, 1.0, 2.0, 4.0]
    assert len(exp.REPORT_FILES) == 4


def test_valid_input(training_data):
    exp.validate_inputs(training_data, {"extended_v2": {}})


def test_missing_https(training_data):
    with pytest.raises(ValueError, match="is_https"):
        exp.validate_inputs(training_data.drop(columns="is_https"), {"extended_v2": {}})


def test_bad_https(training_data):
    modified = training_data.copy()
    modified.loc[0, "is_https"] = 5
    with pytest.raises(ValueError, match="is_https"):
        exp.validate_inputs(modified, {"extended_v2": {}})


def test_missing_optimized_params(training_data):
    with pytest.raises(ValueError, match="optimized"):
        exp.validate_inputs(training_data, {})


def test_exact_reference_weights(training_data):
    expected = exp.calculate_weights(training_data, "source_class")
    actual = exp.path_aware_weights(training_data, 1.0)
    np.testing.assert_array_equal(actual, expected)


def test_2x_boost_legitimate_paths_and_preserve_class_mass(training_data):
    base = exp.path_aware_weights(training_data, 1.0)
    boosted = exp.path_aware_weights(training_data, 2.0)
    y = training_data["label"].to_numpy()
    path = training_data["v2_path_length"].to_numpy() > 0
    np.testing.assert_array_equal(boosted[y == 1], base[y == 1])
    assert boosted[y == 0].sum() == pytest.approx(base[y == 0].sum())
    # Ratio of weighted path/root within the legitimate class is doubled.
    ratio_before = base[(y == 0) & path].sum() / base[(y == 0) & ~path].sum()
    ratio_after = boosted[(y == 0) & path].sum() / boosted[(y == 0) & ~path].sum()
    assert ratio_after / ratio_before == pytest.approx(2.0)


def test_4x_boost_is_finite_positive_and_bounded_relative(training_data):
    base = exp.path_aware_weights(training_data, 1.0)
    boosted = exp.path_aware_weights(training_data, 4.0)
    assert np.isfinite(boosted).all()
    assert (boosted > 0).all()
    assert np.max(boosted / base) <= 4.0 + 1e-10


@pytest.mark.parametrize("factor", [0, -1, 1.5, 100, float("nan"), float("inf")])
def test_reject_unsupported_multipliers(training_data, factor):
    with pytest.raises(ValueError, match="multiplier"):
        exp.path_aware_weights(training_data, factor)


def test_reject_invalid_path_length(training_data):
    modified = training_data.copy()
    modified["v2_path_length"] = modified["v2_path_length"].astype(float)
    modified.loc[0, "v2_path_length"] = -1
    with pytest.raises(ValueError, match="path"):
        exp.path_aware_weights(modified, 2.0)


def test_no_benign_path_reweighting_is_noop(training_data):
    modified = training_data.copy()
    modified.loc[modified["label"] == 0, "v2_path_length"] = 0.0
    before = exp.path_aware_weights(modified, 1.0)
    after = exp.path_aware_weights(modified, 4.0)
    np.testing.assert_array_equal(before, after)


def test_sixteen_trainings_on_group_isolated_folds(reports):
    assert len(FakeModel.fits) == 16
    assert FakeModel.fits.count(31) == 4
    assert FakeModel.fits.count(27) == 12
    assert len(reports["by_source"]) == 16
    assert len(reports["by_path"]) == 64
    assert len(reports["macro"]) == 4
    assert len(reports["weights"]) == 16


def test_macro_deltas_consistent(reports):
    macro = reports["macro"]
    reference = macro.loc[macro["configuration"] == "v2_27_reference"].iloc[0]
    for metric in ("macro_f1", "macro_recall", "macro_fpr", "worst_recall", "worst_fpr"):
        np.testing.assert_allclose(
            macro[f"delta_{metric}_vs_27"], macro[metric] - reference[metric]
        )


def test_weight_diagnostics_consistency(reports):
    weights = reports["weights"]
    assert (weights["legitimate_path_weight_fraction"] >= 0).all()
    assert (weights["legitimate_path_weight_fraction"] <= 1).all()
    assert (weights["effective_sample_size"] > 0).all()
    assert (weights["effective_sample_size"] <= weights["training_samples"] + 1e-9).all()
    assert (weights["legitimate_path_training_samples"] > 0).all()


def test_reports_do_not_contain_private_columns(reports):
    forbidden = {"url", "urls", "sources", "group_key", "hostname", "domain"}
    assert all(forbidden.isdisjoint(df.columns) for df in reports.values())


def test_save_reports(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "REPORT_DIR", tmp_path)
    exp.save_reports(reports)
    assert len(list(tmp_path.glob("path_aware_training_*.csv"))) == 4


def test_reject_sensitive_report(reports, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "REPORT_DIR", tmp_path)
    modified = {name: df.copy() for name, df in reports.items()}
    modified["macro"]["url"] = "https://example.invalid/private"
    with pytest.raises(ValueError, match="sensitive"):
        exp.save_reports(modified)
    assert not list(tmp_path.glob("*.csv"))
