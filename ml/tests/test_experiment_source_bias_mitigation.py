"""Synthetic tests for source-bias mitigation. Never load real datasets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment_source_bias_mitigation as exp


@pytest.fixture
def frame():
    features = set().union(*(set(exp.FEATURE_SETS[name]) for name in exp.MODELS))
    rows = []
    for source in exp.SOURCES:
        for label in (0, 1):
            for i in range(5):
                value = .2 if label == 0 else .8
                if i == 0:
                    value = 1.0 - value
                record = {"url": f"https://{source}-{label}-{i}.example",
                          "group_key": f"group:{source}-{label}-{i}",
                          "sources": source, "label": label, "is_https": i % 2}
                record.update({feature: value for feature in features})
                record["is_https"] = i % 2
                rows.append(record)
    return pd.DataFrame(rows)


class FakeModel:
    fits = 0

    def fit(self, X, y, sample_weight=None):
        type(self).fits += 1
        assert len(X) == len(y)
        if sample_weight is not None:
            assert len(sample_weight) == len(y)
            assert (sample_weight > 0).all()
        return self

    def predict_proba(self, X):
        score = np.clip(X[:, 0], .01, .99)
        return np.column_stack([1-score, score])


@pytest.fixture
def trained(frame, monkeypatch):
    FakeModel.fits = 0
    monkeypatch.setattr(exp, "create_model", lambda params: FakeModel())
    rows, diag = exp.run_experiment(frame, {name: {} for name in exp.MODELS})
    assert FakeModel.fits == 32
    return rows, diag


def test_configurations():
    assert len(exp.MODELS) == 2
    assert len(exp.SOURCES) == 4
    assert len(exp.STRATEGIES) == 4
    assert exp.MIN_CELL_MASS > 0


def test_uniform_weights_none(frame):
    assert exp.calculate_weights(frame, "uniform") is None


@pytest.mark.parametrize("strategy", exp.STRATEGIES[1:])
def test_weights_finite_normalized(frame, strategy):
    w = exp.calculate_weights(frame, strategy)
    assert len(w) == len(frame)
    assert np.isfinite(w).all()
    assert (w > 0).all()
    assert w.mean() == pytest.approx(1)


def test_fractional_memberships_supports_shared_urls(frame):
    frame.loc[0, "sources"] = "legitphish|phiusiil"
    members = exp.source_memberships(frame)
    assert members[0] == ("legitphish", "phiusiil")
    for strategy in exp.STRATEGIES[1:]:
        assert np.isfinite(exp.calculate_weights(frame, strategy)).all()


@pytest.mark.parametrize("members", ["", "unknown", "legitphish|legitphish", None])
def test_invalid_source_memberships(frame, members):
    frame.loc[0, "sources"] = members
    with pytest.raises(ValueError, match="Invalid source"):
        exp.source_memberships(frame)


def test_invalid_https_rejected(frame):
    frame.loc[0, "is_https"] = 3
    with pytest.raises(ValueError, match="is_https"):
        exp.calculate_weights(frame, "source_class_https")


def test_invalid_strategy(frame):
    with pytest.raises(ValueError, match="Unknown strategy"):
        exp.calculate_weights(frame, "bad")


def test_weight_cap(frame):
    frame.loc[:10, "sources"] = "legitphish|phiusiil"
    w = exp.calculate_weights(frame, "source_class_https")
    assert w.max() <= 100  # Absolute cap after renormalization can exceed relative cap.


def test_weight_diagnostics(frame):
    weights = exp.calculate_weights(frame, "source")
    report = exp.weight_diagnostics(weights, "m", "s", "source", len(frame))
    assert 0 < report["effective_sample_size"] <= len(frame) + 1e-8


def test_full_experiment(trained):
    rows, diag = trained
    assert len(rows) == 32
    assert len(diag) == 32
    assert not rows[["feature_set", "evaluation_source", "strategy"]].duplicated().any()
    assert rows["recall"].between(0, 1).all()
    assert rows["fpr"].between(0, 1).all()


def test_reject_missing_parameters(frame):
    with pytest.raises(ValueError, match="Missing optimized"):
        exp.run_experiment(frame, {})


def test_reject_group_leakage(frame, monkeypatch):
    folds = [{"source": source, "training_indices": np.arange(len(frame)),
              "validation_indices": np.flatnonzero(frame["sources"].eq(source))}
             for source in exp.SOURCES]
    monkeypatch.setattr(exp, "create_source_folds", lambda data: folds)
    with pytest.raises(ValueError, match="Group leakage"):
        exp.run_experiment(frame, {name: {} for name in exp.MODELS})


def test_summary_and_baseline(trained):
    rows, _ = trained
    summary = exp.summarize_results(rows)
    assert len(summary) == 8
    comparison = exp.compare_to_baseline(summary)
    baseline = comparison[comparison["strategy"] == "uniform"]
    assert (baseline["delta_macro_fpr"] == 0).all()
    assert (baseline["delta_macro_f1"] == 0).all()


def test_summary_rejects_missing_row(trained):
    rows, _ = trained
    with pytest.raises(ValueError, match="Incomplete"):
        exp.summarize_results(rows.iloc[:-1])


def test_reports_are_aggregate(trained, monkeypatch, tmp_path):
    rows, diag = trained
    summary = exp.summarize_results(rows)
    comparison = exp.compare_to_baseline(summary)
    monkeypatch.setattr(exp, "REPORT_DIR", tmp_path)
    exp.save_reports(rows, summary, comparison, diag)
    assert len(list(tmp_path.glob("source_bias_mitigation_*.csv"))) == 4
    assert pd.read_csv(tmp_path / "source_bias_mitigation_results.csv").shape[0] == 32


def test_private_urls_rejected(trained, monkeypatch, tmp_path):
    rows, diag = trained
    summary = exp.summarize_results(rows)
    comparison = exp.compare_to_baseline(summary)
    monkeypatch.setattr(exp, "REPORT_DIR", tmp_path)
    with pytest.raises(ValueError, match="sensitive"):
        exp.save_reports(rows.assign(url="https://private.example"), summary, comparison, diag)
    assert not list(tmp_path.glob("*.csv"))


def test_main_uses_synthetic_data_only(frame, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(exp, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(exp, "load_training_data", lambda: frame)
    monkeypatch.setattr(exp, "load_best_parameters", lambda: {name: {} for name in exp.MODELS})
    monkeypatch.setattr(exp, "create_model", lambda params: FakeModel())
    exp.main()
    output = capsys.readouterr().out
    assert "WEIGHTING COMPARISON" in output
    assert "FOCAL SOURCE RESULTS" in output
    assert len(list(tmp_path.glob("*.csv"))) == 4
