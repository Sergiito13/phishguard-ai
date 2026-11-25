"""Synthetic tests for PhishGuard AI V3 feature group ablation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_v3_group_ablation as ab


class FakeModel:
    classes_ = np.array([0, 1])
    fit_calls = []

    def fit(self, X, y, sample_weight=None):
        assert set(y) == {0, 1}
        assert sample_weight is not None
        assert len(X) == len(sample_weight)
        assert np.all(np.isfinite(sample_weight))
        self.__class__.fit_calls.append(X.shape[1])
        return self

    def predict_proba(self, X):
        score = np.clip(0.15 + 0.12 * X[:, 0], 0.01, 0.99)
        return np.column_stack([1 - score, score])


@pytest.fixture
def fake_frame():
    rows = []
    for source_index, source in enumerate(ab.SOURCES):
        for label in (0, 1):
            for index in range(4):
                rows.append({
                    "label": label,
                    "sources": source,
                    "group_key": f"domain:{source}-{label}-{index}",
                    "is_https": index % 2,
                    **{
                        feature: float(
                            source_index + 4 * label + 0.2 * index + fidx / 100
                        )
                        for fidx, feature in enumerate(ab.FEATURE_COLUMNS_EXTENDED_V3)
                    },
                })
    return pd.DataFrame(rows)


@pytest.fixture
def fake_source_results():
    rows = []
    for c, features in ab.build_configurations().items():
        for i, source in enumerate(ab.SOURCES):
            rows.append({
                "configuration": c,
                "evaluation_source": source,
                "feature_count": len(features),
                "f1": .5 + i * .01 + (0 if c == "v2_reference" else .02),
                "recall": .6 + i * .01 + (0 if c == "v2_reference" else .01),
                "fpr": .1 + i * .01 + (0 if c == "v2_reference" else .005),
            })
    return pd.DataFrame(rows)


def test_five_unique_configs():
    configs = ab.build_configurations()
    assert len(configs) == 5
    assert [len(x) for x in configs.values()] == [31, 37, 35, 34, 36]
    assert len({tuple(v) for v in configs.values()}) == 5


def test_feature_groups_cover_exactly_six_new_columns():
    groups = ab.V3_GROUPS
    assert set(groups) == {"auth", "path_shape", "query_entropy"}
    assert set().union(*groups.values()) == (
        set(ab.FEATURE_COLUMNS_EXTENDED_V3) - set(ab.FEATURE_COLUMNS_V2)
    )


def test_summary_deltas_and_shape(fake_source_results):
    by_source, summary = ab.summarize(fake_source_results)
    assert len(by_source) == 20
    assert len(summary) == 5
    baseline = summary.loc[summary["configuration"] == "v2_reference"].iloc[0]
    assert baseline["delta_macro_f1_vs_v2"] == pytest.approx(0)
    v3 = summary.loc[summary["configuration"] == "v3_full"].iloc[0]
    assert v3["delta_macro_f1_vs_v2"] == pytest.approx(.02)
    assert v3["delta_macro_recall_vs_v2"] == pytest.approx(.01)
    assert v3["delta_macro_fpr_vs_v2"] == pytest.approx(.005)


def test_missing_or_duplicate_source_rows_rejected(fake_source_results):
    with pytest.raises(ValueError, match="Incomplete"):
        ab.summarize(fake_source_results.iloc[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        ab.summarize(pd.concat([fake_source_results, fake_source_results.iloc[:1]]))


def test_reports_do_not_contain_private_data(fake_source_results):
    by_source, summary = ab.summarize(fake_source_results)
    forbidden = {"url", "domain", "group_key", "sources", "hostname"}
    for frame in [fake_source_results, by_source, summary]:
        assert forbidden.isdisjoint(frame.columns)


def test_full_main_with_synthetic_data(fake_frame, tmp_path, monkeypatch, capsys):
    FakeModel.fit_calls = []
    monkeypatch.setattr(ab, "load_aligned_training", lambda: fake_frame)
    monkeypatch.setattr(ab, "load_best_parameters", lambda: {"extended_v2": {}})
    monkeypatch.setattr(ab, "create_model", lambda params: FakeModel())
    monkeypatch.setattr(ab, "REPORT_DIR", tmp_path)
    ab.main()
    assert len(FakeModel.fit_calls) == 20
    assert set(FakeModel.fit_calls) == {31, 34, 35, 36, 37}
    assert len(list(tmp_path.glob("v3_group_ablation_*.csv"))) == 3
    assert "GROUP ABLATION MACRO SUMMARY" in capsys.readouterr().out
