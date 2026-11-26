"""Synthetic tests: path-feature ablation. No production data or models touched."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment_path_feature_ablation as exp


# ============================================================
# 1. SYNTHETIC FEATURE CONFIGURATION
# ============================================================

@pytest.fixture
def patched_features(monkeypatch):
    features = list(dict.fromkeys([
        *exp.DIRECT_PATH,
        *exp.URL_WIDE_PROXIES,
        *exp.AUTHORITY_ONLY,
        *[f"filler_{i}" for i in range(13)],
    ]))

    assert len(features) == 31

    monkeypatch.setattr(exp, "FEATURES", features)

    return features


# ============================================================
# 2. SYNTHETIC TRAINING DATA
# ============================================================

@pytest.fixture
def training_data(patched_features):
    rows = []

    for source_id, source in enumerate(exp.SOURCES):

        for label in (0, 1):

            for i in range(8):

                row = {
                    "label": label,
                    "sources": source,
                    "group_key": f"example:{source}-{label}-{i}",

                    # Required for source_class sample weighting.
                    # Use valid binary values: 0 = HTTP, 1 = HTTPS.
                    "is_https": i % 2,

                    **{
                        f: float(
                            source_id * 0.2
                            + label * 0.1
                            + i * 0.01
                        )
                        for f in patched_features
                    },
                }

                row["v2_path_length"] = (
                    0.0 if i < 4 else 5.0
                )

                row["v2_path_depth"] = (
                    0.0 if i < 4 else 2.0
                )

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3. FAKE MODEL
# ============================================================

class FakeModel:

    classes_ = np.array([0, 1])

    fits = []

    def fit(
        self,
        X,
        y,
        sample_weight=None,
    ):

        assert set(np.unique(y)) == {0, 1}

        assert sample_weight is not None

        assert np.isfinite(
            sample_weight
        ).all()

        assert len(sample_weight) == len(y)

        type(self).fits.append(
            X.shape[1]
        )

        return self

    def predict_proba(self, X):

        positive = np.clip(
            0.20 + X[:, 0] * 0.5,
            0.001,
            0.999,
        )

        return np.column_stack([
            1 - positive,
            positive,
        ])


# ============================================================
# 4. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def result(training_data, monkeypatch):

    FakeModel.fits = []

    monkeypatch.setattr(
        exp,
        "create_model",
        lambda params: FakeModel(),
    )

    return exp.run_experiment(
        training_data,
        {"extended_v2": {}},
    )


# ============================================================
# 5. FEATURE CONFIGURATION TESTS
# ============================================================

def test_four_configurations(patched_features):

    configs = exp.build_configurations()

    assert list(configs) == [
        "v2_full",
        "without_direct_path",
        "without_direct_path_and_url_proxies",
        "authority_only",
    ]

    assert [
        len(v)
        for v in configs.values()
    ] == [
        31,
        27,
        25,
        12,
    ]


def test_path_columns_excluded(patched_features):

    configs = exp.build_configurations()

    assert not set(
        exp.DIRECT_PATH
    ).intersection(
        configs["without_direct_path"]
    )

    assert not set(
        exp.URL_WIDE_PROXIES
    ).intersection(
        configs["without_direct_path_and_url_proxies"]
    )

    assert configs["v2_full"] == patched_features


def test_invalid_feature_configuration(
    monkeypatch,
    patched_features,
):

    monkeypatch.setattr(
        exp,
        "FEATURES",
        patched_features[:-1],
    )

    # Different count alone is allowed.
    # Invalid feature names are the important guard.
    exp.build_configurations()

    monkeypatch.setattr(
        exp,
        "FEATURES",
        [
            f
            for f in patched_features
            if f != "v2_url_length"
        ],
    )

    with pytest.raises(
        ValueError,
        match="Missing expected",
    ):

        exp.build_configurations()


# ============================================================
# 6. INPUT VALIDATION TESTS
# ============================================================

def test_valid_data(training_data):

    exp.validate_data(
        training_data,
        {"extended_v2": {}},
    )


def test_missing_parameters(training_data):

    with pytest.raises(
        ValueError,
        match="optimized",
    ):

        exp.validate_data(
            training_data,
            {},
        )


def test_missing_feature(training_data):

    with pytest.raises(
        ValueError,
        match="Missing input",
    ):

        exp.validate_data(
            training_data.drop(
                columns="v2_path_entropy"
            ),
            {"extended_v2": {}},
        )


def test_invalid_source(training_data):

    altered = training_data.copy()

    altered.loc[0, "sources"] = "unknown"

    with pytest.raises(
        ValueError,
        match="source membership",
    ):

        exp.validate_data(
            altered,
            {"extended_v2": {}},
        )


def test_nonfinite_values(training_data):

    altered = training_data.copy()

    altered["v2_path_length"] = (
        altered["v2_path_length"].astype(float)
    )

    altered.loc[0, "v2_path_length"] = np.inf

    with pytest.raises(
        ValueError,
        match="Non-finite",
    ):

        exp.validate_data(
            altered,
            {"extended_v2": {}},
        )


# ============================================================
# 7. BINARY METRICS TESTS
# ============================================================

def test_binary_metrics():

    metrics = exp.binary_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.4, 0.9],
    )

    assert (
        metrics["tp"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
    ) == (
        1,
        1,
        1,
        1,
    )

    assert metrics["recall"] == pytest.approx(
        0.5
    )

    assert metrics["fpr"] == pytest.approx(
        0.5
    )


# ============================================================
# 8. PATH GROUP TESTS
# ============================================================

def test_path_groups(training_data):

    frame = training_data.iloc[:16]

    scores = np.array(
        [0.9] * 16
    )

    rows = exp.path_group_rows(
        "legitphish",
        "v2_full",
        frame,
        scores,
    )

    assert len(rows) == 4

    assert {
        r["path_group"]
        for r in rows
    } == {
        "root",
        "nonempty",
    }

    assert all(
        r["samples"] == 4
        for r in rows
    )


# ============================================================
# 9. TRAINING TESTS
# ============================================================

def test_sixteen_trainings(result):

    assert len(
        FakeModel.fits
    ) == 16

    assert sorted(
        FakeModel.fits
    ) == sorted(
        [31, 27, 25, 12] * 4
    )

    assert len(
        result["by_source"]
    ) == 16

    assert len(
        result["by_path"]
    ) == 64

    assert len(
        result["macro"]
    ) == 4


# ============================================================
# 10. MACRO METRICS TESTS
# ============================================================

def test_macro_deltas(result):

    summary = result["macro"]

    assert (
        summary.loc[
            summary["configuration"] == "v2_full",
            "delta_macro_fpr",
        ].iloc[0]
        == 0
    )

    base = summary.loc[
        summary["configuration"] == "v2_full",
        "macro_f1",
    ].iloc[0]

    assert np.allclose(
        summary["delta_macro_f1"],
        summary["macro_f1"] - base,
    )


# ============================================================
# 11. PRIVACY TESTS
# ============================================================

def test_sensitive_data_not_exported(result):

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    for frame in result.values():

        assert forbidden.isdisjoint(
            frame.columns
        )


# ============================================================
# 12. REPORT SAVING TESTS
# ============================================================

def test_save_reports(
    result,
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        exp,
        "REPORT_DIR",
        tmp_path,
    )

    exp.save_reports(result)

    assert len(
        list(
            tmp_path.glob(
                "path_feature_ablation_*.csv"
            )
        )
    ) == 3


def test_reject_sensitive_report(
    result,
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        exp,
        "REPORT_DIR",
        tmp_path,
    )

    altered = {
        name: frame.copy()
        for name, frame in result.items()
    }

    altered["macro"]["url"] = (
        "https://private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        exp.save_reports(altered)

    assert not list(
        tmp_path.glob("*.csv")
    )