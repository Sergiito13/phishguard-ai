"""
PhishGuard AI - Protocol Invariance Tests

Tests:
- HTTP / HTTPS transformation.
- V1 and V2 feature extraction.
- Exact protocol invariance for V2.
- Reproducible sampling.
- Probability sensitivity.
- Floating-point consistency.
- Cross-source evaluation.
- Report generation.

All tests use synthetic URLs and fake classifiers.
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULE
# ============================================================

SRC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import experiment_protocol_invariance as experiment


# ============================================================
# 2. SYNTHETIC TRAINING DATA
# ============================================================

@pytest.fixture
def tiny_training_data():

    records = []

    for source in experiment.SOURCES:

        for label in (0, 1):

            for n in range(3):

                scheme = (
                    "https"
                    if n % 2
                    else "http"
                )

                url = (
                    f"{scheme}://"
                    f"{source}-{label}-{n}.example"
                    "/login?a=1"
                )

                records.append({
                    "url": url,

                    "label": label,

                    "sources": source,

                    "group_key": (
                        f"domain:{source}-{label}-{n}.example"
                    ),

                    **experiment.extract_url_features(url),

                    **experiment.extract_url_features_v2(url),
                })

    return pd.DataFrame.from_records(
        records
    )


# ============================================================
# 3. FAKE MODEL
# ============================================================

class FakeModel:

    fits = 0

    def fit(
        self,
        X,
        y,
    ):

        assert set(
            np.unique(y)
        ) == {0, 1}

        FakeModel.fits += 1

        return self

    def predict_proba(
        self,
        X,
    ):

        probabilities = np.clip(
            0.05 + X[:, 0] / 100,
            0.01,
            0.99,
        )

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])


# ============================================================
# 4. HTTP / HTTPS TRANSFORMATION TESTS
# ============================================================

def test_swap_http_to_https():

    result = experiment.swap_scheme(
        "http://example.com/a?a=1"
    )

    assert result == (
        "https://example.com/a?a=1",
        "http",
    )


def test_swap_https_to_http():

    result = experiment.swap_scheme(
        "HTTPS://example.com/login"
    )

    assert result == (
        "http://example.com/login",
        "https",
    )


@pytest.mark.parametrize(
    "url",
    [
        "example.com",
        "ftp://example.com",
        "",
        None,
    ],
)
def test_reject_unsupported_scheme(url):

    with pytest.raises(
        ValueError,
        match="http",
    ):

        experiment.swap_scheme(url)


# ============================================================
# 5. FEATURE EXTRACTION TESTS
# ============================================================

def test_v1_features_change_after_swap():

    features = experiment.FEATURE_SETS[
        "hybrid_v1_extended"
    ]

    original = experiment.extract_vector(
        "http://example.com/",
        features,
    )

    changed = experiment.extract_vector(
        "https://example.com/",
        features,
    )

    assert (
        original.shape
        == changed.shape
        == (len(features),)
    )

    https_index = features.index(
        "is_https"
    )

    length_index = features.index(
        "url_length"
    )

    assert original[https_index] == 0

    assert changed[https_index] == 1

    assert (
        original[length_index]
        != changed[length_index]
    )


def test_v2_features_are_scheme_independent():

    features = experiment.FEATURE_SETS[
        "extended_v2"
    ]

    original = experiment.extract_vector(
        "http://example.com/login?a=1",
        features,
    )

    changed = experiment.extract_vector(
        "https://example.com/login?a=1",
        features,
    )

    # IMPORTANT:
    # Require exact equality, not approximate equality.

    np.testing.assert_array_equal(
        original,
        changed,
    )


# ============================================================
# 6. SAMPLE SELECTION TESTS
# ============================================================

def test_sample_is_capped_and_reproducible(
    tiny_training_data,
):

    selected_a = experiment.select_probe_rows(
        tiny_training_data,
        max_per_label=3,
    )

    selected_b = experiment.select_probe_rows(
        tiny_training_data,
        max_per_label=3,
    )

    assert len(selected_a) == 6

    assert (
        selected_a.index.tolist()
        == selected_b.index.tolist()
    )

    assert set(
        selected_a["label"]
    ) == {0, 1}


def test_sample_requires_two_classes(
    tiny_training_data,
):

    only_legitimate = tiny_training_data.loc[
        tiny_training_data["label"] == 0
    ]

    with pytest.raises(
        ValueError,
        match="Each class",
    ):

        experiment.select_probe_rows(
            only_legitimate
        )


def test_sample_requires_positive_limit(
    tiny_training_data,
):

    with pytest.raises(
        ValueError,
        match="max_per_label",
    ):

        experiment.select_probe_rows(
            tiny_training_data,
            max_per_label=0,
        )


# ============================================================
# 7. PROBABILITY METRICS TESTS
# ============================================================

def test_flip_metric():

    result = experiment.change_metrics(
        np.array([
            0.49,
            0.10,
            0.80,
        ]),

        np.array([
            0.50,
            0.40,
            0.10,
        ]),
    )

    assert result[
        "flipped_predictions"
    ] == 2

    assert result[
        "flip_fraction"
    ] == pytest.approx(
        2 / 3
    )

    assert result[
        "mean_absolute_delta"
    ] == pytest.approx(
        (0.01 + 0.30 + 0.70) / 3
    )


def test_aligned_scores_required():

    with pytest.raises(ValueError):

        experiment.change_metrics(
            np.array([0.5]),
            np.array([0.3, 0.9]),
        )


# ============================================================
# 8. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def results(
    monkeypatch,
    tiny_training_data,
):

    FakeModel.fits = 0

    monkeypatch.setattr(
        experiment,
        "create_model",
        lambda params: FakeModel(),
    )

    parameters = {
        name: {}
        for name in experiment.MODELS
    }

    data = experiment.run_experiment(
        tiny_training_data,
        parameters,
        max_per_label=3,
    )

    assert FakeModel.fits == 8

    return data


# ============================================================
# 9. EXPERIMENT RESULTS
# ============================================================

def test_run_produces_three_reports(
    results,
):

    assert set(results) == {
        "detail",
        "summary",
        "feature_changes",
    }

    # 8 scheme-swap evaluations.
    # 4 additional HTTPS-flag-only evaluations.

    assert len(
        results["summary"]
    ) == 12

    expected_features = sum(
        len(
            experiment.FEATURE_SETS[model]
        ) * len(experiment.SOURCES)

        for model in experiment.MODELS
    )

    assert len(
        results["feature_changes"]
    ) == expected_features

    assert (
        results["summary"]["samples"] == 6
    ).all()

    assert (
        results["summary"]["flip_fraction"]
        .between(0, 1)
    ).all()


def test_v2_swap_changes_no_features(
    results,
):

    v2 = results["feature_changes"].loc[
        results["feature_changes"]["feature_set"]
        == "extended_v2"
    ]

    assert (
        v2["changed_count"] == 0
    ).all()

    v2_summaries = results["summary"].loc[
        results["summary"]["feature_set"]
        == "extended_v2"
    ]

    assert (
        v2_summaries["mean_absolute_delta"]
        .eq(0.0)
        .all()
    )


# ============================================================
# 10. FLOATING-POINT CONSISTENCY TEST
# ============================================================

def test_float_csv_rounding_does_not_create_v2_score_changes(
    monkeypatch,
    tiny_training_data,
):
    """
    Tiny floating-point differences in stored
    features must not be mistaken for
    protocol sensitivity.
    """

    rounded = tiny_training_data.copy()

    rounded["v2_url_length"] = (
        rounded["v2_url_length"].astype(float)
        + 1e-7
    )

    monkeypatch.setattr(
        experiment,
        "create_model",
        lambda params: FakeModel(),
    )

    reports = experiment.run_experiment(
        rounded,

        {
            name: {}
            for name in experiment.MODELS
        },

        max_per_label=3,
    )

    v2 = reports["summary"].loc[
        reports["summary"]["feature_set"]
        == "extended_v2"
    ]

    assert (
        v2["mean_absolute_delta"]
        .eq(0.0)
        .all()
    )

    assert (
        v2["flipped_predictions"]
        .eq(0)
        .all()
    )


# ============================================================
# 11. FEATURE CONSISTENCY TESTS
# ============================================================

def test_reject_feature_mismatch(
    tiny_training_data,
):

    bad = tiny_training_data.copy()

    bad.loc[
        0,
        "url_length",
    ] += 10

    parameters = {
        name: {}
        for name in experiment.MODELS
    }

    with pytest.raises(
        ValueError,
        match="reproducibility",
    ):

        experiment.run_experiment(
            bad,
            parameters,
            max_per_label=3,
        )


def test_reject_missing_optimized_parameters(
    tiny_training_data,
):

    with pytest.raises(
        ValueError,
        match="Missing optimized",
    ):

        experiment.run_experiment(
            tiny_training_data,
            {},
        )


# ============================================================
# 12. PRIVACY TESTS
# ============================================================

def test_reports_never_contain_urls(
    results,
):

    forbidden = {
        "url",
        "urls",
        "group_key",
        "hostname",
        "domain",
    }

    for dataframe in results.values():

        assert forbidden.isdisjoint(
            dataframe.columns
        )


# ============================================================
# 13. REPORT GENERATION TESTS
# ============================================================

def test_save_csvs(
    results,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.save_reports(
        results
    )

    files = list(
        tmp_path.glob(
            "protocol_invariance_*.csv"
        )
    )

    assert len(files) == 3

    for file in files:

        assert not pd.read_csv(
            file
        ).empty


def test_refuse_sensitive_report(
    results,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    results["detail"]["url"] = (
        "http://private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        experiment.save_reports(
            results
        )

    assert not list(
        tmp_path.glob("*.csv")
    )


# ============================================================
# 14. COMPLETE WORKFLOW TEST
# ============================================================

def test_full_main_on_synthetic_data(
    tiny_training_data,
    tmp_path,
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        experiment,
        "create_model",
        lambda params: FakeModel(),
    )

    monkeypatch.setattr(
        experiment,
        "load_training_data",
        lambda: tiny_training_data,
    )

    monkeypatch.setattr(
        experiment,
        "load_best_parameters",
        lambda: {
            name: {}
            for name in experiment.MODELS
        },
    )

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.main()

    output = capsys.readouterr().out

    assert (
        "PROTOCOL SENSITIVITY SUMMARY"
        in output
    )

    assert len(
        list(
            tmp_path.glob("*.csv")
        )
    ) == 3