"""
PhishGuard AI - Probability Calibration Tests

Validate:
- Calibration experiment configuration.
- Domain-group isolation.
- Reproducible calibration splits.
- Sigmoid and isotonic calibration.
- Probability validation.
- Expected Calibration Error (ECE).
- Classification and calibration metrics.
- Cross-source experiment execution.
- CSV report generation.

All tests use synthetic data.
No real datasets or trained models are modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULE
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import experiment_calibration as calibration


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def synthetic_data():
    """
    Create balanced synthetic data for every source.

    Each domain group is unique.
    Every source contains both classes.
    """

    features = set()

    for candidate in calibration.CANDIDATES:

        features.update(
            calibration.FEATURE_SETS[candidate]
        )

    records = []

    for source in calibration.SOURCES:

        for label in (0, 1):

            for index in range(10):

                record = {
                    "url": (
                        f"https://{source}-{label}-{index}.example"
                    ),
                    "label": label,
                    "sources": source,
                    "group_key": (
                        f"domain:{source}-{label}-{index}.example"
                    ),
                }

                value = (
                    0.2 if label == 0 else 0.8
                ) + index * 0.001

                record.update({
                    feature: value
                    for feature in features
                })

                records.append(record)

    return pd.DataFrame(records)


@pytest.fixture
def best_parameters():
    """
    Lightweight parameters used only in mocked tests.
    """

    return {
        name: {"max_iter": 10}
        for name in calibration.CANDIDATES
    }


# ============================================================
# 3. FAKE CLASSIFIER
# ============================================================

class FakeClassifier:
    """
    Deterministic classifier.

    Prevents expensive real model training
    during unit tests.
    """

    fits = 0

    def fit(self, X, y):

        assert set(
            np.unique(y)
        ) == {0, 1}

        type(self).fits += 1

        return self

    def predict_proba(self, X):

        probabilities = np.clip(
            X[:, 0],
            0.01,
            0.99,
        )

        return np.column_stack(
            (
                1 - probabilities,
                probabilities,
            )
        )


@pytest.fixture
def synthetic_results(
    monkeypatch,
    synthetic_data,
    best_parameters,
):
    """
    Execute the experiment using a fake classifier.
    """

    monkeypatch.setattr(
        calibration,
        "create_model",
        lambda params: FakeClassifier(),
    )

    return calibration.run_experiment(
        synthetic_data,
        best_parameters,
    )


# ============================================================
# 4. EXPERIMENT CONFIGURATION
# ============================================================

def test_experiment_configuration():

    assert set(
        calibration.CANDIDATES
    ) == {
        "hybrid_v1_extended",
        "extended_v2",
    }

    assert set(
        calibration.METHODS
    ) == {
        "none",
        "sigmoid",
        "isotonic",
    }

    assert calibration.THRESHOLD == 0.50


# ============================================================
# 5. CALIBRATION DATA SPLITTING
# ============================================================

def test_group_isolation_in_calibration_split(
    synthetic_data,
):

    fit_df, calibration_df = (
        calibration.split_training_calibration(
            synthetic_data
        )
    )

    assert set(
        fit_df["label"]
    ) == {0, 1}

    assert set(
        calibration_df["label"]
    ) == {0, 1}

    assert set(
        fit_df["group_key"]
    ).isdisjoint(
        calibration_df["group_key"]
    )

    assert (
        len(fit_df)
        + len(calibration_df)
        == len(synthetic_data)
    )


def test_calibration_split_is_reproducible(
    synthetic_data,
):

    first_fit, first_calibration = (
        calibration.split_training_calibration(
            synthetic_data
        )
    )

    second_fit, second_calibration = (
        calibration.split_training_calibration(
            synthetic_data
        )
    )

    assert (
        first_fit["url"].tolist()
        == second_fit["url"].tolist()
    )

    assert (
        first_calibration["url"].tolist()
        == second_calibration["url"].tolist()
    )


def test_calibration_split_rejects_one_class(
    synthetic_data,
):

    one_class = synthetic_data.loc[
        synthetic_data["label"] == 0
    ]

    with pytest.raises(
        ValueError,
        match="Could not produce",
    ):

        calibration.split_training_calibration(
            one_class
        )


# ============================================================
# 6. LOGIT TRANSFORMATION
# ============================================================

def test_logit_handles_extreme_probabilities():

    probabilities = np.array([
        0.0,
        0.5,
        1.0,
    ])

    result = calibration.logit(
        probabilities
    )

    assert result.shape == (3, 1)

    assert np.isfinite(
        result
    ).all()

    assert result[0, 0] < 0

    assert result[1, 0] == pytest.approx(
        0.0
    )

    assert result[2, 0] > 0


# ============================================================
# 7. CALIBRATION METHODS
# ============================================================

def test_calibration_methods_preserve_probability_range():

    calibration_probabilities = np.array([
        0.05,
        0.10,
        0.20,
        0.35,
        0.65,
        0.80,
        0.90,
        0.95,
    ])

    labels = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    evaluation_probabilities = np.array([
        0.12,
        0.42,
        0.76,
        0.98,
    ])

    for method in calibration.METHODS:

        result = calibration.calibrate(
            method,
            calibration_probabilities,
            labels,
            evaluation_probabilities,
        )

        assert (
            result.shape
            == evaluation_probabilities.shape
        )

        assert np.isfinite(
            result
        ).all()

        assert (
            (result >= 0)
            & (result <= 1)
        ).all()

        if method == "none":

            np.testing.assert_allclose(
                result,
                evaluation_probabilities,
            )


def test_invalid_calibration_method():

    with pytest.raises(
        ValueError,
        match="Unknown calibration method",
    ):

        calibration.calibrate(
            "unknown",
            np.array([0.2, 0.8]),
            np.array([0, 1]),
            np.array([0.4]),
        )


# ============================================================
# 8. EXPECTED CALIBRATION ERROR
# ============================================================

def test_ece_perfect_and_imperfect():

    labels = np.array([
        0,
        1,
    ])

    perfect_probabilities = np.array([
        0.0,
        1.0,
    ])

    imperfect_probabilities = np.array([
        0.2,
        0.8,
    ])

    perfect_ece = (
        calibration.expected_calibration_error(
            labels,
            perfect_probabilities,
        )
    )

    imperfect_ece = (
        calibration.expected_calibration_error(
            labels,
            imperfect_probabilities,
        )
    )

    assert perfect_ece == pytest.approx(
        0.0
    )

    assert imperfect_ece == pytest.approx(
        0.2
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([np.nan, 0.7]),
        np.array([-0.1, 0.9]),
        np.array([0.1, 1.2]),
    ],
)
def test_ece_rejects_invalid_probabilities(
    probabilities,
):

    with pytest.raises(
        ValueError,
        match="Invalid probabilities",
    ):

        calibration.expected_calibration_error(
            np.array([0, 1]),
            probabilities,
        )


def test_ece_rejects_invalid_shape_or_bins():

    with pytest.raises(
        ValueError,
        match="bins must be positive",
    ):

        calibration.expected_calibration_error(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            bins=0,
        )

    with pytest.raises(
        ValueError,
        match="Invalid calibration error inputs",
    ):

        calibration.expected_calibration_error(
            np.array([0, 1]),
            np.array([0.2]),
        )


# ============================================================
# 9. CLASSIFICATION METRICS
# ============================================================

def test_metrics_for_perfect_predictions():

    labels = np.array([
        0, 0, 1, 1,
    ])

    probabilities = np.array([
        0.1,
        0.2,
        0.8,
        0.9,
    ])

    metrics = calibration.metrics_for(
        labels,
        probabilities,
    )

    assert metrics["f1"] == pytest.approx(
        1.0
    )

    assert metrics["recall"] == pytest.approx(
        1.0
    )

    assert metrics["fpr"] == pytest.approx(
        0.0
    )

    assert metrics["false_positives"] == 0

    assert metrics["false_negatives"] == 0

    assert metrics["brier"] < 0.05


def test_metrics_for_false_positives_and_negatives():

    labels = np.array([
        0, 0, 1, 1,
    ])

    probabilities = np.array([
        0.9,
        0.1,
        0.2,
        0.9,
    ])

    metrics = calibration.metrics_for(
        labels,
        probabilities,
    )

    assert metrics["false_positives"] == 1

    assert metrics["false_negatives"] == 1

    assert metrics["fpr"] == pytest.approx(
        0.5
    )

    assert metrics["recall"] == pytest.approx(
        0.5
    )


@pytest.mark.parametrize(
    "labels, probabilities, error",
    [
        (
            np.array([0, 0]),
            np.array([0.1, 0.2]),
            "both classes",
        ),
        (
            np.array([0, 1]),
            np.array([0.4]),
            "Invalid evaluation probabilities",
        ),
        (
            np.array([0, 1]),
            np.array([0.1, np.nan]),
            "Invalid evaluation probabilities",
        ),
        (
            np.array([0, 1]),
            np.array([-0.1, 0.7]),
            "Probabilities must lie",
        ),
    ],
)
def test_metrics_for_rejects_invalid_data(
    labels,
    probabilities,
    error,
):

    with pytest.raises(
        ValueError,
        match=error,
    ):

        calibration.metrics_for(
            labels,
            probabilities,
        )


# ============================================================
# 10. CROSS-SOURCE EXPERIMENT
# ============================================================

def test_experiment_produces_24_rows_without_real_training(
    synthetic_results,
):

    assert len(
        synthetic_results
    ) == 24

    unique_rows = synthetic_results.drop_duplicates(
        [
            "feature_set",
            "method",
            "evaluation_source",
        ]
    )

    assert len(unique_rows) == 24

    assert set(
        synthetic_results["evaluation_source"]
    ) == set(calibration.SOURCES)

    assert set(
        synthetic_results["method"]
    ) == set(calibration.METHODS)

    assert set(
        synthetic_results["feature_set"]
    ) == set(calibration.CANDIDATES)

    assert (
        synthetic_results["fit_samples"].min()
        > 0
    )

    assert (
        synthetic_results["calibration_samples"].min()
        > 0
    )


def test_experiment_rejects_missing_parameters(
    synthetic_data,
):

    with pytest.raises(
        ValueError,
        match="Missing optimized parameters",
    ):

        calibration.run_experiment(
            synthetic_data,
            {},
        )


# ============================================================
# 11. GENERATE REPORTS
# ============================================================

def test_make_reports_computes_six_summary_rows(
    synthetic_results,
):

    summary, comparison = (
        calibration.make_reports(
            synthetic_results
        )
    )

    assert len(summary) == 6

    assert len(comparison) == 6

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    none_rows = comparison.loc[
        comparison["method"] == "none"
    ]

    assert (
        none_rows["delta_macro_brier"] == 0
    ).all()

    assert (
        none_rows["delta_macro_fpr"] == 0
    ).all()

    assert set(
        comparison["method"]
    ) == set(calibration.METHODS)


def test_make_reports_rejects_empty_or_incomplete(
    synthetic_results,
):

    with pytest.raises(
        ValueError,
        match="Empty calibration results",
    ):

        calibration.make_reports(
            pd.DataFrame()
        )

    incomplete_results = (
        synthetic_results.iloc[:-1]
    )

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected calibration evaluations",
    ):

        calibration.make_reports(
            incomplete_results
        )


def test_make_reports_rejects_duplicate_rows(
    synthetic_results,
):

    duplicated_results = pd.concat(
        [
            synthetic_results,
            synthetic_results.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected calibration evaluations",
    ):

        calibration.make_reports(
            duplicated_results
        )


# ============================================================
# 12. COMPLETE WORKFLOW
# ============================================================

def test_main_writes_only_three_reports(
    tmp_path,
    monkeypatch,
    synthetic_data,
    best_parameters,
    synthetic_results,
):

    # Use temporary directory for generated reports.

    monkeypatch.setattr(
        calibration,
        "REPORT_DIR",
        tmp_path,
    )

    # Prevent loading real datasets.

    monkeypatch.setattr(
        calibration,
        "load_training_data",
        lambda: synthetic_data,
    )

    # Prevent reading real hyperparameter reports.

    monkeypatch.setattr(
        calibration,
        "load_best_parameters",
        lambda: best_parameters,
    )

    # Reuse synthetic experiment results.

    monkeypatch.setattr(
        calibration,
        "run_experiment",
        lambda dataframe, parameters: synthetic_results,
    )

    calibration.main()

    expected_files = {
        "calibration_source_results.csv": 24,
        "calibration_summary.csv": 6,
        "calibration_comparison.csv": 6,
    }

    generated_files = list(
        tmp_path.glob("*.csv")
    )

    assert len(generated_files) == 3

    for filename, expected_count in expected_files.items():

        file_path = tmp_path / filename

        assert file_path.is_file()

        dataframe = pd.read_csv(
            file_path
        )

        assert len(dataframe) == expected_count