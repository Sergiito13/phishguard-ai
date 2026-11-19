"""
Unit tests for training-only cross-source error diagnostics.

Uses synthetic data and fake classifiers.

Does not read the real datasets, final test split,
or saved model artifacts.
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
    Path(__file__).resolve().parents[1] / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import analyze_cross_source_errors as diagnostic


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def synthetic_data():

    feature_names = set().union(
        *(
            set(
                diagnostic.FEATURE_SETS[name]
            )
            for name in diagnostic.CANDIDATES
        )
    )

    records = []

    for source in diagnostic.SOURCES:

        for label in (0, 1):

            for index in range(6):

                score = (
                    0.85
                    if label
                    else 0.15
                )

                # Force one classification error
                # per class and source.

                if index == 0:
                    score = 1.0 - score

                row = {
                    "url": (
                        f"https://{source}-{label}-{index}"
                        ".example/path"
                    ),

                    "sources": source,

                    "group_key": (
                        f"domain:{source}-{label}-{index}"
                        ".example"
                    ),

                    "label": label,
                }

                row.update({
                    feature: score
                    for feature in feature_names
                })

                records.append(
                    row
                )

    return pd.DataFrame(
        records
    )


# ============================================================
# 3. FAKE CLASSIFIER
# ============================================================

class FakeModel:

    fits = 0

    def fit(self, X, y):

        assert set(
            np.unique(y)
        ) == {0, 1}

        type(self).fits += 1

        return self

    def predict_proba(self, X):

        score = np.clip(
            X[:, 0],
            0.01,
            0.99,
        )

        return np.column_stack(
            (
                1.0 - score,
                score,
            )
        )


# ============================================================
# 4. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def diagnostic_results(
    monkeypatch,
    synthetic_data,
):

    FakeModel.fits = 0

    monkeypatch.setattr(
        diagnostic,
        "create_model",
        lambda params: FakeModel(),
    )

    parameters = {
        name: {}
        for name in diagnostic.CANDIDATES
    }

    result = diagnostic.run_experiment(
        synthetic_data,
        parameters,
    )

    assert FakeModel.fits == 8

    return result


# ============================================================
# 5. CONFIGURATION
# ============================================================

def test_config():

    assert len(
        diagnostic.CANDIDATES
    ) == 2

    assert len(
        diagnostic.SOURCES
    ) == 4

    assert diagnostic.THRESHOLD == 0.5


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def test_perfect_confusion_summary():

    result = diagnostic.confusion_summary(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )

    assert result["fpr"] == 0.0

    assert result["recall"] == 1.0

    assert result["false_positives"] == 0

    assert result["phishing_score_median"] == (
        pytest.approx(0.85)
    )


def test_confusion_mistakes():

    result = diagnostic.confusion_summary(
        [0, 0, 1, 1],
        [0.8, 0.1, 0.2, 0.9],
    )

    assert result["false_positives"] == 1

    assert result["false_negatives"] == 1

    assert result["fpr"] == pytest.approx(
        0.5
    )

    assert result["recall"] == pytest.approx(
        0.5
    )


# ============================================================
# 7. INVALID PREDICTION SCORES
# ============================================================

@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),

        ([0, 0], [0.1, 0.2]),

        ([0, 1], [0.1]),

        ([0, 1], [0.1, np.nan]),

        ([0, 1], [-0.1, 0.9]),

        ([0, 1], [0.2, 1.1]),

        ([0, 1], [[0.1], [0.8]]),
    ],
)
def test_invalid_scores(
    labels,
    scores,
):

    with pytest.raises(
        ValueError
    ):
        diagnostic.validate_scores(
            labels,
            scores,
        )


# ============================================================
# 8. SCORE DISTRIBUTION
# ============================================================

def test_score_distribution_includes_one():

    rows = diagnostic.score_distribution(
        [0, 0, 1, 1],
        [0, 0.2, 0.8, 1.0],
    )

    assert len(rows) == 10

    counts = (
        rows.groupby("label")["count"]
        .sum()
        .to_dict()
    )

    assert counts == {
        0: 2,
        1: 2,
    }

    fractions = (
        rows.groupby("label")["class_fraction"]
        .sum()
        .to_dict()
    )

    assert fractions == pytest.approx({
        0: 1,
        1: 1,
    })


# ============================================================
# 9. FEATURE COMPARISON
# ============================================================

def test_compare_values_difference():

    result = diagnostic.compare_values(
        [0.8, 1.0],
        [0.1, 0.2],
    )

    assert result["standardized_difference"] > 0

    assert result["target_samples"] == 2

    assert result["reference_samples"] == 2


def test_compare_values_empty_returns_nan():

    result = diagnostic.compare_values(
        [],
        [0.4, 0.7],
    )

    assert np.isnan(
        result["standardized_difference"]
    )

    assert result["target_samples"] == 0


def test_compare_values_constant_groups():

    equal = diagnostic.compare_values(
        [0.5] * 3,
        [0.5] * 3,
    )

    different = diagnostic.compare_values(
        [0] * 3,
        [1] * 3,
    )

    assert (
        equal["standardized_difference"]
        == 0.0
    )

    assert different["constant_nonoverlap"]

    assert np.isnan(
        different["standardized_difference"]
    )


def test_compare_values_nonfinite_rejected():

    with pytest.raises(
        ValueError,
        match="Non-finite",
    ):
        diagnostic.compare_values(
            [np.nan],
            [0.1],
        )


# ============================================================
# 10. FEATURE DIAGNOSTICS
# ============================================================

def test_feature_diagnostics_compare_errors(
    synthetic_data,
):

    source = diagnostic.SOURCES[0]

    held = synthetic_data.loc[
        synthetic_data["sources"] == source
    ]

    fit = synthetic_data.loc[
        synthetic_data["sources"] != source
    ]

    feature = diagnostic.FEATURE_SETS[
        diagnostic.CANDIDATES[0]
    ][0]

    labels = held[
        "label"
    ].to_numpy()

    scores = held[
        feature
    ].to_numpy()

    shifts, errors = diagnostic.feature_diagnostics(
        fit,
        held,
        labels,
        scores,
        [feature],
    )

    assert len(shifts) == 2

    assert len(errors) == 2

    fp = errors.loc[
        errors["error_type"] == "FP"
    ].iloc[0]

    fn = errors.loc[
        errors["error_type"] == "FN"
    ].iloc[0]

    assert fp["target_samples"] == 1

    assert fp["reference_samples"] == 5

    assert fn["target_samples"] == 1

    assert fn["reference_samples"] == 5


def test_feature_diagnostics_perfect_predictions_no_errors(
    synthetic_data,
):

    held = synthetic_data.iloc[:12]

    fit = synthetic_data.iloc[12:]

    feature = diagnostic.FEATURE_SETS[
        diagnostic.CANDIDATES[0]
    ][0]

    scores = np.where(
        held["label"].to_numpy() == 1,
        0.9,
        0.1,
    )

    _, errors = diagnostic.feature_diagnostics(
        fit,
        held,
        held["label"].to_numpy(),
        scores,
        [feature],
    )

    assert errors[
        "target_samples"
    ].eq(0).all()

    assert errors[
        "standardized_difference"
    ].isna().all()


# ============================================================
# 11. EXPERIMENT RESULTS
# ============================================================

def test_run_experiment_outputs_four_tables(
    diagnostic_results,
):

    assert set(diagnostic_results) == {
        "summary",
        "score_distribution",
        "feature_shift",
        "error_profiles",
    }

    assert len(
        diagnostic_results["summary"]
    ) == 8

    assert len(
        diagnostic_results["score_distribution"]
    ) == 80

    assert (
        diagnostic_results["summary"]["false_positives"]
        == 1
    ).all()

    assert (
        diagnostic_results["summary"]["false_negatives"]
        == 1
    ).all()

    for value in diagnostic_results.values():

        assert {
            "url",
            "group_key",
            "sources",
        }.isdisjoint(
            value.columns
        )


# ============================================================
# 12. INVALID EXPERIMENT CONFIGURATION
# ============================================================

def test_missing_optimized_parameters(
    synthetic_data,
):

    with pytest.raises(
        ValueError,
        match="Missing optimized",
    ):
        diagnostic.run_experiment(
            synthetic_data,
            {},
        )


def test_group_leakage_is_rejected(
    monkeypatch,
    synthetic_data,
):

    def bad_folds(dataframe):

        return [
            {
                "source": source,

                "training_indices": np.arange(
                    len(dataframe)
                ),

                "validation_indices": np.flatnonzero(
                    dataframe["sources"].eq(source)
                ),
            }

            for source in diagnostic.SOURCES
        ]

    monkeypatch.setattr(
        diagnostic,
        "create_source_folds",
        bad_folds,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage",
    ):
        diagnostic.run_experiment(
            synthetic_data,
            {
                name: {}
                for name in diagnostic.CANDIDATES
            },
        )


def test_source_leakage_is_rejected(
    monkeypatch,
    synthetic_data,
):

    def bad_source(
        dataframe,
        source,
    ):
        return pd.Series(
            True,
            index=dataframe.index,
        )

    monkeypatch.setattr(
        diagnostic,
        "belongs_to_source",
        bad_source,
    )

    with pytest.raises(
        ValueError,
        match="Source leakage",
    ):
        diagnostic.run_experiment(
            synthetic_data,
            {
                name: {}
                for name in diagnostic.CANDIDATES
            },
        )


# ============================================================
# 13. REPORT GENERATION
# ============================================================

def test_save_reports(
    tmp_path,
    monkeypatch,
    diagnostic_results,
):

    monkeypatch.setattr(
        diagnostic,
        "REPORT_DIR",
        tmp_path,
    )

    diagnostic.save_reports(
        diagnostic_results
    )

    files = list(
        tmp_path.glob(
            "cross_source_errors_*.csv"
        )
    )

    assert len(files) == 4

    for file in files:

        dataframe = pd.read_csv(
            file
        )

        assert not dataframe.empty

        assert {
            "url",
            "group_key",
            "sources",
        }.isdisjoint(
            dataframe.columns
        )


def test_save_rejects_private_columns(
    tmp_path,
    monkeypatch,
    diagnostic_results,
):

    monkeypatch.setattr(
        diagnostic,
        "REPORT_DIR",
        tmp_path,
    )

    diagnostic_results["summary"]["url"] = (
        "https://private.example"
    )

    with pytest.raises(
        ValueError,
        match="expose raw",
    ):
        diagnostic.save_reports(
            diagnostic_results
        )


# ============================================================
# 14. COMPLETE WORKFLOW
# ============================================================

def test_main_without_real_data(
    monkeypatch,
    tmp_path,
    synthetic_data,
    diagnostic_results,
    capsys,
):

    monkeypatch.setattr(
        diagnostic,
        "REPORT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        diagnostic,
        "load_training_data",
        lambda: synthetic_data,
    )

    monkeypatch.setattr(
        diagnostic,
        "load_best_parameters",
        lambda: {
            name: {}
            for name in diagnostic.CANDIDATES
        },
    )

    monkeypatch.setattr(
        diagnostic,
        "run_experiment",
        lambda dataframe, params: diagnostic_results,
    )

    diagnostic.main()

    assert (
        "SOURCE SUMMARY"
        in capsys.readouterr().out
    )

    files = list(
        tmp_path.glob(
            "cross_source_errors_*.csv"
        )
    )

    assert len(files) == 4