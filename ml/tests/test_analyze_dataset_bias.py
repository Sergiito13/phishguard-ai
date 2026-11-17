"""
PhishGuard AI - Dataset Bias Analysis Tests

Tests:
- Source distributions.
- Feature distribution shifts.
- False positive analysis.
- Permutation feature importance.
- Cross-source model evaluation.
- Report generation.

All tests use synthetic data and simulated models.

No real datasets are modified.
No trained model artifacts are overwritten.
The test dataset is never accessed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import analyze_dataset_bias as bias

from extract_features import FEATURE_COLUMNS


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

def make_record(
    label: int,
    sources: str,
    group_key: str,
    **features,
) -> dict:
    """
    Create a synthetic URL feature record.
    """

    record = {
        "label": label,
        "sources": sources,
        "group_key": group_key,
    }

    record.update({
        feature: 0.0
        for feature in FEATURE_COLUMNS
    })

    record.update(features)

    return record


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame([
        make_record(
            0,
            "legitphish",
            "group-a",
            is_https=1,
            url_length=20,
            path_length=0,
        ),
        make_record(
            1,
            "legitphish",
            "group-b",
            is_https=0,
            url_length=50,
            path_length=15,
        ),
        make_record(
            0,
            "url_phish",
            "group-c",
            is_https=0,
            url_length=30,
            path_length=5,
        ),
        make_record(
            0,
            "url_phish",
            "group-d",
            is_https=1,
            url_length=40,
            path_length=10,
        ),
        make_record(
            1,
            "url_phish|phiusiil",
            "group-e",
            is_https=1,
            url_length=60,
            path_length=20,
        ),
    ])


# ============================================================
# 3. SOURCE DISTRIBUTION ANALYSIS
# ============================================================

def test_analyze_source_distributions(sample_dataframe):

    results = bias.analyze_source_distributions(
        sample_dataframe
    )

    assert not results.empty

    target_legitimate = results.loc[
        (results["source"] == "url_phish")
        & (results["label"] == 0)
    ].iloc[0]

    assert target_legitimate["samples"] == 2

    assert target_legitimate["https_rate"] == pytest.approx(
        0.5
    )

    assert target_legitimate["url_length_mean"] == pytest.approx(
        35.0
    )

    # URLs associated with multiple sources must
    # contribute to each source summary.

    target_phishing = results.loc[
        (results["source"] == "url_phish")
        & (results["label"] == 1)
    ].iloc[0]

    phiusiil_phishing = results.loc[
        (results["source"] == "phiusiil")
        & (results["label"] == 1)
    ].iloc[0]

    assert target_phishing["samples"] == 1

    assert phiusiil_phishing["samples"] == 1


# ============================================================
# 4. FEATURE DISTRIBUTION SHIFT
# ============================================================

def test_analyze_feature_shift(sample_dataframe):

    results = bias.analyze_feature_shift(
        sample_dataframe
    )

    assert len(results) == len(FEATURE_COLUMNS) * 2

    assert set(results["label"]) == {0, 1}

    assert set(results["feature"]) == set(
        FEATURE_COLUMNS
    )

    https_shift = results.loc[
        (results["label"] == 0)
        & (results["feature"] == "is_https")
    ].iloc[0]

    assert https_shift["target_mean"] == pytest.approx(
        0.5
    )

    assert https_shift["other_mean"] == pytest.approx(
        1.0
    )

    assert https_shift["mean_difference"] == pytest.approx(
        -0.5
    )

    assert https_shift["standardized_difference"] < 0

    assert https_shift["absolute_shift"] == pytest.approx(
        abs(https_shift["standardized_difference"])
    )

    # Constant features must not cause division
    # by zero.

    constant_feature = results.loc[
        results["feature"] == "has_punycode"
    ]

    assert (
        constant_feature["standardized_difference"] == 0
    ).all()

    # Verify descending shift ranking within each label.

    for label in [0, 1]:

        shifts = results.loc[
            results["label"] == label,
            "absolute_shift",
        ]

        assert shifts.is_monotonic_decreasing


def test_feature_shift_requires_comparison_samples():

    dataframe = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "group-a",
        ),
        make_record(
            1,
            "url_phish",
            "group-b",
        ),
    ])

    with pytest.raises(
        ValueError,
        match="Missing comparison samples",
    ):

        bias.analyze_feature_shift(
            dataframe
        )


# ============================================================
# 5. FALSE POSITIVE ANALYSIS
# ============================================================

def test_analyze_false_positives():

    dataframe = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "group-a",
            url_length=40,
            is_https=0,
        ),
        make_record(
            0,
            "url_phish",
            "group-b",
            url_length=20,
            is_https=1,
        ),
        make_record(
            1,
            "url_phish",
            "group-c",
            url_length=100,
            is_https=1,
        ),
        make_record(
            1,
            "url_phish",
            "group-d",
            url_length=90,
            is_https=0,
        ),
    ])

    predictions = np.array([
        1,  # False positive
        0,  # True negative
        1,  # True positive
        0,  # False negative
    ])

    results = bias.analyze_false_positives(
        dataframe,
        predictions,
        "baseline",
    )

    assert len(results) == len(FEATURE_COLUMNS)

    assert (
        results["experiment"] == "baseline"
    ).all()

    url_length = results.loc[
        results["feature"] == "url_length"
    ].iloc[0]

    assert url_length["false_positive_mean"] == pytest.approx(
        40.0
    )

    assert url_length["true_negative_mean"] == pytest.approx(
        20.0
    )

    assert url_length["mean_difference"] == pytest.approx(
        20.0
    )

    assert url_length["absolute_difference"] == pytest.approx(
        20.0
    )

    assert results[
        "absolute_difference"
    ].is_monotonic_decreasing

    # The original dataframe must remain unchanged.

    assert "prediction" not in dataframe.columns


def test_false_positive_analysis_without_errors():

    dataframe = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "group-a",
            url_length=20,
        ),
        make_record(
            1,
            "url_phish",
            "group-b",
            url_length=50,
        ),
    ])

    predictions = np.array([
        0,
        1,
    ])

    results = bias.analyze_false_positives(
        dataframe,
        predictions,
        "baseline",
    )

    # There are no false positives.

    assert results[
        "false_positive_mean"
    ].isna().all()

    # True negative statistics are still available.

    url_length = results.loc[
        results["feature"] == "url_length"
    ].iloc[0]

    assert url_length["true_negative_mean"] == pytest.approx(
        20.0
    )


def test_false_positive_prediction_length_mismatch():

    dataframe = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "group-a",
        ),
        make_record(
            1,
            "url_phish",
            "group-b",
        ),
    ])

    predictions = np.array([0])

    with pytest.raises(ValueError):

        bias.analyze_false_positives(
            dataframe,
            predictions,
            "baseline",
        )


# ============================================================
# 6. PERMUTATION FEATURE IMPORTANCE
# ============================================================

def test_analyze_feature_importance(
    sample_dataframe,
    monkeypatch,
):

    features = [
        "url_length",
        "is_https",
        "path_length",
    ]

    model = object()

    def fake_permutation_importance(
        estimator,
        X,
        y,
        scoring,
        n_repeats,
        random_state,
        n_jobs,
    ):

        assert estimator is model

        assert X.shape == (
            len(sample_dataframe),
            len(features),
        )

        assert set(y) == {0, 1}

        assert scoring == "roc_auc"

        assert n_repeats == bias.PERMUTATION_REPEATS

        assert random_state == bias.RANDOM_STATE

        return SimpleNamespace(
            importances_mean=np.array([
                0.10,
                -0.15,
                0.20,
            ]),
            importances_std=np.array([
                0.01,
                0.02,
                0.03,
            ]),
        )

    monkeypatch.setattr(
        bias,
        "permutation_importance",
        fake_permutation_importance,
    )

    results = bias.analyze_feature_importance(
        model,
        sample_dataframe,
        features,
        "baseline",
    )

    assert len(results) == 3

    assert results["feature"].tolist() == [
        "path_length",
        "url_length",
        "is_https",
    ]

    assert results["importance_mean"].tolist() == pytest.approx([
        0.20,
        0.10,
        -0.15,
    ])

    assert results["importance_std"].tolist() == pytest.approx([
        0.03,
        0.01,
        0.02,
    ])

    assert results[
        "importance_mean"
    ].is_monotonic_decreasing


def test_feature_importance_requires_both_classes():

    dataframe = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "group-a",
        ),
        make_record(
            0,
            "url_phish",
            "group-b",
        ),
    ])

    with pytest.raises(
        ValueError,
        match="Permutation analysis requires both classes",
    ):

        bias.analyze_feature_importance(
            object(),
            dataframe,
            list(FEATURE_COLUMNS),
            "baseline",
        )


# ============================================================
# 7. CROSS-SOURCE MODEL EVALUATION
# ============================================================

def test_evaluate_models(monkeypatch):

    trained_models = []

    class FakeModel:

        def fit(self, X, y):

            # Only the two samples from non-target
            # sources should be used for training.

            assert len(X) == 2

            assert set(y) == {0, 1}

            # URL_PHISH training records have a
            # synthetic url_length greater than 900.

            assert np.all(X[:, 0] < 900)

            self.feature_count = X.shape[1]

            trained_models.append(self)

            return self

        def predict_proba(self, X):

            assert len(X) == 2

            # Only URL_PHISH validation samples
            # should reach the model.

            assert np.all(X[:, 0] >= 100)

            return np.array([
                [0.90, 0.10],
                [0.10, 0.90],
            ])

    monkeypatch.setattr(
        bias,
        "create_models",
        lambda: {
            "HistGradientBoosting": FakeModel()
        },
    )

    def fake_analyze_feature_importance(
        model,
        validation_df,
        features,
        experiment_name,
    ):

        assert len(validation_df) == 2

        assert model.feature_count == len(features)

        return pd.DataFrame({
            "experiment": experiment_name,
            "feature": features,
            "importance_mean": [
                0.1
            ] * len(features),
            "importance_std": [
                0.0
            ] * len(features),
        })

    monkeypatch.setattr(
        bias,
        "analyze_feature_importance",
        fake_analyze_feature_importance,
    )

    train_df = pd.DataFrame([
        make_record(
            0,
            "legitphish",
            "train-a",
            url_length=20,
        ),
        make_record(
            1,
            "phiusiil",
            "train-b",
            url_length=50,
        ),
        make_record(
            0,
            "url_phish",
            "train-c",
            url_length=999,
        ),
        make_record(
            1,
            "legitphish|url_phish",
            "train-d",
            url_length=999,
        ),
    ])

    validation_df = pd.DataFrame([
        make_record(
            0,
            "url_phish",
            "validation-a",
            url_length=100,
        ),
        make_record(
            1,
            "url_phish",
            "validation-b",
            url_length=200,
        ),
        make_record(
            0,
            "legitphish",
            "validation-c",
            url_length=30,
        ),
    ])

    (
        models_df,
        errors_df,
        importance_df,
    ) = bias.evaluate_models(
        train_df,
        validation_df,
    )

    assert len(trained_models) == 2

    assert len(models_df) == 2

    assert set(models_df["experiment"]) == {
        "baseline",
        "without_https",
    }

    assert (
        models_df["accuracy"] == 1.0
    ).all()

    assert (
        models_df["roc_auc"] == 1.0
    ).all()

    assert (
        models_df["false_positives"] == 0
    ).all()

    assert (
        models_df["false_negatives"] == 0
    ).all()

    baseline = models_df.loc[
        models_df["experiment"] == "baseline"
    ].iloc[0]

    without_https = models_df.loc[
        models_df["experiment"] == "without_https"
    ].iloc[0]

    assert baseline["features"] == len(
        FEATURE_COLUMNS
    )

    assert without_https["features"] == (
        len(FEATURE_COLUMNS) - 1
    )

    assert len(errors_df) == (
        len(FEATURE_COLUMNS) * 2
    )

    assert len(importance_df) == (
        len(FEATURE_COLUMNS) * 2 - 1
    )

    without_https_importance = importance_df.loc[
        importance_df["experiment"] == "without_https"
    ]

    assert "is_https" not in without_https_importance[
        "feature"
    ].values


def test_evaluate_models_rejects_empty_source():

    train_df = pd.DataFrame([
        make_record(
            0,
            "legitphish",
            "train-a",
        ),
        make_record(
            1,
            "phiusiil",
            "train-b",
        ),
    ])

    validation_df = pd.DataFrame([
        make_record(
            0,
            "phishbd",
            "validation-a",
        ),
    ])

    with pytest.raises(
        ValueError,
        match="Training or validation data is empty",
    ):

        bias.evaluate_models(
            train_df,
            validation_df,
        )


# ============================================================
# 8. REPORT GENERATION
# ============================================================

def test_save_reports(tmp_path, monkeypatch):

    monkeypatch.setattr(
        bias,
        "REPORT_DIR",
        tmp_path,
    )

    source_df = pd.DataFrame({
        "source": ["url_phish"],
        "label": [0],
        "samples": [100],
    })

    shift_df = pd.DataFrame({
        "feature": ["is_https"],
        "absolute_shift": [1.5],
    })

    models_df = pd.DataFrame({
        "experiment": ["baseline"],
        "roc_auc": [0.75],
    })

    errors_df = pd.DataFrame({
        "feature": ["url_length"],
        "mean_difference": [10.0],
    })

    importance_df = pd.DataFrame({
        "feature": ["is_https"],
        "importance_mean": [-0.15],
    })

    bias.save_reports(
        source_df,
        shift_df,
        models_df,
        errors_df,
        importance_df,
    )

    expected_reports = {
        "dataset_bias_sources.csv": source_df,
        "dataset_bias_feature_shift.csv": shift_df,
        "dataset_bias_models.csv": models_df,
        "dataset_bias_false_positives.csv": errors_df,
        "dataset_bias_feature_importance.csv": importance_df,
    }

    assert len(list(tmp_path.glob("*.csv"))) == 5

    for filename, expected_df in expected_reports.items():

        report_file = tmp_path / filename

        assert report_file.is_file()

        loaded_df = pd.read_csv(report_file)

        pd.testing.assert_frame_equal(
            loaded_df,
            expected_df,
        )