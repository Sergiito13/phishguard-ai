"""
PhishGuard AI - Hyperparameter Optimization Tests

Tests use synthetic data and mocked training.

No real datasets or model artifacts are modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import experiment_hyperparameters as hp


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_dataframe():

    records = []

    features = set()

    for feature_set in hp.FEATURE_SETS.values():
        features.update(feature_set)

    for source in hp.SOURCES:

        for label in [0, 1]:

            record = {
                "url": f"https://{source}-{label}.example",
                "label": label,
                "sources": source,
                "group_key": f"group-{source}-{label}",
                "domain_status": "domain",
            }

            record.update({
                feature: float(label + 1)
                for feature in features
            })

            records.append(record)

    return pd.DataFrame(records)


@pytest.fixture
def fake_metrics():

    return {
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.90,
        "f1": 0.87,
        "roc_auc": 0.92,
        "average_precision": 0.93,
        "false_positives": 1,
        "false_negatives": 1,
        "fpr": 0.10,
    }


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

def test_feature_sets():

    assert len(hp.FEATURE_SETS) == 3

    assert len(
        hp.FEATURE_SETS["baseline_v1"]
    ) == 20

    assert len(
        hp.FEATURE_SETS["hybrid_v1_extended"]
    ) == 32

    assert len(
        hp.FEATURE_SETS["extended_v2"]
    ) == 31

    for features in hp.FEATURE_SETS.values():

        assert len(features) == len(set(features))


# ============================================================
# PARAMETER GENERATION
# ============================================================

def test_generate_parameters():

    candidates = hp.generate_parameters(6)

    assert len(candidates) == 6

    assert candidates[0] == hp.BASELINE_PARAMS

    assert len({
        tuple(sorted(params.items()))
        for params in candidates
    }) == 6


def test_generate_parameters_reproducibility():

    first = hp.generate_parameters(6)

    second = hp.generate_parameters(6)

    assert first == second


def test_generate_single_parameter():

    candidates = hp.generate_parameters(1)

    assert candidates == [
        hp.BASELINE_PARAMS
    ]


@pytest.mark.parametrize(
    "trials",
    [0, -1, -10],
)
def test_invalid_trial_count(trials):

    with pytest.raises(
        ValueError,
        match="Trials must be greater than zero",
    ):

        hp.generate_parameters(trials)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def test_create_model():

    model = hp.create_model(
        hp.BASELINE_PARAMS
    )

    params = model.get_params()

    for name, value in hp.BASELINE_PARAMS.items():

        assert params[name] == value

    assert params["random_state"] == 42


# ============================================================
# TRAINING DATA
# ============================================================

def test_load_training_data(
    monkeypatch,
    sample_dataframe,
):

    calls = []

    def fake_load_dataset(original, v2):

        calls.append((original, v2))

        return sample_dataframe

    monkeypatch.setattr(
        hp,
        "load_dataset",
        fake_load_dataset,
    )

    result = hp.load_training_data()

    assert len(result) == 8

    assert calls == [
        (
            hp.TRAIN_FILE,
            hp.V2_TRAIN_FILE,
        )
    ]


def test_load_training_data_missing_features(
    monkeypatch,
    sample_dataframe,
):

    dataframe = sample_dataframe.drop(
        columns=["v2_hostname_entropy"]
    )

    monkeypatch.setattr(
        hp,
        "load_dataset",
        lambda original, v2: dataframe,
    )

    with pytest.raises(
        ValueError,
        match="Missing features",
    ):

        hp.load_training_data()


# ============================================================
# SOURCE FOLDS
# ============================================================

def test_create_source_folds(sample_dataframe):

    folds = hp.create_source_folds(
        sample_dataframe
    )

    assert len(folds) == 4

    for fold in folds:

        train_idx = fold["training_indices"]

        val_idx = fold["validation_indices"]

        assert len(train_idx) == 6

        assert len(val_idx) == 2

        assert set(train_idx).isdisjoint(val_idx)

        train_df = sample_dataframe.iloc[train_idx]

        val_df = sample_dataframe.iloc[val_idx]

        assert not hp.belongs_to_source(
            train_df,
            fold["source"],
        ).any()

        assert hp.belongs_to_source(
            val_df,
            fold["source"],
        ).all()

        assert set(
            train_df["group_key"]
        ).isdisjoint(
            val_df["group_key"]
        )


def test_multisource_leakage_prevention(
    sample_dataframe,
):

    dataframe = sample_dataframe.copy()

    shared = dataframe.iloc[[0]].copy()

    shared["url"] = "https://shared.example"

    shared["sources"] = "legitphish|url_phish"

    shared["group_key"] = "shared-group"

    dataframe = pd.concat(
        [dataframe, shared],
        ignore_index=True,
    )

    folds = hp.create_source_folds(dataframe)

    for fold in folds:

        if fold["source"] not in [
            "legitphish",
            "url_phish",
        ]:
            continue

        training = dataframe.iloc[
            fold["training_indices"]
        ]

        assert "https://shared.example" not in (
            training["url"].values
        )


def test_group_leakage_prevention(
    sample_dataframe,
):

    dataframe = sample_dataframe.copy()

    shared = dataframe.iloc[[0]].copy()

    shared["url"] = "https://another-source.example"

    shared["sources"] = "phiusiil"

    # Same domain group as a LegitPhish URL.

    dataframe = pd.concat(
        [dataframe, shared],
        ignore_index=True,
    )

    folds = hp.create_source_folds(dataframe)

    legitphish_fold = next(
        fold
        for fold in folds
        if fold["source"] == "legitphish"
    )

    training = dataframe.iloc[
        legitphish_fold["training_indices"]
    ]

    assert "https://another-source.example" not in (
        training["url"].values
    )


def test_missing_source_fold(
    sample_dataframe,
):

    dataframe = sample_dataframe.loc[
        sample_dataframe["sources"] != "url_phish"
    ]

    with pytest.raises(
        ValueError,
        match="Empty validation fold",
    ):

        hp.create_source_folds(dataframe)


def test_source_fold_requires_two_classes(
    sample_dataframe,
):

    dataframe = sample_dataframe.copy()

    dataframe["label"] = 0

    with pytest.raises(
        ValueError,
        match="requires both classes",
    ):

        hp.create_source_folds(dataframe)


# ============================================================
# FOLD EVALUATION
# ============================================================

def test_evaluate_fold(monkeypatch):

    class FakeModel:

        def fit(self, X, y):

            self.trained = True

            return self

        def predict_proba(self, X):

            probabilities = X[:, 0]

            return np.column_stack([
                1 - probabilities,
                probabilities,
            ])

    monkeypatch.setattr(
        hp,
        "create_model",
        lambda params: FakeModel(),
    )

    X = np.array([
        [0.0],
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ])

    y = np.array([
        0, 1, 0, 1, 0, 1
    ])

    metrics = hp.evaluate_fold(
        X=X,
        y=y,
        training_indices=np.array([0, 1, 2, 3]),
        validation_indices=np.array([4, 5]),
        params=hp.BASELINE_PARAMS,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)

    assert metrics["recall"] == pytest.approx(1.0)

    assert metrics["roc_auc"] == pytest.approx(1.0)

    assert metrics["fpr"] == pytest.approx(0.0)


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def test_run_search(
    monkeypatch,
    sample_dataframe,
    fake_metrics,
):

    monkeypatch.setattr(
        hp,
        "generate_parameters",
        lambda trials: [hp.BASELINE_PARAMS],
    )

    calls = []

    def fake_evaluate_fold(
        X,
        y,
        training_indices,
        validation_indices,
        params,
    ):

        calls.append(1)

        return fake_metrics.copy()

    monkeypatch.setattr(
        hp,
        "evaluate_fold",
        fake_evaluate_fold,
    )

    results = hp.run_search(
        sample_dataframe,
        trials=1,
    )

    assert len(results) == 12

    assert len(calls) == 12

    assert results["feature_set"].nunique() == 3

    assert results["evaluation_source"].nunique() == 4

    assert (results["trial_id"] == 0).all()


# ============================================================
# SUMMARY AND REPORTS
# ============================================================

@pytest.fixture
def sample_results(
    monkeypatch,
    sample_dataframe,
    fake_metrics,
):

    monkeypatch.setattr(
        hp,
        "generate_parameters",
        lambda trials: [hp.BASELINE_PARAMS],
    )

    monkeypatch.setattr(
        hp,
        "evaluate_fold",
        lambda **kwargs: fake_metrics.copy(),
    )

    return hp.run_search(
        sample_dataframe,
        trials=1,
    )


def test_generate_summary(sample_results):

    summary = hp.generate_summary(
        sample_results
    )

    assert len(summary) == 3

    assert (summary["evaluated_sources"] == 4).all()

    assert (summary["rank"] == 1).all()

    assert (
        summary["delta_macro_roc_auc"] == 0
    ).all()

    assert (
        summary["delta_macro_fpr"] == 0
    ).all()

    assert (
        summary["delta_worst_recall"] == 0
    ).all()


def test_summary_rejects_missing_source(sample_results):

    incomplete = sample_results.iloc[:-1]

    with pytest.raises(
        ValueError,
        match="Incomplete cross-source evaluation",
    ):

        hp.generate_summary(incomplete)


def test_summary_rejects_empty_results():

    with pytest.raises(
        ValueError,
        match="Experiment results cannot be empty",
    ):

        hp.generate_summary(pd.DataFrame())


def test_save_reports(
    tmp_path,
    monkeypatch,
    sample_results,
):

    monkeypatch.setattr(
        hp,
        "REPORT_DIR",
        tmp_path,
    )

    summary = hp.generate_summary(
        sample_results
    )

    best = hp.save_reports(
        sample_results,
        summary,
    )

    assert len(best) == 3

    expected_files = {
        "hyperparameters_source_results.csv": 12,
        "hyperparameters_trials.csv": 3,
        "hyperparameters_best.csv": 3,
    }

    for filename, expected_rows in expected_files.items():

        path = tmp_path / filename

        assert path.is_file()

        dataframe = pd.read_csv(path)

        assert len(dataframe) == expected_rows
        