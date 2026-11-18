"""
PhishGuard AI - Optimized Models Evaluation Tests

Validate:
- Optimized hyperparameter loading.
- Experiment configuration.
- Source and domain group isolation.
- Model evaluation.
- Experiment execution.
- Cross-source comparison.
- CSV report generation.
- Complete workflow.

Tests use synthetic data and mocked training.
No real datasets or model artifacts are modified.
"""

import json
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import evaluate_optimized_models as evaluation

from experiment_hyperparameters import (
    FEATURE_SETS,
    BASELINE_PARAMS,
)


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def sample_datasets():

    features = set()

    for feature_list in FEATURE_SETS.values():
        features.update(feature_list)

    train_records = []
    validation_records = []

    for source in evaluation.SOURCES:

        for label in [0, 1]:

            train_record = {
                "url": f"https://train-{source}-{label}.example",
                "label": label,
                "sources": source,
                "group_key": f"train-{source}-{label}",
            }

            validation_record = {
                "url": f"https://val-{source}-{label}.example",
                "label": label,
                "sources": source,
                "group_key": f"validation-{source}-{label}",
            }

            for feature in features:

                train_record[feature] = float(label)

                validation_record[feature] = float(label)

            train_records.append(train_record)

            validation_records.append(validation_record)

    return (
        pd.DataFrame(train_records),
        pd.DataFrame(validation_records),
    )


@pytest.fixture
def optimized_parameters():

    return {
        name: BASELINE_PARAMS.copy()
        for name in FEATURE_SETS
    }


@pytest.fixture
def fake_metrics():

    return {
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.90,
        "f1": 0.87,
        "roc_auc": 0.92,
        "average_precision": 0.93,
        "fpr": 0.10,
        "false_positives": 1,
        "false_negatives": 1,
    }


# ============================================================
# 3. HYPERPARAMETER LOADING
# ============================================================

def test_load_best_parameters(
    tmp_path,
    monkeypatch,
):

    output_file = tmp_path / "hyperparameters_best.csv"

    records = []

    for name, features in FEATURE_SETS.items():

        records.append({
            "feature_set": name,
            "feature_count": len(features),
            "parameters": json.dumps(BASELINE_PARAMS),
        })

    pd.DataFrame(records).to_csv(
        output_file,
        index=False,
    )

    monkeypatch.setattr(
        evaluation,
        "BEST_PARAMS_FILE",
        output_file,
    )

    result = evaluation.load_best_parameters()

    assert set(result) == set(FEATURE_SETS)

    for params in result.values():
        assert params == BASELINE_PARAMS


def test_missing_hyperparameter_report(
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        evaluation,
        "BEST_PARAMS_FILE",
        tmp_path / "missing.csv",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing hyperparameter report",
    ):

        evaluation.load_best_parameters()


def test_invalid_hyperparameter_feature_count(
    tmp_path,
    monkeypatch,
):

    output_file = tmp_path / "invalid.csv"

    records = []

    for name in FEATURE_SETS:

        records.append({
            "feature_set": name,
            "feature_count": 999,
            "parameters": json.dumps(BASELINE_PARAMS),
        })

    pd.DataFrame(records).to_csv(
        output_file,
        index=False,
    )

    monkeypatch.setattr(
        evaluation,
        "BEST_PARAMS_FILE",
        output_file,
    )

    with pytest.raises(
        ValueError,
        match="Feature count mismatch",
    ):

        evaluation.load_best_parameters()


# ============================================================
# 4. EXPERIMENT CONFIGURATION
# ============================================================

def test_create_experiments(
    optimized_parameters,
):

    experiments = evaluation.create_experiments(
        optimized_parameters
    )

    assert len(experiments) == 6

    for name, features in FEATURE_SETS.items():

        original = experiments[
            f"{name}_original"
        ]

        optimized = experiments[
            f"{name}_optimized"
        ]

        assert original["features"] == features

        assert optimized["features"] == features

        assert original["variant"] == "original"

        assert optimized["variant"] == "optimized"

        assert original["parameters"] == BASELINE_PARAMS


# ============================================================
# 5. EVALUATION FOLDS
# ============================================================

def test_create_evaluation_folds(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    folds = evaluation.create_evaluation_folds(
        train_df,
        validation_df,
    )

    assert len(folds) == 5

    assert folds[0]["source"] == "overall"

    assert len(
        folds[0]["train_indices"]
    ) == 8

    assert len(
        folds[0]["validation_indices"]
    ) == 8

    for fold in folds[1:]:

        training = train_df.iloc[
            fold["train_indices"]
        ]

        validation = validation_df.iloc[
            fold["validation_indices"]
        ]

        assert len(training) == 6

        assert len(validation) == 2

        assert not evaluation.belongs_to_source(
            training,
            fold["source"],
        ).any()

        assert evaluation.belongs_to_source(
            validation,
            fold["source"],
        ).all()

        assert set(
            training["group_key"]
        ).isdisjoint(
            validation["group_key"]
        )


def test_source_group_exclusion(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    train_df = train_df.copy()

    train_df.loc[
        0,
        "sources",
    ] = "phiusiil"

    train_df.loc[
        0,
        "group_key",
    ] = validation_df.loc[0, "group_key"]

    # Overall evaluation must reject cross-split
    # domain leakage.

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        evaluation.create_evaluation_folds(
            train_df,
            validation_df,
        )


def test_group_exclusion_between_sources(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    train_df = train_df.copy()

    # A PhiUSIIL training URL shares a group
    # with a LegitPhish training URL.

    train_df.loc[
        2,
        "group_key",
    ] = train_df.loc[0, "group_key"]

    folds = evaluation.create_evaluation_folds(
        train_df,
        validation_df,
    )

    legitphish_fold = next(
        fold
        for fold in folds
        if fold["source"] == "legitphish"
    )

    training = train_df.iloc[
        legitphish_fold["train_indices"]
    ]

    assert train_df.loc[2, "url"] not in (
        training["url"].values
    )


def test_missing_validation_source(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    validation_df = validation_df.loc[
        validation_df["sources"] != "url_phish"
    ]

    with pytest.raises(
        ValueError,
        match="No validation samples",
    ):

        evaluation.create_evaluation_folds(
            train_df,
            validation_df,
        )


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

def test_evaluate_model(
    monkeypatch,
    sample_datasets,
):

    class FakeModel:

        def fit(self, X, y):

            assert len(X) == len(y)

            return self

        def predict_proba(self, X):

            probabilities = X[:, 0]

            return np.column_stack([
                1 - probabilities,
                probabilities,
            ])

    monkeypatch.setattr(
        evaluation,
        "create_model",
        lambda params: FakeModel(),
    )

    train_df, validation_df = sample_datasets

    metrics = evaluation.evaluate_model(
        train_df=train_df,
        validation_df=validation_df,
        features=["url_length"],
        parameters=BASELINE_PARAMS,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)

    assert metrics["recall"] == pytest.approx(1.0)

    assert metrics["roc_auc"] == pytest.approx(1.0)

    assert metrics["fpr"] == pytest.approx(0.0)

    assert metrics["false_positives"] == 0

    assert metrics["false_negatives"] == 0


# ============================================================
# 7. EXPERIMENT EXECUTION
# ============================================================

@pytest.fixture
def experiment_results(
    monkeypatch,
    sample_datasets,
    optimized_parameters,
    fake_metrics,
):

    train_df, validation_df = sample_datasets

    experiments = evaluation.create_experiments(
        optimized_parameters
    )

    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda *args, **kwargs: fake_metrics.copy(),
    )

    return evaluation.run_experiments(
        train_df,
        validation_df,
        experiments,
    )


def test_run_experiments(
    experiment_results,
):

    assert len(experiment_results) == 30

    assert experiment_results[
        "experiment"
    ].nunique() == 6

    assert experiment_results[
        "evaluation_source"
    ].nunique() == 5

    assert set(
        experiment_results["variant"]
    ) == {
        "original",
        "optimized",
    }

    assert (
        experiment_results["feature_count"] > 0
    ).all()


# ============================================================
# 8. CROSS-SOURCE SUMMARY
# ============================================================

def test_cross_source_summary(
    experiment_results,
):

    summary = (
        evaluation.generate_cross_source_summary(
            experiment_results
        )
    )

    assert len(summary) == 6

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    assert (
        summary["macro_roc_auc"] == 0.92
    ).all()

    assert (
        summary["macro_fpr"] == 0.10
    ).all()


# ============================================================
# 9. ORIGINAL VS OPTIMIZED
# ============================================================

def test_generate_comparison(
    experiment_results,
):

    summary = (
        evaluation.generate_cross_source_summary(
            experiment_results
        )
    )

    comparison = evaluation.generate_comparison(
        summary
    )

    assert len(comparison) == 6

    expected_columns = [
        "delta_macro_f1",
        "delta_macro_roc_auc",
        "delta_macro_recall",
        "delta_macro_fpr",
        "delta_worst_fpr",
        "delta_worst_recall",
    ]

    for column in expected_columns:

        assert column in comparison.columns

        assert (
            comparison[column] == 0
        ).all()


def test_comparison_requires_original(
    experiment_results,
):

    summary = (
        evaluation.generate_cross_source_summary(
            experiment_results
        )
    )

    summary = summary.loc[
        summary["experiment"]
        != "baseline_v1_original"
    ]

    with pytest.raises(
        ValueError,
        match="Missing original reference metrics",
    ):

        evaluation.generate_comparison(
            summary
        )


# ============================================================
# 10. REPORT GENERATION
# ============================================================

def test_save_reports(
    tmp_path,
    monkeypatch,
    experiment_results,
):

    monkeypatch.setattr(
        evaluation,
        "REPORT_DIR",
        tmp_path,
    )

    overall, summary, comparison = (
        evaluation.save_reports(
            experiment_results
        )
    )

    expected_files = {
        "optimized_models_results.csv": 30,
        "optimized_models_overall.csv": 6,
        "optimized_models_summary.csv": 6,
        "optimized_models_comparison.csv": 6,
    }

    for filename, expected_rows in (
        expected_files.items()
    ):

        file_path = tmp_path / filename

        assert file_path.is_file()

        dataframe = pd.read_csv(
            file_path
        )

        assert len(dataframe) == expected_rows

    assert len(overall) == 6

    assert len(summary) == 6

    assert len(comparison) == 6


def test_display_results(
    capsys,
    experiment_results,
):

    overall = experiment_results.loc[
        experiment_results["evaluation_source"]
        == "overall"
    ]

    summary = (
        evaluation.generate_cross_source_summary(
            experiment_results
        )
    )

    comparison = evaluation.generate_comparison(
        summary
    )

    evaluation.display_results(
        overall,
        summary,
        comparison,
        experiment_results,
    )

    output = capsys.readouterr().out

    assert "OVERALL VALIDATION COMPARISON" in output

    assert "CROSS-SOURCE COMPARISON" in output

    assert "OPTIMIZED VS ORIGINAL DELTAS" in output

    assert "URL_PHISH COMPARISON" in output


# ============================================================
# 11. COMPLETE WORKFLOW
# ============================================================

def test_main(
    tmp_path,
    monkeypatch,
    sample_datasets,
    optimized_parameters,
    experiment_results,
):

    monkeypatch.setattr(
        evaluation,
        "REPORT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        evaluation,
        "load_best_parameters",
        lambda: optimized_parameters,
    )

    monkeypatch.setattr(
        evaluation,
        "load_datasets",
        lambda: sample_datasets,
    )

    monkeypatch.setattr(
        evaluation,
        "run_experiments",
        lambda *args: experiment_results,
    )

    evaluation.main()

    assert (
        tmp_path / "optimized_models_results.csv"
    ).is_file()

    assert (
        tmp_path / "optimized_models_summary.csv"
    ).is_file()