"""
PhishGuard AI - Hybrid Feature Engineering Tests

Validate:
- Hybrid feature configurations.
- Dataset integrity.
- Group and URL leakage prevention.
- Overall and cross-source evaluation.
- Experiment result validation.
- Baseline and Extended V2 comparisons.
- Report generation.

All tests use synthetic data and simulated training.
No real datasets or trained models are modified.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import experiment_hybrid_features as hybrid

from extract_features import FEATURE_COLUMNS

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    ADDITIONAL_FEATURES,
)


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

def make_record(
    url: str,
    label: int,
    source: str,
    group_key: str,
) -> dict:
    """
    Create a synthetic record with V1 and V2 features.
    """

    record = {
        "url": url,
        "label": label,
        "sources": source,
        "group_key": group_key,
        "domain_status": "domain",
    }

    record.update({
        feature: 1.0
        for feature in FEATURE_COLUMNS
    })

    record.update({
        feature: 2.0
        for feature in FEATURE_COLUMNS_V2
    })

    return record


@pytest.fixture
def sample_datasets():

    train_records = []

    validation_records = []

    for index, source in enumerate(hybrid.SOURCES):

        for label in [0, 1]:

            train_records.append(
                make_record(
                    f"https://train-{index}-{label}.example",
                    label,
                    source,
                    f"train-group-{index}-{label}",
                )
            )

            validation_records.append(
                make_record(
                    f"https://validation-{index}-{label}.example",
                    label,
                    source,
                    f"validation-group-{index}-{label}",
                )
            )

    train_df = pd.DataFrame(train_records)

    validation_df = pd.DataFrame(validation_records)

    return train_df, validation_df


# ============================================================
# 3. FEATURE CONFIGURATION
# ============================================================

def test_experiment_configuration():

    expected_counts = {
        "baseline_v1": 20,
        "extended_v2": 31,
        "hybrid_v1_extended": 32,
        "hybrid_v1_extended_no_https": 31,
        "hybrid_full": 51,
        "hybrid_full_no_https": 50,
        "extended_v2_with_https": 32,
    }

    assert len(hybrid.EXPERIMENTS) == 7

    assert hybrid.EXPECTED_FEATURE_COUNTS == expected_counts

    assert set(hybrid.EXPERIMENTS) == set(expected_counts)

    assert len(hybrid.ALL_FEATURES) == 51

    assert len(set(hybrid.ALL_FEATURES)) == 51

    for name, expected_count in expected_counts.items():

        config = hybrid.EXPERIMENTS[name]

        features = config["features"]

        assert len(features) == expected_count

        assert len(features) == len(set(features))

        assert set(features).issubset(
            hybrid.ALL_FEATURES
        )

        assert config["weighting"] == "none"


def test_baseline_features():

    assert hybrid.EXPERIMENTS[
        "baseline_v1"
    ]["features"] == list(FEATURE_COLUMNS)


def test_extended_v2_features():

    assert hybrid.EXPERIMENTS[
        "extended_v2"
    ]["features"] == list(FEATURE_COLUMNS_V2)


def test_hybrid_v1_extended_features():

    expected = (
        list(FEATURE_COLUMNS)
        + list(ADDITIONAL_FEATURES)
    )

    assert hybrid.EXPERIMENTS[
        "hybrid_v1_extended"
    ]["features"] == expected


def test_hybrid_full_features():

    expected = (
        list(FEATURE_COLUMNS)
        + list(FEATURE_COLUMNS_V2)
    )

    assert hybrid.EXPERIMENTS[
        "hybrid_full"
    ]["features"] == expected


@pytest.mark.parametrize(
    "experiment_name",
    [
        "extended_v2",
        "hybrid_v1_extended_no_https",
        "hybrid_full_no_https",
    ],
)
def test_https_exclusion(experiment_name):

    features = hybrid.EXPERIMENTS[
        experiment_name
    ]["features"]

    assert "is_https" not in features


def test_extended_v2_with_https():

    features = hybrid.EXPERIMENTS[
        "extended_v2_with_https"
    ]["features"]

    assert features == (
        list(FEATURE_COLUMNS_V2)
        + ["is_https"]
    )

    assert features.count("is_https") == 1


# ============================================================
# 4. CONFIGURATION VALIDATION
# ============================================================

def test_validate_experiment_configurations():

    hybrid.validate_experiment_configurations()


@pytest.mark.parametrize(
    "invalid_features,error_message",
    [
        (
            [],
            "Empty feature configuration",
        ),
        (
            ["url_length", "url_length"],
            "Duplicate features",
        ),
        (
            ["nonexistent_feature"],
            "Unknown features",
        ),
        (
            ["url_length"],
            "Unexpected feature count",
        ),
    ],
)
def test_reject_invalid_feature_configurations(
    monkeypatch,
    invalid_features,
    error_message,
):

    monkeypatch.setitem(
        hybrid.EXPERIMENTS,
        "baseline_v1",
        {
            "features": invalid_features,
            "weighting": "none",
        },
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):

        hybrid.validate_experiment_configurations()


def test_reject_invalid_weighting(monkeypatch):

    monkeypatch.setitem(
        hybrid.EXPERIMENTS,
        "baseline_v1",
        {
            "features": list(FEATURE_COLUMNS),
            "weighting": "class",
        },
    )

    with pytest.raises(
        ValueError,
        match="Unexpected weighting",
    ):

        hybrid.validate_experiment_configurations()


def test_reject_missing_experiment(monkeypatch):

    monkeypatch.delitem(
        hybrid.EXPERIMENTS,
        "baseline_v1",
    )

    with pytest.raises(
        ValueError,
        match="Unexpected experiment configurations",
    ):

        hybrid.validate_experiment_configurations()


# ============================================================
# 5. DATASET VALIDATION
# ============================================================

def test_validate_dataset_features(sample_datasets):

    train_df, validation_df = sample_datasets

    hybrid.validate_dataset_features(
        train_df,
        validation_df,
    )


def test_reject_missing_dataset_features(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    train_df = train_df.drop(
        columns=["v2_hostname_entropy"]
    )

    with pytest.raises(
        ValueError,
        match="Missing columns in training dataset",
    ):

        hybrid.validate_dataset_features(
            train_df,
            validation_df,
        )


@pytest.mark.parametrize(
    "dataset_name",
    ["training", "validation"],
)
def test_reject_empty_dataset(
    sample_datasets,
    dataset_name,
):

    train_df, validation_df = sample_datasets

    if dataset_name == "training":

        train_df = train_df.iloc[:0]

    else:

        validation_df = validation_df.iloc[:0]

    with pytest.raises(
        ValueError,
        match=f"Empty {dataset_name} dataset",
    ):

        hybrid.validate_dataset_features(
            train_df,
            validation_df,
        )


def test_reject_group_leakage(sample_datasets):

    train_df, validation_df = sample_datasets

    validation_df = validation_df.copy()

    validation_df.loc[0, "group_key"] = (
        train_df.loc[0, "group_key"]
    )

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        hybrid.validate_dataset_features(
            train_df,
            validation_df,
        )


def test_reject_url_leakage(sample_datasets):

    train_df, validation_df = sample_datasets

    validation_df = validation_df.copy()

    validation_df.loc[0, "url"] = (
        train_df.loc[0, "url"]
    )

    with pytest.raises(
        ValueError,
        match="URL leakage detected",
    ):

        hybrid.validate_dataset_features(
            train_df,
            validation_df,
        )


# ============================================================
# 6. OVERALL EVALUATION
# ============================================================

def test_evaluate_overall(
    monkeypatch,
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    calls = []

    def fake_run_experiment(
        experiment_name,
        config,
        train_df,
        validation_df,
        evaluation_source,
    ):

        assert evaluation_source == "overall"

        assert len(train_df) == 8

        assert len(validation_df) == 8

        assert config["weighting"] == "none"

        assert config["features"] == (
            hybrid.EXPERIMENTS[
                experiment_name
            ]["features"]
        )

        calls.append(experiment_name)

        return {
            "experiment": experiment_name,
            "evaluation_source": evaluation_source,
        }

    monkeypatch.setattr(
        hybrid,
        "run_experiment",
        fake_run_experiment,
    )

    results = hybrid.evaluate_overall(
        train_df,
        validation_df,
    )

    assert len(results) == 7

    assert len(calls) == 7

    assert set(calls) == set(
        hybrid.EXPERIMENTS
    )


# ============================================================
# 7. CROSS-SOURCE EVALUATION
# ============================================================

def test_evaluate_cross_source(
    monkeypatch,
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    # A URL associated with multiple sources must
    # be excluded when any of its sources is held out.

    shared_record = make_record(
        "https://shared.example",
        1,
        "legitphish|url_phish",
        "shared-training-group",
    )

    train_df = pd.concat(
        [
            train_df,
            pd.DataFrame([shared_record]),
        ],
        ignore_index=True,
    )

    calls = []

    def fake_run_experiment(
        experiment_name,
        config,
        train_df,
        validation_df,
        evaluation_source,
    ):

        assert evaluation_source in hybrid.SOURCES

        assert not hybrid.belongs_to_source(
            train_df,
            evaluation_source,
        ).any()

        assert hybrid.belongs_to_source(
            validation_df,
            evaluation_source,
        ).all()

        assert set(train_df["label"]) == {0, 1}

        assert set(validation_df["label"]) == {0, 1}

        assert len(validation_df) == 2

        assert set(
            train_df["group_key"]
        ).isdisjoint(
            validation_df["group_key"]
        )

        if evaluation_source in [
            "legitphish",
            "url_phish",
        ]:

            assert "https://shared.example" not in (
                train_df["url"].values
            )

        calls.append((
            experiment_name,
            evaluation_source,
        ))

        return {
            "experiment": experiment_name,
            "evaluation_source": evaluation_source,
        }

    monkeypatch.setattr(
        hybrid,
        "run_experiment",
        fake_run_experiment,
    )

    results = hybrid.evaluate_cross_source(
        train_df,
        validation_df,
    )

    expected_count = (
        len(hybrid.EXPERIMENTS)
        * len(hybrid.SOURCES)
    )

    assert expected_count == 28

    assert len(results) == expected_count

    assert len(set(calls)) == expected_count

    for source in hybrid.SOURCES:

        source_results = [
            result
            for result in results
            if result["evaluation_source"] == source
        ]

        assert len(source_results) == 7


def test_cross_source_rejects_empty_training(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    train_df = train_df.copy()

    train_df["sources"] = hybrid.SOURCES[0]

    with pytest.raises(
        ValueError,
        match="Empty training dataset",
    ):

        hybrid.evaluate_cross_source(
            train_df,
            validation_df,
        )


def test_cross_source_rejects_empty_validation(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    validation_df = validation_df.copy()

    validation_df["sources"] = "unknown_source"

    with pytest.raises(
        ValueError,
        match="Empty validation dataset",
    ):

        hybrid.evaluate_cross_source(
            train_df,
            validation_df,
        )


def test_cross_source_requires_training_classes(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    train_df = train_df.copy()

    train_df["label"] = 0

    with pytest.raises(
        ValueError,
        match="Training requires both classes",
    ):

        hybrid.evaluate_cross_source(
            train_df,
            validation_df,
        )


def test_cross_source_requires_validation_classes(
    sample_datasets,
):

    train_df, validation_df = sample_datasets

    validation_df = validation_df.copy()

    validation_df["label"] = 1

    with pytest.raises(
        ValueError,
        match="Validation requires both classes",
    ):

        hybrid.evaluate_cross_source(
            train_df,
            validation_df,
        )


# ============================================================
# 8. SYNTHETIC EXPERIMENT RESULTS
# ============================================================

@pytest.fixture
def experiment_results():
    """
    Generate synthetic metrics for all 35 experiments.
    """

    results = []

    for name, config in hybrid.EXPERIMENTS.items():

        is_baseline = name == "baseline_v1"

        is_extended_v2 = name == "extended_v2"

        if is_baseline:

            f1 = 0.80

            recall = 0.90

            roc_auc = 0.83

            fpr = 0.30

        elif is_extended_v2:

            f1 = 0.75

            recall = 0.80

            roc_auc = 0.88

            fpr = 0.15

        else:

            f1 = 0.85

            recall = 0.92

            roc_auc = 0.90

            fpr = 0.20

        for source in [
            "overall",
            *hybrid.SOURCES,
        ]:

            results.append({
                "experiment": name,

                "evaluation_source": source,

                "feature_count": len(
                    config["features"]
                ),

                "weighting": "none",

                "training_samples": 100,

                "validation_samples": 20,

                "accuracy": 0.85,

                "precision": 0.85,

                "recall": recall,

                "f1": f1,

                "roc_auc": roc_auc,

                "average_precision": 0.90,

                "fpr": fpr,

                "false_positives": 3,

                "false_negatives": 2,
            })

    return results


# ============================================================
# 9. RESULT VALIDATION
# ============================================================

def test_validate_results(experiment_results):

    results_df = pd.DataFrame(
        experiment_results
    )

    assert len(results_df) == 35

    hybrid.validate_results(results_df)


def test_validate_results_missing_columns(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    ).drop(
        columns=["roc_auc"]
    )

    with pytest.raises(
        ValueError,
        match="Missing result columns",
    ):

        hybrid.validate_results(results_df)


def test_validate_results_incomplete(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    ).iloc[:-1]

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected experiment results",
    ):

        hybrid.validate_results(results_df)


def test_validate_results_unexpected_source(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df.loc[
        0,
        "evaluation_source",
    ] = "unknown_source"

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected experiment results",
    ):

        hybrid.validate_results(results_df)


def test_validate_results_duplicates(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df = pd.concat(
        [
            results_df,
            results_df.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate experiment results",
    ):

        hybrid.validate_results(results_df)


def test_validate_results_invalid_feature_count(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df.loc[
        0,
        "feature_count",
    ] = 999

    with pytest.raises(
        ValueError,
        match="Invalid feature count in results",
    ):

        hybrid.validate_results(results_df)


@pytest.mark.parametrize(
    "column,value",
    [
        ("accuracy", 1.5),
        ("precision", -0.1),
        ("recall", 1.1),
        ("f1", -1.0),
        ("roc_auc", 2.0),
        ("average_precision", float("nan")),
        ("fpr", -0.5),
    ],
)
def test_validate_results_invalid_metrics(
    experiment_results,
    column,
    value,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    results_df.loc[0, column] = value

    with pytest.raises(
        ValueError,
        match="Invalid metric values",
    ):

        hybrid.validate_results(results_df)


# ============================================================
# 10. CROSS-SOURCE SUMMARY
# ============================================================

def test_generate_cross_source_summary(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    # Overall validation must not influence macro metrics.

    overall_mask = (
        results_df["evaluation_source"]
        == "overall"
    )

    results_df.loc[
        overall_mask,
        "f1",
    ] = 0.99

    summary = (
        hybrid.generate_cross_source_summary(
            results_df
        )
    )

    assert len(summary) == 7

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    baseline = summary.loc[
        summary["experiment"] == "baseline_v1"
    ].iloc[0]

    assert baseline["macro_f1"] == pytest.approx(
        0.80
    )

    assert baseline["macro_fpr"] == pytest.approx(
        0.30
    )

    assert baseline["macro_roc_auc"] == pytest.approx(
        0.83
    )

    assert summary[
        "macro_roc_auc"
    ].is_monotonic_decreasing


# ============================================================
# 11. BASELINE COMPARISON
# ============================================================

def test_calculate_baseline_deltas(
    experiment_results,
):

    summary = (
        hybrid.generate_cross_source_summary(
            pd.DataFrame(experiment_results)
        )
    )

    comparison = hybrid.calculate_baseline_deltas(
        summary
    )

    baseline = comparison.loc[
        comparison["experiment"] == "baseline_v1"
    ].iloc[0]

    delta_columns = [
        "delta_macro_f1",
        "delta_macro_roc_auc",
        "delta_macro_recall",
        "delta_macro_fpr",
        "delta_worst_fpr",
    ]

    for column in delta_columns:

        assert baseline[column] == pytest.approx(
            0.0
        )

    extended_v2 = comparison.loc[
        comparison["experiment"] == "extended_v2"
    ].iloc[0]

    assert extended_v2[
        "delta_macro_f1"
    ] == pytest.approx(-0.05)

    assert extended_v2[
        "delta_macro_roc_auc"
    ] == pytest.approx(0.05)

    assert extended_v2[
        "delta_macro_fpr"
    ] == pytest.approx(-0.15)


# ============================================================
# 12. EXTENDED V2 COMPARISON
# ============================================================

def test_calculate_v2_deltas(
    experiment_results,
):

    summary = (
        hybrid.generate_cross_source_summary(
            pd.DataFrame(experiment_results)
        )
    )

    comparison = hybrid.calculate_v2_deltas(
        summary
    )

    reference = comparison.loc[
        comparison["experiment"] == "extended_v2"
    ].iloc[0]

    delta_columns = [
        "delta_v2_macro_f1",
        "delta_v2_macro_roc_auc",
        "delta_v2_macro_recall",
        "delta_v2_macro_fpr",
        "delta_v2_worst_fpr",
        "delta_v2_worst_recall",
    ]

    for column in delta_columns:

        assert reference[column] == pytest.approx(
            0.0
        )

    hybrid_model = comparison.loc[
        comparison["experiment"] == "hybrid_full"
    ].iloc[0]

    assert hybrid_model[
        "delta_v2_macro_f1"
    ] == pytest.approx(0.10)

    assert hybrid_model[
        "delta_v2_macro_roc_auc"
    ] == pytest.approx(0.02)

    assert hybrid_model[
        "delta_v2_macro_recall"
    ] == pytest.approx(0.12)

    assert hybrid_model[
        "delta_v2_macro_fpr"
    ] == pytest.approx(0.05)

    assert hybrid_model[
        "delta_v2_worst_recall"
    ] == pytest.approx(0.12)


def test_v2_comparison_requires_reference(
    experiment_results,
):

    summary = (
        hybrid.generate_cross_source_summary(
            pd.DataFrame(experiment_results)
        )
    )

    summary = summary.loc[
        summary["experiment"] != "extended_v2"
    ]

    with pytest.raises(
        ValueError,
        match="Expected exactly one Extended V2 result",
    ):

        hybrid.calculate_v2_deltas(summary)


# ============================================================
# 13. REPORT GENERATION
# ============================================================

def test_generate_reports(
    tmp_path,
    monkeypatch,
    experiment_results,
):

    monkeypatch.setattr(
        hybrid,
        "REPORT_DIR",
        tmp_path,
    )

    hybrid.generate_reports(
        experiment_results
    )

    expected_reports = {
        "hybrid_features_results.csv": 35,

        "hybrid_features_overall.csv": 7,

        "hybrid_features_summary.csv": 7,

        "hybrid_features_baseline_comparison.csv": 7,

        "hybrid_features_v2_comparison.csv": 7,
    }

    assert len(
        list(tmp_path.glob("*.csv"))
    ) == 5

    for filename, expected_rows in (
        expected_reports.items()
    ):

        output_file = tmp_path / filename

        assert output_file.is_file()

        dataframe = pd.read_csv(
            output_file
        )

        assert len(dataframe) == expected_rows

    overall_df = pd.read_csv(
        tmp_path / "hybrid_features_overall.csv"
    )

    assert (
        overall_df["evaluation_source"] == "overall"
    ).all()

    summary_df = pd.read_csv(
        tmp_path / "hybrid_features_summary.csv"
    )

    assert "worst_recall" in summary_df.columns

    assert "worst_fpr" in summary_df.columns

    baseline_df = pd.read_csv(
        tmp_path
        / "hybrid_features_baseline_comparison.csv"
    )

    assert "delta_macro_fpr" in baseline_df.columns

    v2_df = pd.read_csv(
        tmp_path
        / "hybrid_features_v2_comparison.csv"
    )

    assert "delta_v2_macro_fpr" in v2_df.columns

    assert "delta_v2_worst_recall" in v2_df.columns


def test_invalid_results_do_not_generate_reports(
    tmp_path,
    monkeypatch,
    experiment_results,
):

    monkeypatch.setattr(
        hybrid,
        "REPORT_DIR",
        tmp_path,
    )

    incomplete_results = experiment_results[:-1]

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected experiment results",
    ):

        hybrid.generate_reports(
            incomplete_results
        )

    assert not list(
        tmp_path.glob("*.csv")
    )


# ============================================================
# 14. COMPLETE EXPERIMENT WORKFLOW
# ============================================================

def test_main(
    monkeypatch,
    sample_datasets,
    experiment_results,
):

    train_df, validation_df = sample_datasets

    calls = []

    overall_results = [
        result
        for result in experiment_results
        if result["evaluation_source"] == "overall"
    ]

    cross_source_results = [
        result
        for result in experiment_results
        if result["evaluation_source"] != "overall"
    ]

    monkeypatch.setattr(
        hybrid,
        "load_datasets",
        lambda: (train_df, validation_df),
    )

    monkeypatch.setattr(
        hybrid,
        "evaluate_overall",
        lambda train, validation: overall_results,
    )

    monkeypatch.setattr(
        hybrid,
        "evaluate_cross_source",
        lambda train, validation: cross_source_results,
    )

    def fake_generate_reports(results):

        assert len(results) == 35

        hybrid.validate_results(
            pd.DataFrame(results)
        )

        calls.append(len(results))

    monkeypatch.setattr(
        hybrid,
        "generate_reports",
        fake_generate_reports,
    )

    hybrid.main()

    assert calls == [35]