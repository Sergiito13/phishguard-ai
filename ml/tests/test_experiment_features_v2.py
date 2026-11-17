"""
PhishGuard AI - Feature Engineering V2 Experiment Tests

Validate:
- Experiment configurations.
- Dataset alignment.
- Original feature preservation.
- Group and URL leakage prevention.
- Overall evaluation.
- Cross-source evaluation.
- Macro metrics.
- Baseline comparisons.
- Report generation.

No real datasets or trained model artifacts are modified.
"""

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

import experiment_features_v2 as experiment

from extract_features import FEATURE_COLUMNS

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    PROTOCOL_INDEPENDENT_FEATURES,
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
    Create a synthetic record containing V1 and V2 features.
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
def sample_dataframe():

    return pd.DataFrame([
        make_record(
            "https://example.com",
            0,
            "legitphish",
            "domain:example.com",
        ),
        make_record(
            "https://example.org/login",
            1,
            "url_phish",
            "domain:example.org",
        ),
    ])


@pytest.fixture
def dataset_files(
    tmp_path,
    sample_dataframe,
):
    """
    Generate aligned synthetic V1 and V2 CSV files.
    """

    original_file = tmp_path / "original.csv"

    v2_file = tmp_path / "v2.csv"

    original_df = sample_dataframe[
        [
            *experiment.METADATA_COLUMNS,
            *FEATURE_COLUMNS,
        ]
    ]

    original_df.to_csv(
        original_file,
        index=False,
    )

    sample_dataframe.to_csv(
        v2_file,
        index=False,
    )

    return original_file, v2_file


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

def test_experiment_configuration():

    assert set(experiment.EXPERIMENTS) == {
        "baseline_v1",
        "without_https_v1",
        "protocol_independent_v2",
        "extended_v2",
    }

    baseline = experiment.EXPERIMENTS[
        "baseline_v1"
    ]

    without_https = experiment.EXPERIMENTS[
        "without_https_v1"
    ]

    independent = experiment.EXPERIMENTS[
        "protocol_independent_v2"
    ]

    extended = experiment.EXPERIMENTS[
        "extended_v2"
    ]

    assert baseline["features"] == list(
        FEATURE_COLUMNS
    )

    assert len(baseline["features"]) == 20

    assert len(without_https["features"]) == 19

    assert "is_https" not in (
        without_https["features"]
    )

    assert independent["features"] == list(
        PROTOCOL_INDEPENDENT_FEATURES
    )

    assert len(independent["features"]) == 19

    assert extended["features"] == list(
        FEATURE_COLUMNS_V2
    )

    assert len(extended["features"]) == 31

    assert len(experiment.ALL_FEATURES) == 51

    assert len(set(experiment.ALL_FEATURES)) == 51

    # All configurations must use identical
    # training strategies.

    for config in experiment.EXPERIMENTS.values():

        assert config["weighting"] == "none"


# ============================================================
# 4. DATASET LOADING
# ============================================================

def test_load_dataset(
    dataset_files,
    sample_dataframe,
):

    original_file, v2_file = dataset_files

    result = experiment.load_dataset(
        original_file,
        v2_file,
    )

    assert len(result) == len(
        sample_dataframe
    )

    assert set(
        experiment.ALL_FEATURES
    ).issubset(result.columns)

    assert result["label"].tolist() == [0, 1]

    assert result["url"].tolist() == (
        sample_dataframe["url"].tolist()
    )

    assert np.isfinite(
        result[
            experiment.ALL_FEATURES
        ].to_numpy()
    ).all()

    np.testing.assert_allclose(
        result[FEATURE_COLUMNS].to_numpy(),
        sample_dataframe[
            FEATURE_COLUMNS
        ].to_numpy(),
    )


def test_load_dataset_missing_original(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError,
        match="Original dataset not found",
    ):

        experiment.load_dataset(
            tmp_path / "missing.csv",
            tmp_path / "v2.csv",
        )


def test_load_dataset_missing_v2(
    dataset_files,
    tmp_path,
):

    original_file, _ = dataset_files

    with pytest.raises(
        FileNotFoundError,
        match="V2 dataset not found",
    ):

        experiment.load_dataset(
            original_file,
            tmp_path / "missing.csv",
        )


# ============================================================
# 5. DATASET ALIGNMENT
# ============================================================

def test_load_dataset_row_count_mismatch(
    dataset_files,
):

    original_file, v2_file = dataset_files

    dataframe = pd.read_csv(v2_file)

    dataframe = dataframe.iloc[:1]

    dataframe.to_csv(
        v2_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="different row counts",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


@pytest.mark.parametrize(
    "column,value",
    [
        (
            "url",
            "https://different.example",
        ),
        (
            "label",
            1,
        ),
        (
            "sources",
            "phishbd",
        ),
        (
            "group_key",
            "domain:different.example",
        ),
        (
            "domain_status",
            "different_status",
        ),
    ],
)
def test_load_dataset_metadata_mismatch(
    dataset_files,
    column,
    value,
):

    original_file, v2_file = dataset_files

    dataframe = pd.read_csv(v2_file)

    dataframe.loc[0, column] = value

    dataframe.to_csv(
        v2_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="metadata do not match",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


def test_load_dataset_reordered_rows(
    dataset_files,
):

    original_file, v2_file = dataset_files

    dataframe = pd.read_csv(v2_file)

    dataframe = dataframe.iloc[::-1]

    dataframe.to_csv(
        v2_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="metadata do not match",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


def test_load_dataset_modified_v1_features(
    dataset_files,
):

    original_file, v2_file = dataset_files

    dataframe = pd.read_csv(v2_file)

    dataframe.loc[0, "url_length"] = 999.0

    dataframe.to_csv(
        v2_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Original V1 features were modified",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


def test_load_dataset_invalid_v2_feature(
    dataset_files,
):

    original_file, v2_file = dataset_files

    dataframe = pd.read_csv(v2_file)

    dataframe.loc[
        0,
        "v2_url_length",
    ] = np.inf

    dataframe.to_csv(
        v2_file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="invalid numerical features",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


def test_load_dataset_invalid_labels(
    dataset_files,
):

    original_file, v2_file = dataset_files

    # Modify both datasets so the metadata
    # remains aligned but the label is invalid.

    for file_path in [
        original_file,
        v2_file,
    ]:

        dataframe = pd.read_csv(file_path)

        dataframe.loc[0, "label"] = 2

        dataframe.to_csv(
            file_path,
            index=False,
        )

    with pytest.raises(
        ValueError,
        match="Invalid labels found",
    ):

        experiment.load_dataset(
            original_file,
            v2_file,
        )


# ============================================================
# 6. DATA LEAKAGE PREVENTION
# ============================================================

def test_load_datasets_valid(
    monkeypatch,
    sample_dataframe,
):

    train_df = sample_dataframe.copy()

    validation_df = sample_dataframe.copy()

    validation_df["url"] = [
        "https://example.net",
        "https://example.edu/login",
    ]

    validation_df["group_key"] = [
        "domain:example.net",
        "domain:example.edu",
    ]

    def fake_load_dataset(
        original_file,
        v2_file,
    ):

        if original_file == experiment.TRAIN_FILE:

            return train_df

        return validation_df

    monkeypatch.setattr(
        experiment,
        "load_dataset",
        fake_load_dataset,
    )

    loaded_train, loaded_validation = (
        experiment.load_datasets()
    )

    assert len(loaded_train) == 2

    assert len(loaded_validation) == 2

    assert set(
        loaded_train["group_key"]
    ).isdisjoint(
        loaded_validation["group_key"]
    )


def test_load_datasets_detects_group_leakage(
    monkeypatch,
    sample_dataframe,
):

    train_df = sample_dataframe.copy()

    validation_df = sample_dataframe.copy()

    def fake_load_dataset(
        original_file,
        v2_file,
    ):

        if original_file == experiment.TRAIN_FILE:

            return train_df

        return validation_df

    monkeypatch.setattr(
        experiment,
        "load_dataset",
        fake_load_dataset,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        experiment.load_datasets()


def test_load_datasets_detects_url_leakage(
    monkeypatch,
    sample_dataframe,
):

    train_df = sample_dataframe.copy()

    validation_df = sample_dataframe.copy()

    # Different groups but identical URLs.

    validation_df["group_key"] = [
        "validation-group-a",
        "validation-group-b",
    ]

    def fake_load_dataset(
        original_file,
        v2_file,
    ):

        if original_file == experiment.TRAIN_FILE:

            return train_df

        return validation_df

    monkeypatch.setattr(
        experiment,
        "load_dataset",
        fake_load_dataset,
    )

    with pytest.raises(
        ValueError,
        match="URL leakage detected",
    ):

        experiment.load_datasets()


# ============================================================
# 7. OVERALL EVALUATION
# ============================================================

def test_evaluate_overall(
    monkeypatch,
    sample_dataframe,
):

    calls = []

    def fake_run_experiment(
        experiment_name,
        config,
        train_df,
        validation_df,
        evaluation_source,
    ):

        assert evaluation_source == "overall"

        assert train_df is sample_dataframe

        assert validation_df is sample_dataframe

        assert config["weighting"] == "none"

        calls.append(experiment_name)

        return {
            "experiment": experiment_name,
            "evaluation_source": evaluation_source,
        }

    monkeypatch.setattr(
        experiment,
        "run_experiment",
        fake_run_experiment,
    )

    results = experiment.evaluate_overall(
        sample_dataframe,
        sample_dataframe,
    )

    assert len(results) == 4

    assert set(calls) == set(
        experiment.EXPERIMENTS
    )


# ============================================================
# 8. CROSS-SOURCE EVALUATION
# ============================================================

def test_evaluate_cross_source(
    monkeypatch,
):

    train_records = []

    validation_records = []

    for index, source in enumerate(
        experiment.SOURCES
    ):

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

    # This record belongs to two sources.
    # It must be excluded when either is held out.

    train_records.append(
        make_record(
            "https://shared.example",
            1,
            "legitphish|url_phish",
            "shared-train-group",
        )
    )

    train_df = pd.DataFrame(train_records)

    validation_df = pd.DataFrame(
        validation_records
    )

    calls = []

    def fake_run_experiment(
        experiment_name,
        config,
        train_df,
        validation_df,
        evaluation_source,
    ):

        train_membership = (
            experiment.belongs_to_source(
                train_df,
                evaluation_source,
            )
        )

        validation_membership = (
            experiment.belongs_to_source(
                validation_df,
                evaluation_source,
            )
        )

        assert not train_membership.any()

        assert validation_membership.all()

        assert len(validation_df) == 2

        assert set(train_df["label"]) == {0, 1}

        assert set(
            validation_df["label"]
        ) == {0, 1}

        calls.append((
            experiment_name,
            evaluation_source,
        ))

        return {
            "experiment": experiment_name,
            "evaluation_source": evaluation_source,
        }

    monkeypatch.setattr(
        experiment,
        "run_experiment",
        fake_run_experiment,
    )

    results = experiment.evaluate_cross_source(
        train_df,
        validation_df,
    )

    expected_count = (
        len(experiment.EXPERIMENTS)
        * len(experiment.SOURCES)
    )

    assert len(results) == expected_count

    assert len(set(calls)) == expected_count

    for source in experiment.SOURCES:

        source_results = [
            result
            for result in results
            if result["evaluation_source"] == source
        ]

        assert len(source_results) == 4


def test_evaluate_cross_source_empty_validation(
    sample_dataframe,
):

    train_df = sample_dataframe.copy()

    validation_df = sample_dataframe.copy()

    validation_df["sources"] = "unknown_source"

    with pytest.raises(
        ValueError,
        match="Empty validation dataset",
    ):

        experiment.evaluate_cross_source(
            train_df,
            validation_df,
        )


# ============================================================
# 9. SYNTHETIC EXPERIMENT RESULTS
# ============================================================

@pytest.fixture
def experiment_results():
    """
    Generate synthetic metrics for all 20 experiments.
    """

    results = []

    for name, config in experiment.EXPERIMENTS.items():

        is_baseline = name == "baseline_v1"

        for index, source in enumerate(
            experiment.SOURCES
        ):

            results.append({
                "experiment": name,
                "evaluation_source": source,
                "feature_count": len(
                    config["features"]
                ),
                "training_samples": 100,
                "validation_samples": 20,
                "weighting": "none",
                "accuracy": 0.80,
                "precision": 0.85,
                "recall": 0.90,
                "f1": (
                    0.80
                    if is_baseline
                    else 0.75
                ),
                "roc_auc": (
                    0.85
                    if is_baseline
                    else 0.90
                ),
                "average_precision": 0.88,
                "fpr": (
                    0.20 + index * 0.10
                    if is_baseline
                    else 0.10 + index * 0.10
                ),
                "false_positives": 3,
                "false_negatives": 1,
            })

        # Overall metrics must not affect
        # the cross-source macro averages.

        results.append({
            "experiment": name,
            "evaluation_source": "overall",
            "feature_count": len(
                config["features"]
            ),
            "training_samples": 400,
            "validation_samples": 80,
            "weighting": "none",
            "accuracy": 0.99,
            "precision": 0.99,
            "recall": 0.99,
            "f1": 0.99,
            "roc_auc": 0.99,
            "average_precision": 0.99,
            "fpr": 0.01,
            "false_positives": 1,
            "false_negatives": 1,
        })

    return results


# ============================================================
# 10. CROSS-SOURCE SUMMARY
# ============================================================

def test_generate_cross_source_summary(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    summary = (
        experiment.generate_cross_source_summary(
            results_df
        )
    )

    assert len(summary) == 4

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    baseline = summary.loc[
        summary["experiment"] == "baseline_v1"
    ].iloc[0]

    assert baseline["macro_f1"] == pytest.approx(
        0.80
    )

    assert baseline[
        "macro_roc_auc"
    ] == pytest.approx(
        0.85
    )

    assert baseline["macro_fpr"] == pytest.approx(
        0.35
    )

    assert baseline["worst_fpr"] == pytest.approx(
        0.50
    )

    assert baseline[
        "worst_recall"
    ] == pytest.approx(
        0.90
    )

    assert summary[
        "macro_roc_auc"
    ].is_monotonic_decreasing


def test_summary_rejects_missing_source(
    experiment_results,
):

    results_df = pd.DataFrame(
        experiment_results
    )

    incomplete = results_df.loc[
        ~(
            (
                results_df["experiment"]
                == "baseline_v1"
            )
            & (
                results_df["evaluation_source"]
                == "url_phish"
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match="Incomplete cross-source evaluation",
    ):

        experiment.generate_cross_source_summary(
            incomplete
        )


# ============================================================
# 11. BASELINE COMPARISON
# ============================================================

def test_calculate_baseline_deltas(
    experiment_results,
):

    summary = (
        experiment.generate_cross_source_summary(
            pd.DataFrame(experiment_results)
        )
    )

    comparison = (
        experiment.calculate_baseline_deltas(
            summary
        )
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

    extended = comparison.loc[
        comparison["experiment"] == "extended_v2"
    ].iloc[0]

    assert extended[
        "delta_macro_f1"
    ] == pytest.approx(
        -0.05
    )

    assert extended[
        "delta_macro_roc_auc"
    ] == pytest.approx(
        0.05
    )

    assert extended[
        "delta_macro_fpr"
    ] == pytest.approx(
        -0.10
    )

    assert extended[
        "delta_worst_fpr"
    ] == pytest.approx(
        -0.10
    )


def test_baseline_comparison_requires_baseline(
    experiment_results,
):

    summary = (
        experiment.generate_cross_source_summary(
            pd.DataFrame(experiment_results)
        )
    )

    summary = summary.loc[
        summary["experiment"] != "baseline_v1"
    ]

    with pytest.raises(
        ValueError,
        match="Expected exactly one baseline result",
    ):

        experiment.calculate_baseline_deltas(
            summary
        )


# ============================================================
# 12. REPORT GENERATION
# ============================================================

def test_generate_reports(
    tmp_path,
    monkeypatch,
    experiment_results,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.generate_reports(
        experiment_results
    )

    expected_reports = {
        "features_v2_results.csv": 20,
        "features_v2_overall.csv": 4,
        "features_v2_summary.csv": 4,
        "features_v2_baseline_comparison.csv": 4,
    }

    assert len(
        list(tmp_path.glob("*.csv"))
    ) == 4

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
        tmp_path / "features_v2_overall.csv"
    )

    assert (
        overall_df["evaluation_source"]
        == "overall"
    ).all()

    comparison_df = pd.read_csv(
        tmp_path
        / "features_v2_baseline_comparison.csv"
    )

    assert "delta_macro_fpr" in (
        comparison_df.columns
    )

    assert "delta_macro_roc_auc" in (
        comparison_df.columns
    )