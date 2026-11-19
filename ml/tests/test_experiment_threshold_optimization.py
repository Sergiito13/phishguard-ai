"""
PhishGuard AI - Threshold Optimization Tests.

Validate:
- Threshold configuration.
- Classification metrics.
- Source isolation.
- Domain-group isolation.
- Experiment execution.
- Cross-source summaries.
- Diagnostic shortlist.
- CSV report generation.

All tests use synthetic data.
No real models or datasets are modified.
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

import experiment_threshold_optimization as sweep


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def synthetic_data():

    features = set()

    for name in sweep.CANDIDATES:
        features.update(
            sweep.FEATURE_SETS[name]
        )

    records = []

    for source in sweep.SOURCES:

        for label in (0, 1):

            for index in range(5):

                record = {
                    "url": (
                        f"https://{source}-{label}-{index}.example"
                    ),
                    "sources": source,
                    "group_key": (
                        f"domain:{source}-{label}-{index}.example"
                    ),
                    "label": label,
                }

                value = (
                    0.2
                    + label * 0.6
                    + index * 0.01
                )

                record.update({
                    feature: value
                    for feature in features
                })

                records.append(
                    record
                )

    return pd.DataFrame(
        records
    )


# ============================================================
# 3. FAKE MODEL
# ============================================================

class FakeModel:

    fits = 0

    def fit(self, X, y):

        assert set(
            np.unique(y)
        ) == {0, 1}

        self.__class__.fits += 1

        return self

    def predict_proba(self, X):

        positive = np.clip(
            X[:, 0],
            0.01,
            0.99,
        )

        return np.column_stack([
            1 - positive,
            positive,
        ])


@pytest.fixture
def fitted_results(
    monkeypatch,
    synthetic_data,
):

    FakeModel.fits = 0

    monkeypatch.setattr(
        sweep,
        "create_model",
        lambda params: FakeModel(),
    )

    parameters = {
        name: {"max_iter": 5}
        for name in sweep.CANDIDATES
    }

    results = sweep.run_experiment(
        synthetic_data,
        parameters,
    )

    assert FakeModel.fits == 8

    return results


# ============================================================
# 4. CONFIGURATION TESTS
# ============================================================

def test_configurations():

    assert set(
        sweep.CANDIDATES
    ) == {
        "hybrid_v1_extended",
        "extended_v2",
    }

    assert len(
        sweep.SOURCES
    ) == 4

    assert (
        sweep.REFERENCE_THRESHOLD
        in sweep.THRESHOLDS
    )

    assert (
        0
        < min(sweep.THRESHOLDS)
        < max(sweep.THRESHOLDS)
        < 1
    )


def test_threshold_validation():

    result = sweep.validate_thresholds(
        [0.7, 0.5, 0.3]
    )

    assert result == (
        0.3,
        0.5,
        0.7,
    )


@pytest.mark.parametrize(
    "thresholds",
    [
        [0.3],
        [],
        [0.5, 0.5],
        [0.5, 1.1],
        [0.5, np.nan],
        [0.5, -0.1],
    ],
)
def test_invalid_threshold_grid(
    thresholds,
):

    with pytest.raises(
        ValueError
    ):
        sweep.validate_thresholds(
            thresholds
        )


# ============================================================
# 5. CLASSIFICATION METRICS
# ============================================================

def test_threshold_metrics_perfect_classification():

    metrics = sweep.threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        0.5,
    )

    assert metrics["recall"] == 1

    assert metrics["fpr"] == 0

    assert metrics["f1"] == 1

    assert metrics["true_positives"] == 2

    assert metrics["true_negatives"] == 2


def test_threshold_metrics_fp_fn():

    metrics = sweep.threshold_metrics(
        [0, 0, 1, 1],
        [0.8, 0.1, 0.1, 0.8],
        0.5,
    )

    assert metrics["false_positives"] == 1

    assert metrics["false_negatives"] == 1

    assert metrics["fpr"] == pytest.approx(
        0.5
    )

    assert metrics["recall"] == pytest.approx(
        0.5
    )


def test_threshold_boundary():

    metrics = sweep.threshold_metrics(
        [0, 1],
        [0.5, 0.5],
        0.5,
    )

    assert metrics["false_positives"] == 1

    assert metrics["true_positives"] == 1


@pytest.mark.parametrize(
    "labels,scores,threshold",
    [
        (
            [0, 0],
            [0.1, 0.2],
            0.5,
        ),
        (
            [0, 1],
            [0.3],
            0.5,
        ),
        (
            [0, 1],
            [0.5, np.nan],
            0.5,
        ),
        (
            [0, 1],
            [1.1, 0.5],
            0.5,
        ),
        (
            [0, 1],
            [0.1, 0.9],
            1.0,
        ),
        (
            [],
            [],
            0.5,
        ),
    ],
)
def test_threshold_metrics_invalid(
    labels,
    scores,
    threshold,
):

    with pytest.raises(
        ValueError
    ):
        sweep.threshold_metrics(
            labels,
            scores,
            threshold,
        )


def test_threshold_recall_and_fpr_monotonicity():

    labels = [
        0, 0, 1, 1,
    ]

    scores = [
        0.1, 0.6, 0.4, 0.9,
    ]

    low = sweep.threshold_metrics(
        labels,
        scores,
        0.3,
    )

    high = sweep.threshold_metrics(
        labels,
        scores,
        0.7,
    )

    assert (
        low["recall"]
        >= high["recall"]
    )

    assert (
        low["fpr"]
        >= high["fpr"]
    )


# ============================================================
# 6. EXPERIMENT EXECUTION
# ============================================================

def test_run_experiment(
    fitted_results,
):

    expected_rows = (
        2 * 4 * len(sweep.THRESHOLDS)
    )

    assert len(
        fitted_results
    ) == expected_rows

    assert not fitted_results[
        [
            "feature_set",
            "evaluation_source",
            "threshold",
        ]
    ].duplicated().any()

    assert set(
        fitted_results["evaluation_source"]
    ) == set(sweep.SOURCES)

    assert set(
        fitted_results["feature_set"]
    ) == set(sweep.CANDIDATES)


def test_run_experiment_rejects_missing_parameters(
    synthetic_data,
):

    with pytest.raises(
        ValueError,
        match="Missing optimized parameters",
    ):

        sweep.run_experiment(
            synthetic_data,
            {},
        )


def test_run_experiment_rejects_group_leakage(
    monkeypatch,
    synthetic_data,
):

    bad_folds = []

    for source in sweep.SOURCES:

        bad_folds.append({
            "source": source,
            "training_indices": np.arange(
                len(synthetic_data)
            ),
            "validation_indices": np.flatnonzero(
                synthetic_data["sources"]
                .eq(source)
                .to_numpy()
            ),
        })

    monkeypatch.setattr(
        sweep,
        "create_source_folds",
        lambda dataframe: bad_folds,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        sweep.run_experiment(
            synthetic_data,
            {
                name: {}
                for name in sweep.CANDIDATES
            },
        )


# ============================================================
# 7. SUMMARY TESTS
# ============================================================

def test_summarize_results(
    fitted_results,
):

    summary = sweep.summarize_results(
        fitted_results
    )

    assert len(summary) == (
        len(sweep.CANDIDATES)
        * len(sweep.THRESHOLDS)
    )

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    assert summary["worst_fpr"].between(
        0,
        1,
    ).all()

    assert summary["worst_recall"].between(
        0,
        1,
    ).all()

    assert (
        summary["policy_feasible"].dtype
        == bool
    )


def test_summarize_rejects_missing_or_duplicates(
    fitted_results,
):

    with pytest.raises(
        ValueError
    ):
        sweep.summarize_results(
            fitted_results.iloc[:-1]
        )

    duplicated = pd.concat([
        fitted_results,
        fitted_results.iloc[[0]],
    ])

    with pytest.raises(
        ValueError
    ):
        sweep.summarize_results(
            duplicated
        )


# ============================================================
# 8. SHORTLIST TESTS
# ============================================================

def test_shortlist_includes_reference(
    fitted_results,
):

    summary = sweep.summarize_results(
        fitted_results
    )

    shortlist = sweep.build_shortlist(
        summary
    )

    baseline = shortlist.loc[
        shortlist["selection_reason"]
        == "reference_0_50"
    ]

    assert set(
        baseline["feature_set"]
    ) == set(sweep.CANDIDATES)

    assert (
        baseline["threshold"] == 0.5
    ).all()

    assert (
        "highest_macro_f1"
        in set(shortlist["selection_reason"])
    )


def test_shortlist_no_feasible_threshold(
    fitted_results,
):

    summary = sweep.summarize_results(
        fitted_results
    )

    summary["policy_feasible"] = False

    shortlist = sweep.build_shortlist(
        summary
    )

    assert (
        "policy_feasible_candidate"
        not in set(shortlist["selection_reason"])
    )


# ============================================================
# 9. REPORT GENERATION
# ============================================================

def test_report_saving(
    tmp_path,
    monkeypatch,
    fitted_results,
):

    monkeypatch.setattr(
        sweep,
        "REPORT_DIR",
        tmp_path,
    )

    summary = sweep.summarize_results(
        fitted_results
    )

    shortlist = sweep.build_shortlist(
        summary
    )

    sweep.save_reports(
        fitted_results,
        summary,
        shortlist,
    )

    expected_rows = {
        "threshold_optimization_source_results.csv":
            len(fitted_results),

        "threshold_optimization_summary.csv":
            len(summary),

        "threshold_optimization_shortlist.csv":
            len(shortlist),
    }

    for filename, rows in expected_rows.items():

        path = tmp_path / filename

        assert path.exists()

        assert len(
            pd.read_csv(path)
        ) == rows


# ============================================================
# 10. COMPLETE WORKFLOW
# ============================================================

def test_main_no_real_datasets(
    tmp_path,
    monkeypatch,
    synthetic_data,
    fitted_results,
):

    monkeypatch.setattr(
        sweep,
        "REPORT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        sweep,
        "load_training_data",
        lambda: synthetic_data,
    )

    monkeypatch.setattr(
        sweep,
        "load_best_parameters",
        lambda: {
            name: {}
            for name in sweep.CANDIDATES
        },
    )

    monkeypatch.setattr(
        sweep,
        "run_experiment",
        lambda *args: fitted_results,
    )

    sweep.main()

    assert (
        tmp_path
        / "threshold_optimization_summary.csv"
    ).is_file()