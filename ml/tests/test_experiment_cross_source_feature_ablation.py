"""
PhishGuard AI - Cross-Source Feature Ablation Tests.

Validate:
- Feature configurations.
- Feature removal.
- Source and domain isolation.
- Model training.
- Classification metrics.
- Summary generation.
- Baseline comparisons.
- CSV report generation.

All tests use synthetic data and fake classifiers.
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

sys.path.insert(
    0,
    str(SRC_DIR),
)

import experiment_cross_source_feature_ablation as ablation


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def synthetic_data():

    features = set().union(
        *(
            set(ablation.FEATURE_SETS[name])
            for name in ablation.ABLATIONS
        )
    )

    rows = []

    for source in ablation.SOURCES:

        for label in (0, 1):

            for index in range(7):

                score = (
                    0.2
                    if label == 0
                    else 0.8
                )

                # Generate one error per class and source.

                if index == 0:
                    score = 1 - score

                row = {
                    "url": (
                        f"https://{source}-{label}-{index}.example/"
                    ),

                    "label": label,

                    "sources": source,

                    "group_key": (
                        f"domain:{source}-{label}-{index}.example"
                    ),
                }

                row.update({
                    feature: score
                    for feature in features
                })

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 3. FAKE CLASSIFIER
# ============================================================

class FakeClassifier:

    fits = 0

    def fit(
        self,
        X,
        y,
    ):

        assert set(
            np.unique(y)
        ) == {0, 1}

        self.__class__.fits += 1

        return self

    def predict_proba(
        self,
        X,
    ):

        probabilities = np.clip(
            np.asarray(X)[:, 0],
            0.01,
            0.99,
        )

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])


# ============================================================
# 4. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def results(
    monkeypatch,
    synthetic_data,
):

    FakeClassifier.fits = 0

    monkeypatch.setattr(
        ablation,
        "create_model",
        lambda params: FakeClassifier(),
    )

    parameters = {
        name: {}
        for name in ablation.ABLATIONS
    }

    result = ablation.run_experiment(
        synthetic_data,
        parameters,
    )

    assert FakeClassifier.fits == 32

    return result


# ============================================================
# 5. FEATURE CONFIGURATION TESTS
# ============================================================

def test_build_configurations():

    configs = ablation.build_configurations()

    assert len(configs) == 8

    assert set(
        config["feature_set"]
        for config in configs
    ) == set(ablation.ABLATIONS)

    assert sum(
        config["ablation"] == "full"
        for config in configs
    ) == 2

    for config in configs:

        assert config["features"]

        assert set(
            config["features"]
        ).issubset(
            ablation.FEATURE_SETS[
                config["feature_set"]
            ]
        )


def test_full_has_original_features():

    for config in ablation.build_configurations():

        if config["ablation"] == "full":

            assert (
                config["features"]
                == ablation.FEATURE_SETS[
                    config["feature_set"]
                ]
            )


def test_https_removed_only_in_requested_hybrid_variants():

    candidates = {
        config["ablation"]: config

        for config in ablation.build_configurations()

        if (
            config["feature_set"]
            == "hybrid_v1_extended"
        )
    }

    assert (
        "is_https"
        in candidates["full"]["features"]
    )

    assert (
        "is_https"
        not in candidates["without_https"]["features"]
    )

    assert (
        "slash_count"
        not in candidates["without_slash_count"]["features"]
    )

    assert (
        "is_https"
        not in candidates["without_https_and_slash"]["features"]
    )

    assert (
        "slash_count"
        not in candidates["without_https_and_slash"]["features"]
    )


# ============================================================
# 6. INVALID FEATURE CONFIGURATION TESTS
# ============================================================

def test_unknown_ablation_rejected():

    with pytest.raises(
        ValueError,
        match="Unknown excluded features",
    ):

        ablation.build_configurations(
            ablations={
                "hybrid_v1_extended": {
                    "full": (),
                    "bad": (
                        "missing_column",
                    ),
                }
            }
        )


def test_duplicate_exclusions_rejected():

    with pytest.raises(
        ValueError,
        match="Duplicate or invalid",
    ):

        ablation.build_configurations(
            ablations={
                "hybrid_v1_extended": {
                    "full": (),
                    "bad": (
                        "is_https",
                        "is_https",
                    ),
                }
            }
        )


def test_duplicate_feature_sets_rejected():

    with pytest.raises(
        ValueError,
        match="Empty or duplicate ablation",
    ):

        ablation.build_configurations(
            ablations={
                "hybrid_v1_extended": {
                    "full": (),
                    "duplicate": (),
                }
            }
        )


def test_empty_features_rejected():

    with pytest.raises(
        ValueError,
        match="Empty or duplicate ablation",
    ):

        ablation.build_configurations(
            feature_sets={
                "a": ["x"],
            },
            ablations={
                "a": {
                    "full": (),
                    "none": ("x",),
                }
            },
        )


def test_missing_baseline_rejected():

    with pytest.raises(
        ValueError,
        match="Missing baseline",
    ):

        ablation.build_configurations(
            ablations={
                "hybrid_v1_extended": {
                    "drop": (
                        "is_https",
                    ),
                }
            }
        )


# ============================================================
# 7. EXPERIMENT EXECUTION TESTS
# ============================================================

def test_eight_configs_four_sources(
    results,
):

    assert len(results) == 32

    assert not results.duplicated(
        [
            "feature_set",
            "ablation",
            "evaluation_source",
        ]
    ).any()

    assert set(
        results["evaluation_source"]
    ) == set(ablation.SOURCES)

    assert set(
        results["feature_set"]
    ) == set(ablation.ABLATIONS)


def test_evaluation_metrics(
    results,
):

    for column in (
        "fpr",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):

        assert results[
            column
        ].between(0, 1).all()

    assert (
        results["false_positives"] == 1
    ).all()

    assert (
        results["false_negatives"] == 1
    ).all()

    assert (
        results["training_samples"] > 0
    ).all()

    assert (
        results["evaluation_samples"] == 14
    ).all()


def test_missing_parameters_rejected(
    synthetic_data,
):

    with pytest.raises(
        ValueError,
        match="Missing optimized parameters",
    ):

        ablation.run_experiment(
            synthetic_data,
            {},
        )


def test_missing_feature_rejected(
    synthetic_data,
):

    missing_feature_data = synthetic_data.drop(
        columns="is_https"
    )

    with pytest.raises(
        ValueError,
        match="Missing features",
    ):

        ablation.run_experiment(
            missing_feature_data,
            {
                name: {}
                for name in ablation.ABLATIONS
            },
        )


# ============================================================
# 8. DATA ISOLATION TESTS
# ============================================================

def test_group_leakage_rejected(
    monkeypatch,
    synthetic_data,
):

    folds = [
        {
            "source": source,

            "training_indices": np.arange(
                len(synthetic_data)
            ),

            "validation_indices": np.flatnonzero(
                synthetic_data["sources"].eq(
                    source
                )
            ),
        }

        for source in ablation.SOURCES
    ]

    monkeypatch.setattr(
        ablation,
        "create_source_folds",
        lambda dataframe: folds,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage",
    ):

        ablation.run_experiment(
            synthetic_data,
            {
                name: {}
                for name in ablation.ABLATIONS
            },
        )


def test_source_leakage_rejected(
    monkeypatch,
    synthetic_data,
):

    monkeypatch.setattr(
        ablation,
        "belongs_to_source",
        lambda dataframe, source: pd.Series(
            True,
            index=dataframe.index,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Source leakage",
    ):

        ablation.run_experiment(
            synthetic_data,
            {
                name: {}
                for name in ablation.ABLATIONS
            },
        )


# ============================================================
# 9. SUMMARY TESTS
# ============================================================

def test_summarize_results(
    results,
):

    summary = ablation.summarize_results(
        results
    )

    assert len(summary) == 8

    assert (
        summary["evaluated_sources"] == 4
    ).all()

    assert (
        summary["worst_recall"]
        <= summary["macro_recall"]
    ).all()

    assert (
        summary["worst_fpr"]
        >= summary["macro_fpr"]
    ).all()


def test_summarize_rejects_missing_row(
    results,
):

    with pytest.raises(
        ValueError,
        match="Incomplete or duplicate",
    ):

        ablation.summarize_results(
            results.iloc[:-1]
        )


def test_summarize_rejects_duplicate_row(
    results,
):

    duplicate_results = pd.concat([
        results,
        results.iloc[[0]],
    ])

    with pytest.raises(
        ValueError,
        match="Incomplete or duplicate",
    ):

        ablation.summarize_results(
            duplicate_results
        )


# ============================================================
# 10. BASELINE COMPARISON TESTS
# ============================================================

def test_baseline_deltas_are_zero(
    results,
):

    summary = ablation.summarize_results(
        results
    )

    comparison = ablation.compare_to_baseline(
        summary
    )

    baseline = comparison.loc[
        comparison["ablation"] == "full"
    ]

    for column in comparison.columns:

        if column.startswith("delta_"):

            assert np.allclose(
                baseline[column],
                0,
            )


def test_delta_directions(
    results,
):

    summary = ablation.summarize_results(
        results
    )

    index = summary.index[
        (
            summary["feature_set"]
            == "extended_v2"
        )
        &
        (
            summary["ablation"]
            == "without_entropy"
        )
    ][0]

    summary.loc[
        index,
        "macro_fpr",
    ] += 0.1

    comparison = ablation.compare_to_baseline(
        summary
    )

    row = comparison.iloc[
        index
    ]

    assert row[
        "delta_macro_fpr"
    ] == pytest.approx(
        0.1
    )


def test_missing_baseline_summary_rejected(
    results,
):

    summary = ablation.summarize_results(
        results
    )

    with pytest.raises(
        ValueError,
        match="Missing exactly one baseline",
    ):

        ablation.compare_to_baseline(
            summary.loc[
                summary["ablation"] != "full"
            ]
        )


# ============================================================
# 11. CSV REPORT TESTS
# ============================================================

def test_save_reports(
    tmp_path,
    monkeypatch,
    results,
):

    monkeypatch.setattr(
        ablation,
        "REPORT_DIR",
        tmp_path,
    )

    summary = ablation.summarize_results(
        results
    )

    comparison = ablation.compare_to_baseline(
        summary
    )

    ablation.save_reports(
        results,
        summary,
        comparison,
    )

    assert len(
        list(
            tmp_path.glob(
                "cross_source_ablation_*.csv"
            )
        )
    ) == 3

    assert len(
        pd.read_csv(
            tmp_path / "cross_source_ablation_source_results.csv"
        )
    ) == 32

    assert len(
        pd.read_csv(
            tmp_path / "cross_source_ablation_summary.csv"
        )
    ) == 8


def test_private_columns_rejected(
    tmp_path,
    monkeypatch,
    results,
):

    monkeypatch.setattr(
        ablation,
        "REPORT_DIR",
        tmp_path,
    )

    summary = ablation.summarize_results(
        results
    )

    comparison = ablation.compare_to_baseline(
        summary
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        ablation.save_reports(
            results.assign(
                url="https://hidden.example"
            ),
            summary,
            comparison,
        )


# ============================================================
# 12. COMPLETE WORKFLOW TEST
# ============================================================

def test_main_with_synthetic_data(
    tmp_path,
    monkeypatch,
    synthetic_data,
    capsys,
):

    monkeypatch.setattr(
        ablation,
        "REPORT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        ablation,
        "load_training_data",
        lambda: synthetic_data,
    )

    monkeypatch.setattr(
        ablation,
        "load_best_parameters",
        lambda: {
            name: {}
            for name in ablation.ABLATIONS
        },
    )

    monkeypatch.setattr(
        ablation,
        "create_model",
        lambda params: FakeClassifier(),
    )

    ablation.main()

    output = capsys.readouterr().out

    assert (
        "ABLATION COMPARISON"
        in output
    )

    assert (
        "FOCAL SOURCE COMPARISON"
        in output
    )

    assert len(
        list(
            tmp_path.glob("*.csv")
        )
    ) == 3