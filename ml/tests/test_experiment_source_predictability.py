"""
PhishGuard AI - Source Predictability Tests.

Tests use synthetic data.

No real training, validation or test dataset is modified.
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

import experiment_source_predictability as experiment


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture(scope="module")
def synthetic_data():

    rows = []

    for source_id, source in enumerate(experiment.SOURCES):

        for label in (0, 1):

            for index in range(30):

                record = {
                    "sources": source,

                    "label": label,

                    "group_key": (
                        f"domain:{source}-{label}-{index}.example"
                    ),
                }

                record.update({
                    feature: float(
                        source_id * 3
                        + label
                        + (index % 7) * 0.1
                        + feature_id * 0.01
                    )

                    for feature_id, feature
                    in enumerate(experiment.FEATURES)
                })

                rows.append(
                    record
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 3. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def reports(synthetic_data):

    return experiment.run_experiment(
        synthetic_data,

        train_limit=12,

        holdout_limit=6,

        importance_limit=4,

        importance_repeats=2,
    )


# ============================================================
# 4. CONFIGURATION
# ============================================================

def test_configuration():

    assert len(experiment.SOURCES) == 4

    assert len(experiment.REPORT_FILES) == 5

    assert len(experiment.FEATURES) > 0

    assert all(
        feature.startswith("v2_")
        for feature in experiment.FEATURES
    )


# ============================================================
# 5. SINGLE SOURCE DATA
# ============================================================

def test_keep_all_single_source_rows(
    synthetic_data,
):

    single, coverage = experiment.prepare_single_source(
        synthetic_data
    )

    assert len(single) == len(
        synthetic_data
    )

    assert coverage.iloc[0][
        "excluded_multi_source_rows"
    ] == 0


def test_multisource_urls_are_excluded(
    synthetic_data,
):

    altered = synthetic_data.copy()

    altered.loc[0, "sources"] = (
        f"{experiment.SOURCES[0]}|"
        f"{experiment.SOURCES[1]}"
    )

    single, coverage = experiment.prepare_single_source(
        altered
    )

    assert len(single) == len(altered) - 1

    assert coverage.iloc[0][
        "excluded_multi_source_rows"
    ] == 1

    assert "|" not in "".join(
        single["source"].tolist()
    )


# ============================================================
# 6. INVALID SOURCE MEMBERSHIPS
# ============================================================

@pytest.mark.parametrize(
    "membership",
    [
        "unknown",
        "",
        None,
        "legitphish|legitphish",
    ],
)
def test_invalid_source_memberships(
    synthetic_data,
    membership,
):

    altered = synthetic_data.copy()

    altered.loc[
        0,
        "sources",
    ] = membership

    with pytest.raises(
        ValueError,
        match="source membership",
    ):

        experiment.prepare_single_source(
            altered
        )


# ============================================================
# 7. DATA VALIDATION
# ============================================================

def test_missing_feature_rejected(
    synthetic_data,
):

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):

        experiment.prepare_single_source(
            synthetic_data.drop(
                columns=experiment.FEATURES[0]
            )
        )


def test_invalid_feature_rejected(
    synthetic_data,
):

    altered = synthetic_data.copy()

    altered.loc[
        0,
        experiment.FEATURES[0],
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="Invalid V2 features",
    ):

        experiment.prepare_single_source(
            altered
        )


def test_missing_group_rejected(
    synthetic_data,
):

    altered = synthetic_data.copy()

    altered.loc[
        0,
        "group_key",
    ] = ""

    with pytest.raises(
        ValueError,
        match="Missing domain group",
    ):

        experiment.prepare_single_source(
            altered
        )


# ============================================================
# 8. GROUP SPLIT ISOLATION
# ============================================================

def test_group_split_isolation(
    synthetic_data,
):

    single, _ = experiment.prepare_single_source(
        synthetic_data
    )

    fit, held = experiment.group_holdout(
        single
    )

    assert fit["group_key"].nunique() > 0

    assert held["group_key"].nunique() > 0

    assert set(
        fit["group_key"]
    ).isdisjoint(
        set(held["group_key"])
    )


def test_group_split_isolation_when_sources_share_group(
    synthetic_data,
):

    altered = synthetic_data.copy()

    altered.loc[
        30,
        "group_key",
    ] = altered.loc[
        0,
        "group_key",
    ]

    single, _ = experiment.prepare_single_source(
        altered
    )

    fit, held = experiment.group_holdout(
        single
    )

    assert set(
        fit["group_key"]
    ).isdisjoint(
        set(held["group_key"])
    )


# ============================================================
# 9. BALANCED SAMPLING
# ============================================================

def test_balanced_sampling(
    synthetic_data,
):

    single, _ = experiment.prepare_single_source(
        synthetic_data
    )

    result = experiment.balanced_sample(
        single,
        5,
    )

    assert len(result) == (
        5 * len(experiment.SOURCES) * 2
    )

    assert result.groupby(
        [
            "source",
            "label",
        ]
    ).size().eq(5).all()

    assert result.index.is_unique


def test_zero_cap_rejected(
    synthetic_data,
):

    single, _ = experiment.prepare_single_source(
        synthetic_data
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):

        experiment.balanced_sample(
            single,
            0,
        )


def test_missing_source_cell_rejected(
    synthetic_data,
):

    single, _ = experiment.prepare_single_source(
        synthetic_data
    )

    single = single.loc[
        ~(
            (
                single["source"]
                == experiment.SOURCES[0]
            )
            &
            (
                single["label"] == 1
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing source/label cell",
    ):

        experiment.balanced_sample(
            single,
            5,
        )


# ============================================================
# 10. REPORT GENERATION
# ============================================================

def test_experiment_generates_five_reports(
    reports,
):

    assert set(reports) == set(
        experiment.REPORT_FILES
    )

    assert len(
        reports["summary"]
    ) == 3

    assert len(
        reports["per_source"]
    ) == 12

    assert len(
        reports["confusion"]
    ) == 48

    assert len(
        reports["importance"]
    ) == (
        3 * len(experiment.FEATURES)
    )


# ============================================================
# 11. CLASSIFICATION METRICS
# ============================================================

def test_accuracy_ranges_and_chance(
    reports,
):

    summary = reports[
        "summary"
    ]

    assert summary[
        "accuracy"
    ].between(0, 1).all()

    assert summary[
        "balanced_accuracy"
    ].between(0, 1).all()

    assert summary[
        "macro_f1"
    ].between(0, 1).all()

    assert summary[
        "chance_balanced_accuracy"
    ].eq(0.25).all()

    assert set(
        summary["scenario"]
    ) == {
        "all",
        "legitimate_only",
        "phishing_only",
    }


# ============================================================
# 12. CONFUSION MATRIX VALIDATION
# ============================================================

def test_confusion_matrix_counts_match(
    reports,
):

    for scenario in reports[
        "summary"
    ]["scenario"]:

        summary = reports[
            "summary"
        ].set_index(
            "scenario"
        ).loc[
            scenario
        ]

        confusion = reports[
            "confusion"
        ].loc[
            reports["confusion"]["scenario"]
            == scenario
        ]

        assert confusion[
            "count"
        ].sum() == summary[
            "holdout_samples"
        ]

        by_source = confusion.groupby(
            "true_source"
        )["count"].sum()

        per_source = reports[
            "per_source"
        ].loc[
            reports["per_source"]["scenario"]
            == scenario
        ].set_index(
            "source"
        )

        assert all(
            by_source[source]
            == per_source.loc[
                source,
                "holdout_samples",
            ]

            for source in experiment.SOURCES
        )


# ============================================================
# 13. PRIVACY VALIDATION
# ============================================================

def test_no_sensitive_columns(
    reports,
):

    forbidden = {
        "url",
        "urls",
        "group_key",
        "sources",
        "hostname",
        "domain",
    }

    assert all(
        forbidden.isdisjoint(
            frame.columns
        )

        for frame in reports.values()
    )


# ============================================================
# 14. SAVE REPORTS
# ============================================================

def test_save_reports(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.save_reports(
        reports
    )

    assert len(
        list(
            tmp_path.glob(
                "source_predictability_*.csv"
            )
        )
    ) == 5

    assert len(
        pd.read_csv(
            tmp_path
            / experiment.REPORT_FILES["summary"]
        )
    ) == 3


# ============================================================
# 15. REJECT SENSITIVE REPORT
# ============================================================

def test_reject_sensitive_report(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    altered = {
        name: frame.copy()

        for name, frame in reports.items()
    }

    altered["summary"]["url"] = (
        "https://private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        experiment.save_reports(
            altered
        )

    assert not list(
        tmp_path.glob("*.csv")
    )


# ============================================================
# 16. COMPLETE WORKFLOW
# ============================================================

def test_main_with_synthetic_data(
    synthetic_data,
    tmp_path,
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        experiment,
        "load_training_data",
        lambda: synthetic_data,
    )

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.main()

    output = capsys.readouterr().out

    assert (
        "SOURCE PREDICTABILITY SUMMARY"
        in output
    )

    assert len(
        list(
            tmp_path.glob("*.csv")
        )
    ) == 5