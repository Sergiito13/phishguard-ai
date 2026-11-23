"""Synthetic tests for paired Extended V2 failure diagnostics.

No real datasets, saved models, or reserved validation/test files are accessed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import diagnose_candidate_failures as diag


# ============================================================
# 1. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def data():
    rows = []

    for source_id, source in enumerate(diag.SOURCES):
        for label in (0, 1):
            for i in range(10):

                rows.append({
                    "url": f"https://{source}-{label}-{i}.example/path",
                    "sources": source,
                    "group_key": f"domain:{source}-{label}-{i}.example",
                    "label": label,
                    "is_https": i % 2,

                    **{
                        feature: float(
                            source_id * 0.13
                            + label * 0.50
                            + i * 0.05
                            + feature_index * 0.01
                        )
                        for feature_index, feature
                        in enumerate(diag.FEATURES)
                    },
                })

    return pd.DataFrame(rows)


# ============================================================
# 2. FAKE MODEL
# ============================================================

class FakeModel:

    classes_ = np.array([0, 1])
    fits = []

    def fit(self, X, y, sample_weight=None):

        self.weighted = sample_weight is not None

        self.__class__.fits.append({
            "rows": len(y),
            "weighted": self.weighted,
            "weights_valid": sample_weight is None or (
                len(sample_weight) == len(y)
                and np.isfinite(sample_weight).all()
                and (sample_weight > 0).all()
            ),
        })

        return self

    def predict_proba(self, X):

        s = np.clip(
            0.10
            + np.asarray(X)[:, 0] * 0.75
            + (0.12 if self.weighted else 0.0),
            0.01,
            0.99,
        )

        return np.column_stack((1 - s, s))


# ============================================================
# 3. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def reports(data, monkeypatch):

    FakeModel.fits = []

    monkeypatch.setattr(
        diag,
        "create_model",
        lambda params: FakeModel(),
    )

    result = diag.run_experiment(
        data,
        {diag.FEATURE_SET: {}},
    )

    assert len(FakeModel.fits) == 8

    return result


# ============================================================
# 4. CONFIGURATION TESTS
# ============================================================

def test_configuration():

    assert diag.FEATURE_SET == "extended_v2"
    assert len(diag.SOURCES) == 4
    assert len(diag.FEATURES) == 31

    assert diag.STRATEGIES == (
        "uniform",
        "source_class",
    )

    assert len(diag.REPORT_FILES) == 6


def test_valid_input(data):

    diag.validate_input(
        data,
        {diag.FEATURE_SET: {}},
    )


def test_missing_parameters(data):

    with pytest.raises(
        ValueError,
        match="optimized",
    ):
        diag.validate_input(
            data,
            {},
        )


def test_missing_feature(data):

    with pytest.raises(
        ValueError,
        match="Missing training fields",
    ):
        diag.validate_input(
            data.drop(columns=diag.FEATURES[0]),
            {diag.FEATURE_SET: {}},
        )


# ============================================================
# 5. DATA VALIDATION TESTS
# ============================================================

def test_bad_labels(data):

    modified = data.copy()
    modified.loc[0, "label"] = 2

    with pytest.raises(
        ValueError,
        match="binary labels",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


def test_bad_feature(data):

    modified = data.copy()

    modified.loc[
        0,
        diag.FEATURES[0],
    ] = float("inf")

    with pytest.raises(
        ValueError,
        match="Non-finite",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


def test_bad_protocol(data):

    modified = data.copy()
    modified.loc[0, "is_https"] = 9

    with pytest.raises(
        ValueError,
        match="is_https",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


def test_bad_source(data):

    modified = data.copy()
    modified.loc[0, "sources"] = "unknown"

    with pytest.raises(
        ValueError,
        match="membership",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


def test_duplicate_source_membership(data):

    modified = data.copy()

    modified.loc[
        0,
        "sources",
    ] = "phishbd|phishbd"

    with pytest.raises(
        ValueError,
        match="membership",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


def test_missing_group(data):

    modified = data.copy()
    modified.loc[0, "group_key"] = ""

    with pytest.raises(
        ValueError,
        match="group keys",
    ):
        diag.validate_input(
            modified,
            {diag.FEATURE_SET: {}},
        )


# ============================================================
# 6. FALSE POSITIVE / FALSE NEGATIVE TESTS
# ============================================================

def test_wrong_masks_threshold_inclusive():

    masks = diag.wrong_masks(
        [0, 0, 1, 1],
        [0.50, 0.49, 0.50, 0.49],
    )

    assert masks["FP"].tolist() == [
        True, False, False, False
    ]

    assert masks["TN"].tolist() == [
        False, True, False, False
    ]

    assert masks["TP"].tolist() == [
        False, False, True, False
    ]

    assert masks["FN"].tolist() == [
        False, False, False, True
    ]


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 1], [np.nan, 0.5]),
        ([0, 1], [-0.1, 0.5]),
        ([0], [0.1, 0.9]),
        ([0, 2], [0.2, 0.9]),
    ],
)
def test_invalid_scores(labels, scores):

    with pytest.raises(
        ValueError,
        match="Invalid aligned",
    ):
        diag.wrong_masks(
            labels,
            scores,
        )


# ============================================================
# 7. ERROR CONFIDENCE TESTS
# ============================================================

def test_confidence_rows():

    rows = diag.confidence_rows(
        "phishbd",
        "uniform",
        [0, 0, 1, 1],
        [0.95, 0.52, 0.05, 0.49],
    )

    fp, fn = rows

    assert fp["errors"] == 2
    assert fn["errors"] == 2

    assert fp["high_confidence_errors"] == 1
    assert fn["high_confidence_errors"] == 1

    assert fp["near_threshold_errors"] == 1
    assert fn["near_threshold_errors"] == 1


def test_confidence_no_errors():

    rows = diag.confidence_rows(
        "phishbd",
        "uniform",
        [0, 1],
        [0.05, 0.95],
    )

    assert all(
        row["errors"] == 0
        for row in rows
    )

    assert all(
        row["high_confidence_fraction"] == 0
        for row in rows
    )


# ============================================================
# 8. PAIRED ERROR COMPARISON TESTS
# ============================================================

def test_paired_accounting():

    rows = diag.paired_change_rows(
        "phishbd",
        [0, 0, 1, 1],
        [0.8, 0.1, 0.2, 0.8],
        [0.1, 0.9, 0.7, 0.2],
    )

    assert all(
        row["resolved_by_source_class"] == 1
        for row in rows
    )

    assert all(
        row["introduced_by_source_class"] == 1
        for row in rows
    )

    assert all(
        row["persistent_errors"] == 0
        for row in rows
    )

    assert all(
        row["net_error_change"] == 0
        for row in rows
    )


# ============================================================
# 9. SCORE DISTRIBUTION TESTS
# ============================================================

def test_distribution_includes_score_one():

    rows = diag.distribution_rows(
        "phiusiil",
        "uniform",
        [0, 1],
        [0.0, 1.0],
    )

    assert len(rows) == (
        len(diag.SCORE_BINS) - 1
    ) * 2

    assert sum(
        row["count"]
        for row in rows
    ) == 2

    assert rows[-1]["count"] == 1


# ============================================================
# 10. FEATURE PROFILE TESTS
# ============================================================

def test_profile_rows_have_same_class_references(data):

    held = data.iloc[:4]

    labels = np.array([
        0, 0, 0, 0
    ])

    scores = np.array([
        0.8, 0.1, 0.9, 0.2
    ])

    rows = diag.profile_rows(
        "legitphish",
        "uniform",
        held,
        labels,
        scores,
    )

    assert len(rows) == (
        len(diag.FEATURES) * 2
    )

    assert rows[0]["error_type"] == "FP"
    assert rows[0]["reference_type"] == "TN"

    assert rows[0]["target_samples"] == 2
    assert rows[0]["reference_samples"] == 2

    fn = next(
        row
        for row in rows
        if row["error_type"] == "FN"
    )

    assert fn["target_samples"] == 0


def test_profile_rejects_misalignment(data):

    with pytest.raises(
        ValueError,
        match="not aligned",
    ):
        diag.profile_rows(
            "phishbd",
            "uniform",
            data.iloc[:2],
            [0],
            [0.5],
        )


# ============================================================
# 11. REPORT VALIDATION TESTS
# ============================================================

def test_reports_complete(reports):

    assert set(reports) == set(
        diag.REPORT_FILES
    )

    assert len(reports["summary"]) == 8

    assert len(reports["paired_changes"]) == 8

    assert len(reports["error_confidence"]) == 16

    assert len(reports["error_profiles"]) == (
        16 * len(diag.FEATURES)
    )

    assert len(reports["feature_shift"]) == (
        8 * len(diag.FEATURES)
    )

    assert len(reports["score_distribution"]) == (
        16 * (len(diag.SCORE_BINS) - 1)
    )


def test_eight_fits_weights(reports):

    assert len(FakeModel.fits) == 8

    assert sum(
        record["weighted"]
        for record in FakeModel.fits
    ) == 4

    assert all(
        record["weights_valid"]
        for record in FakeModel.fits
    )


# ============================================================
# 12. PAIRED COUNTS TESTS
# ============================================================

def test_paired_counts_match_summary(reports):

    summary = reports["summary"].set_index([
        "evaluation_source",
        "strategy",
    ])

    for row in reports["paired_changes"].itertuples():

        metric = (
            "false_positives"
            if row.error_type == "FP"
            else "false_negatives"
        )

        assert row.uniform_errors == summary.loc[
            (row.evaluation_source, "uniform"),
            metric,
        ]

        assert row.source_class_errors == summary.loc[
            (row.evaluation_source, "source_class"),
            metric,
        ]


# ============================================================
# 13. HISTOGRAM VALIDATION TESTS
# ============================================================

def test_histogram_counts_match_summary(reports):

    bins = reports["score_distribution"]

    for row in reports["summary"].itertuples():

        selected = bins.loc[
            (
                bins["evaluation_source"]
                == row.evaluation_source
            )
            &
            (
                bins["strategy"]
                == row.strategy
            )
        ]

        assert selected["count"].sum() == (
            row.evaluation_samples
        )


# ============================================================
# 14. PRIVACY TESTS
# ============================================================

def test_reports_private_columns(reports):

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    assert all(
        forbidden.isdisjoint(frame.columns)
        for frame in reports.values()
    )


# ============================================================
# 15. SAVE REPORTS TESTS
# ============================================================

def test_save_reports(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        diag,
        "REPORT_DIR",
        tmp_path,
    )

    diag.save_reports(reports)

    saved = list(
        tmp_path.glob(
            "candidate_failures_*.csv"
        )
    )

    assert len(saved) == 6

    assert pd.read_csv(
        tmp_path
        / diag.REPORT_FILES["paired_changes"]
    ).shape[0] == 8


def test_reject_sensitive_report(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        diag,
        "REPORT_DIR",
        tmp_path,
    )

    bad = {
        name: frame.copy()
        for name, frame in reports.items()
    }

    bad["summary"]["url"] = (
        "https://private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):
        diag.save_reports(bad)

    assert not list(
        tmp_path.glob("*.csv")
    )


# ============================================================
# 16. INVALID FOLD TESTS
# ============================================================

def test_invalid_fold_count(data, monkeypatch):

    monkeypatch.setattr(
        diag,
        "create_source_folds",
        lambda frame: [],
    )

    with pytest.raises(
        ValueError,
        match="Incomplete",
    ):
        diag.run_experiment(
            data,
            {diag.FEATURE_SET: {}},
        )


def test_source_leakage(data, monkeypatch):

    def bad_folds(frame):

        return [
            {
                "source": source,

                "training_indices": np.arange(
                    len(frame)
                ),

                "validation_indices": np.flatnonzero(
                    diag.belongs_to_source(
                        frame,
                        source,
                    ).to_numpy()
                ),
            }
            for source in diag.SOURCES
        ]

    monkeypatch.setattr(
        diag,
        "create_source_folds",
        bad_folds,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage",
    ):
        diag.run_experiment(
            data,
            {diag.FEATURE_SET: {}},
        )


# ============================================================
# 17. COMPLETE WORKFLOW TEST
# ============================================================

def test_main_synthetic(
    data,
    tmp_path,
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        diag,
        "load_training_data",
        lambda: data,
    )

    monkeypatch.setattr(
        diag,
        "load_best_parameters",
        lambda: {
            diag.FEATURE_SET: {}
        },
    )

    monkeypatch.setattr(
        diag,
        "create_model",
        lambda params: FakeModel(),
    )

    monkeypatch.setattr(
        diag,
        "REPORT_DIR",
        tmp_path,
    )

    diag.main()

    output = capsys.readouterr().out

    assert "PAIRED CHANGES" in output

    assert "FOCAL ERROR CONFIDENCE" in output

    assert len(
        list(
            tmp_path.glob("*.csv")
        )
    ) == 6