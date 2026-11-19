"""
PhishGuard AI - Source Quality Audit Tests

Unit tests using synthetic data.

No real datasets or model artifacts are modified.
"""

import sys
from pathlib import Path

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

import audit_source_quality as audit


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def sample_data():

    rows = []

    for source in audit.SOURCES:

        for label in (0, 1):

            for idx in range(4):

                scheme = (
                    "https"
                    if label == 0
                    else "http"
                )

                url = (
                    f"{scheme}://"
                    f"{source}-{label}-{idx}.example"
                    "/login?a=1"
                )

                rows.append({
                    "url": url,

                    "label": label,

                    "sources": source,

                    "group_key": (
                        f"domain:"
                        f"{source}-{label}-{idx}.example"
                    ),

                    "domain_status": "ok",

                    "is_https": int(
                        scheme == "https"
                    ),

                    "has_ip": 0,

                    "url_length": len(url),

                    "slash_count": url.count("/"),
                })

    # One URL belongs to two sources.

    rows[0]["sources"] = (
        f"{audit.SOURCES[0]}|"
        f"{audit.SOURCES[1]}"
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 3. CONFIGURATION TESTS
# ============================================================

def test_configuration():

    assert len(audit.SOURCES) == 4

    assert len(audit.REPORT_NAMES) == 4


def test_valid_data(sample_data):

    audit.validate_training_data(
        sample_data
    )


# ============================================================
# 4. DATA VALIDATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "column",
    list(audit.REQUIRED),
)
def test_missing_column(
    sample_data,
    column,
):

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        audit.validate_training_data(
            sample_data.drop(columns=column)
        )


def test_duplicate_url(sample_data):

    broken = pd.concat(
        [
            sample_data,
            sample_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate URLs",
    ):
        audit.validate_training_data(
            broken
        )


def test_invalid_label(sample_data):

    sample_data.loc[0, "label"] = 2

    with pytest.raises(
        ValueError,
        match="Invalid labels",
    ):
        audit.validate_training_data(
            sample_data
        )


def test_missing_group_key(sample_data):

    sample_data.loc[0, "group_key"] = ""

    with pytest.raises(
        ValueError,
        match="group keys",
    ):
        audit.validate_training_data(
            sample_data
        )


def test_unknown_source(sample_data):

    sample_data.loc[0, "sources"] = "unknown"

    with pytest.raises(
        ValueError,
        match="Unknown",
    ):
        audit.validate_training_data(
            sample_data
        )


def test_duplicate_membership(sample_data):

    sample_data.loc[0, "sources"] = (
        f"{audit.SOURCES[0]}|"
        f"{audit.SOURCES[0]}"
    )

    with pytest.raises(
        ValueError,
        match="Duplicate source membership",
    ):
        audit.validate_training_data(
            sample_data
        )


def test_empty_url(sample_data):

    sample_data.loc[0, "url"] = "   "

    with pytest.raises(
        ValueError,
        match="Empty URLs",
    ):
        audit.validate_training_data(
            sample_data
        )


# ============================================================
# 5. REPORT GENERATION
# ============================================================

def test_build_reports(sample_data):

    reports = audit.build_reports(
        sample_data
    )

    assert set(reports) == set(
        audit.REPORT_NAMES
    )

    assert len(
        reports["overview"]
    ) == len(audit.SOURCES)

    assert len(
        reports["structure"]
    ) == 2 * len(audit.SOURCES)

    integrity = reports["integrity"].iloc[0]

    assert integrity["training_rows"] == len(
        sample_data
    )

    assert integrity["source_memberships"] == (
        len(sample_data) + 1
    )

    assert integrity["urls_in_multiple_sources"] == 1


# ============================================================
# 6. SCHEME ANALYSIS TESTS
# ============================================================

def test_scheme_fractions_sum_to_one(
    sample_data,
):

    report = audit.build_reports(
        sample_data
    )["scheme"]

    sums = report.groupby(
        [
            "source",
            "label",
        ]
    )["fraction_within_class"].sum()

    assert (
        sums - 1
    ).abs().lt(1e-10).all()


def test_scheme_bias_by_label(
    sample_data,
):

    structure = audit.build_reports(
        sample_data
    )["structure"]

    assert structure.loc[
        structure["label"] == 0,
        "https_fraction",
    ].eq(1).all()

    assert structure.loc[
        structure["label"] == 1,
        "https_fraction",
    ].eq(0).all()


# ============================================================
# 7. FEATURE CONSISTENCY TESTS
# ============================================================

def test_feature_mismatches_are_detected(
    sample_data,
):

    sample_data.loc[0, "is_https"] = 0

    sample_data.loc[0, "url_length"] += 2

    sample_data.loc[0, "slash_count"] += 1

    integrity = audit.build_reports(
        sample_data
    )["integrity"].iloc[0]

    assert integrity[
        "https_feature_mismatches"
    ] == 1

    assert integrity[
        "length_feature_mismatches"
    ] == 1

    assert integrity[
        "slash_feature_mismatches"
    ] == 1


# ============================================================
# 8. DOMAIN GROUP TESTS
# ============================================================

def test_mixed_label_group_is_not_a_conflict(
    sample_data,
):

    sample_data.loc[
        1,
        "group_key",
    ] = sample_data.loc[
        0,
        "group_key",
    ]

    integrity = audit.build_reports(
        sample_data
    )["integrity"].iloc[0]

    # Two legitimate URLs share one group.

    assert integrity[
        "mixed_label_group_keys"
    ] == 0

    # Add a phishing URL to the same group.

    sample_data.loc[
        4,
        "group_key",
    ] = sample_data.loc[
        0,
        "group_key",
    ]

    integrity = audit.build_reports(
        sample_data
    )["integrity"].iloc[0]

    assert integrity[
        "mixed_label_group_keys"
    ] == 1


# ============================================================
# 9. PRIVACY TESTS
# ============================================================

def test_sensitive_columns_never_exported(
    sample_data,
):

    reports = audit.build_reports(
        sample_data
    )

    forbidden = {
        "url",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    for frame in reports.values():

        assert forbidden.isdisjoint(
            frame.columns
        )


# ============================================================
# 10. FILE GENERATION TESTS
# ============================================================

def test_save_reports(
    sample_data,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        audit,
        "REPORT_DIR",
        tmp_path,
    )

    reports = audit.build_reports(
        sample_data
    )

    audit.save_reports(
        reports
    )

    for filename in audit.REPORT_NAMES.values():

        path = tmp_path / filename

        assert path.is_file()

        assert not pd.read_csv(
            path
        ).empty


def test_reject_sensitive_columns(
    sample_data,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        audit,
        "REPORT_DIR",
        tmp_path,
    )

    reports = audit.build_reports(
        sample_data
    )

    reports["overview"]["url"] = (
        "private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):
        audit.save_reports(
            reports
        )

    assert not list(
        tmp_path.glob("*.csv")
    )


# ============================================================
# 11. TRAINING DATA LOADING
# ============================================================

def test_read_training_only(
    sample_data,
    tmp_path,
):

    path = tmp_path / "train.csv"

    sample_data.to_csv(
        path,
        index=False,
    )

    result = audit.load_training_data(
        path
    )

    assert len(result) == len(
        sample_data
    )


# ============================================================
# 12. COMPLETE WORKFLOW
# ============================================================

def test_main(
    sample_data,
    tmp_path,
    monkeypatch,
    capsys,
):

    path = tmp_path / "train.csv"

    sample_data.to_csv(
        path,
        index=False,
    )

    monkeypatch.setattr(
        audit,
        "TRAIN_FILE",
        path,
    )

    monkeypatch.setattr(
        audit,
        "REPORT_DIR",
        tmp_path / "reports",
    )

    audit.main()

    output = capsys.readouterr().out

    assert "SOURCE OVERVIEW" in output

    assert len(
        list(
            (tmp_path / "reports").glob("*.csv")
        )
    ) == 4