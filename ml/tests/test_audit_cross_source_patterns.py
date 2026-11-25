"""Synthetic tests; never need real datasets or network access."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audit_cross_source_patterns as audit


@pytest.fixture
def sample():
    rows = []
    for i, source in enumerate(audit.SOURCES):
        for label in (0, 1):
            value = i + label
            rows.append({
                "sources": source,
                "label": label,
                "v2_url_length": 18 + 10 * value,
                "v2_hostname_length": 10 + value,
                "v2_path_length": 0 if value == 0 else 3 + value,
                "v2_path_depth": 0 if value == 0 else value,
                "v2_path_entropy": float(value),
                "v2_url_entropy": 3.0 + value / 10,
                "v2_hostname_hyphen_count": value % 2,
                "v2_slash_count": value,
                "v2_query_length": value,
            })
    return pd.DataFrame(rows)


def test_valid_data(sample):
    audit.validate_training(sample)


def test_reject_missing_feature(sample):
    with pytest.raises(ValueError, match="Missing required columns"):
        audit.validate_training(sample.drop(columns=["v2_path_depth"]))


def test_reject_bad_labels(sample):
    sample.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="binary"):
        audit.validate_training(sample)


@pytest.mark.parametrize("value", ["", "unknown", "phiusiil|phiusiil", "phiusiil|unknown"])
def test_reject_invalid_memberships(sample, value):
    sample.loc[0, "sources"] = value
    with pytest.raises(ValueError, match="source membership"):
        audit.validate_training(sample)


def test_reject_infinite_features(sample):
    sample["v2_path_entropy"] = sample["v2_path_entropy"].astype(float)
    sample.loc[0, "v2_path_entropy"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        audit.validate_training(sample)


def test_report_structure(sample):
    reports = audit.describe_patterns(sample)
    assert set(reports) == set(audit.REPORT_FILES)
    assert len(reports["counts"]) == 8
    assert len(reports["patterns"]) == 8 * 7
    assert len(reports["shifts"]) == 2 * len(audit.FEATURES)
    assert all(not frame.empty for frame in reports.values())


def test_source_label_counts(sample):
    result = audit.describe_patterns(sample)["counts"]
    assert (result["samples"] == 1).all()
    assert (result["class_fraction_within_source"] == 0.5).all()


def test_pattern_rates(sample):
    rows = audit.describe_patterns(sample)["patterns"]
    target = rows.loc[
        (rows["source"] == "legitphish") &
        (rows["label"] == 0) &
        (rows["pattern"] == "root_or_empty_path")
    ]
    assert target.iloc[0]["fraction"] == pytest.approx(1.0)


def test_overlapping_memberships_count_each_source(sample):
    sample.loc[0, "sources"] = "legitphish|phiusiil"
    rows = audit.describe_patterns(sample)["counts"]
    phiusiil_zero = rows.loc[
        (rows["source"] == "phiusiil") & (rows["label"] == 0)
    ].iloc[0]
    assert phiusiil_zero["samples"] == 2
    assert phiusiil_zero["source_total"] == 3


def test_shift_reference_excludes_target_source(sample):
    shifts = audit.describe_patterns(sample)["shifts"]
    row = shifts.loc[
        (shifts["source"] == "url_phish") &
        (shifts["label"] == 0) &
        (shifts["feature"] == "v2_url_length")
    ].iloc[0]
    assert row["source_samples"] == 1
    assert row["other_sources_samples"] == 3
    assert row["source_mean"] == 48
    assert row["other_sources_mean"] == pytest.approx((18+28+38)/3)
    assert row["standardized_difference"] > 0


def test_reports_do_not_export_url_or_group_keys(sample):
    reports = audit.describe_patterns(sample)
    forbidden = {"url", "urls", "sources", "group_key", "domain"}
    assert all(forbidden.isdisjoint(f.columns) for f in reports.values())


def test_reject_private_report(sample, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPORT_DIR", tmp_path)
    reports = audit.describe_patterns(sample)
    reports["counts"]["url"] = "https://private.example"
    with pytest.raises(ValueError, match="sensitive"):
        audit.save_reports(reports)
    assert list(tmp_path.iterdir()) == []


def test_read_and_save_end_to_end(sample, tmp_path, monkeypatch):
    dataset = tmp_path / "data" / "splits" / "v2" / "train.csv"
    dataset.parent.mkdir(parents=True)
    sample.assign(url="https://should-not-be-loaded.example").to_csv(dataset, index=False)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(audit, "TRAIN_FILE", dataset)
    monkeypatch.setattr(audit, "REPORT_DIR", reports_dir)
    audit.main()
    assert len(list(reports_dir.glob("cross_source_audit_*.csv"))) == 3
    for path in reports_dir.glob("*.csv"):
        content = path.read_text(encoding="utf-8")
        assert "should-not-be-loaded" not in content
