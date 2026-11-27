"""Synthetic, offline tests for the V2 training-only legitimate path audit."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_legitimate_path_coverage as audit


@pytest.fixture
def training():
    rows = []
    for source_id, source in enumerate(audit.SOURCES):
        for label in (0, 1):
            for i in range(5):
                rows.append({
                    "sources": source,
                    "label": label,
                    "group_key": f"domain:{source}-{label}-{i}",
                    "v2_path_length": 0 if i < 2 else 4 + i,
                    "v2_path_depth": 0 if i < 2 else (1 if i < 4 else 3),
                    "v2_query_length": 5 if i == 4 else 0,
                })
    # A row associated with two sources contributes fractional provenance.
    rows[2]["sources"] = "legitphish|phiusiil"
    # A non-held-source domain that overlaps a held source must be excluded.
    rows[15]["group_key"] = rows[0]["group_key"]
    return pd.DataFrame(rows)


@pytest.fixture
def reports(training):
    return audit.build_reports(training)


def test_configuration():
    assert len(audit.SOURCES) == 4
    assert len(audit.REPORT_FILES) == 3
    assert "group_key" in audit.REQUIRED_COLUMNS
    assert "url" not in audit.REQUIRED_COLUMNS


def test_valid_training(training):
    audit.validate_training(training)


def test_missing_required_column(training):
    with pytest.raises(ValueError, match="Missing required"):
        audit.validate_training(training.drop(columns="v2_path_length"))


def test_empty_data(training):
    with pytest.raises(ValueError, match="Empty"):
        audit.validate_training(training.iloc[:0])


def test_bad_labels(training):
    training.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="binary"):
        audit.validate_training(training)


@pytest.mark.parametrize("membership", ["unknown", "", "legitphish|legitphish", "legitphish|unknown"])
def test_bad_sources(training, membership):
    training.loc[0, "sources"] = membership
    with pytest.raises(ValueError, match="source membership"):
        audit.validate_training(training)


def test_blank_group(training):
    training.loc[0, "group_key"] = " "
    with pytest.raises(ValueError, match="domain group"):
        audit.validate_training(training)


def test_nonfinite_length(training):
    training["v2_path_length"] = training["v2_path_length"].astype(float)
    training.loc[0, "v2_path_length"] = np.inf
    with pytest.raises(ValueError, match="structural"):
        audit.validate_training(training)


def test_negative_depth(training):
    training.loc[0, "v2_path_depth"] = -1
    with pytest.raises(ValueError, match="structural"):
        audit.validate_training(training)


def test_read_only_required_columns(tmp_path, training):
    f = tmp_path / "train.csv"
    training["url"] = "https://secret.example/private"
    training.to_csv(f, index=False)
    loaded = audit.load_training(f)
    assert set(loaded.columns) == set(audit.REQUIRED_COLUMNS)
    assert len(loaded) == len(training)
    assert "url" not in loaded.columns


def test_report_shapes(reports):
    assert set(reports) == set(audit.REPORT_FILES)
    assert len(reports["by_source"]) == 8
    assert len(reports["by_fold"]) == 4
    assert len(reports["provenance"]) == 12


def test_source_coverage(reports):
    legit = reports["by_source"].query('source=="legitphish" and label==0').iloc[0]
    assert legit["samples"] == 5
    assert legit["nonempty_path_rows"] == 3
    assert legit["nonempty_path_domain_groups"] == 3
    assert legit["multi_source_rows"] == 1
    assert legit["deep_path_rows"] == 1


def test_group_concentration(training):
    n, top5 = audit.concentration(training.loc[training["label"] == 0, "group_key"])
    assert n > 0
    assert 0 < top5 <= 1
    assert audit.concentration(training.iloc[:0]["group_key"])[0] == 0


def test_fold_group_isolation(training):
    folds = audit.create_source_folds(training)
    f = next(f for f in folds if f["source"] == "legitphish")
    fit = training.iloc[f["training_indices"]]
    held = training.iloc[f["validation_indices"]]
    assert set(fit["group_key"]).isdisjoint(set(held["group_key"]))
    assert "domain:legitphish-0-0" not in set(fit["group_key"])
    assert not audit.belongs_to_source(fit, "legitphish").any()


def test_fractional_provenance_mass(reports):
    folds = reports["by_fold"].set_index("excluded_source")
    for held_source, group in reports["provenance"].groupby("excluded_source"):
        expected = folds.loc[held_source, "training_benign_path_rows"]
        assert group["fractional_rows"].sum() == pytest.approx(expected)
        assert (group["fractional_rows"] <= group["membership_rows"]).all()
    phiusiil = reports["provenance"].query('excluded_source=="phishbd"')
    assert (phiusiil["fractional_rows"] < phiusiil["membership_rows"]).any()


def test_focal_fold_nonempty_benign(training, reports):
    fold = reports["by_fold"].query('excluded_source=="url_phish"').iloc[0]
    assert fold["training_benign_path_rows"] > 0
    assert fold["held_out_benign_path_rows"] > 0
    assert fold["training_benign_path_domain_groups"] > 0


def test_no_raw_identifiers_exported(reports):
    for table in reports.values():
        assert audit.SENSITIVE_COLUMNS.isdisjoint(table.columns)


def test_reports_written(reports, monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "REPORT_DIR", tmp_path)
    audit.save_reports(reports)
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == sorted(audit.REPORT_FILES.values())


def test_reject_sensitive_report_before_writing(reports, monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "REPORT_DIR", tmp_path)
    altered = {name: table.copy() for name, table in reports.items()}
    altered["by_fold"]["group_key"] = "private:domain"
    with pytest.raises(ValueError, match="sensitive"):
        audit.save_reports(altered)
    assert not list(tmp_path.iterdir())
