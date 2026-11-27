"""Synthetic, offline regression tests for the path-attrition audit."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audit_pipeline_path_attrition as module  # noqa: E402


@pytest.mark.parametrize("url, expected", [
    ("https://site.test", 0),
    ("https://site.test/", 0),
    ("https://site.test/catalog/123?x=1", len("/catalog/123")),
    ("site.test/articles", len("/articles")),
    ("'site.test/products", len("/products")),
    ("http://site.test/?page=2", 0),
    ("https://site.test/login#fragment", len("/login")),
    ("ftp://site.test/path", None),
    ("not a website", None),
    (None, None),
    ("https://site.test:bad/path", None),
])
def test_path_detection(url, expected):
    assert module.url_path_length(url) == expected


def write_csv(base, relative, rows, columns):
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(target, index=False)


@pytest.fixture
def pipeline(tmp_path):
    datasets = {
        "legitphish": [
            ("https://a.example/path", 1),
            ("https://a.example/path", 1),
            ("https://shared.example/ok", 1),
            ("https://conflict.example/path", 1),
            ("https://conflict.example/path", 0),
            ("https://root.example/", 1),
            ("ftp://ftp.example/something", 1),
            ("https://invalid.example/path", "no_label"),
        ],
        "phiusiil": [
            ("https://phi.example/", 1),
            ("https://phi.example/login", 0),
            ("https://nogroup.example/x", 1),
        ],
        "phishbd": [
            ("https://bd.example/articles", 0),
            ("https://bd.example/login", 1),
        ],
        "url_phish": [
            ("https://shared.example/ok", 0),
            ("https://url.example/post", 0),
            ("https://url.example/phish", 1),
        ],
    }
    for src, records in datasets.items():
        config = module.DATASETS[src]
        write_csv(tmp_path, f"data/raw/{config['file']}", records,
                  [config["url_column"], config["label_column"]])

    clean = [
        ("https://a.example/path", 0, "legitphish"),
        ("https://a.example/path", 0, "legitphish"),
        ("https://shared.example/ok", 0, "legitphish"),
        ("https://conflict.example/path", 0, "legitphish"),
        ("https://conflict.example/path", 1, "legitphish"),
        ("https://root.example/", 0, "legitphish"),
        ("https://phi.example/", 0, "phiusiil"),
        ("https://phi.example/login", 1, "phiusiil"),
        ("https://nogroup.example/x", 0, "phiusiil"),
        ("https://bd.example/articles", 0, "phishbd"),
        ("https://bd.example/login", 1, "phishbd"),
        ("https://shared.example/ok", 0, "url_phish"),
        ("https://url.example/post", 0, "url_phish"),
        ("https://url.example/phish", 1, "url_phish"),
    ]
    write_csv(tmp_path, "data/processed/cleaned_urls.csv", clean,
              ["url", "label", "source"])
    quarantine = [
        ("ftp://ftp.example/something", 0, "legitphish", "unsupported_url_format"),
        ("https://invalid.example/path", "", "legitphish", "invalid_label"),
    ]
    write_csv(tmp_path, "data/processed/quarantine_urls.csv", quarantine,
              ["url", "label", "source", "reason"])
    dedup = [
        ("https://a.example/path", 0, "legitphish"),
        ("https://shared.example/ok", 0, "legitphish|url_phish"),
        ("https://root.example/", 0, "legitphish"),
        ("https://phi.example/", 0, "phiusiil"),
        ("https://phi.example/login", 1, "phiusiil"),
        ("https://nogroup.example/x", 0, "phiusiil"),
        ("https://bd.example/articles", 0, "phishbd"),
        ("https://bd.example/login", 1, "phishbd"),
        ("https://url.example/post", 0, "url_phish"),
        ("https://url.example/phish", 1, "url_phish"),
    ]
    write_csv(tmp_path, "data/processed/deduplicated_urls.csv", dedup,
              ["url", "label", "sources"])
    write_csv(tmp_path, "data/processed/conflicting_labels.csv", [
        ("https://conflict.example/path", 0, "legitphish"),
        ("https://conflict.example/path", 1, "legitphish"),
    ], ["url", "label", "source"])
    group_rows = [
        (*r, "" if r[0] == "https://nogroup.example/x"
         else "domain:" + r[0].split("//")[1].split("/")[0])
        for r in dedup
    ]
    write_csv(tmp_path, "data/processed/domain_groups.csv", group_rows,
              ["url", "label", "sources", "group_key"])
    selected = [r for r in group_rows if r[0] in {
        "https://a.example/path", "https://phi.example/",
        "https://phi.example/login", "https://bd.example/articles",
        "https://url.example/post", "https://url.example/phish",
    }]
    write_csv(tmp_path, "data/splits/train.csv", selected,
              ["url", "label", "sources", "group_key"])
    v2 = [(*r, module.url_path_length(r[0])) for r in selected]
    write_csv(tmp_path, "data/splits/v2/train.csv", v2,
              ["url", "label", "sources", "group_key", "v2_path_length"])
    return tmp_path


def test_audit_expected_shapes_and_provenance(pipeline):
    reports = module.audit(pipeline)
    assert set(reports) == set(module.REPORT_FILES)
    assert len(reports["coverage"]) == len(module.STAGES) * len(module.SOURCES) * 2
    assert len(reports["transitions"]) == 6 * len(module.SOURCES) * 2
    row = reports["coverage"].query(
        'stage == "raw_valid_labels" and source == "legitphish" and label == 0'
    ).iloc[0]
    assert row["records"] == 6  # includes the rejected non-HTTP URL
    row = reports["coverage"].query(
        'stage == "deduplicated" and source == "url_phish" and label == 0'
    ).iloc[0]
    assert row["path_records"] == 2  # includes the shared multi-source URL
    assert int(reports["integrity"]["anomalies"].sum()) == 0


def test_clean_to_dedup_unique_not_row_count(pipeline):
    reports = module.audit(pipeline)
    row = reports["transitions"].query(
        'transition == "cleaned_to_deduplicated" and source == "legitphish" and label == 0'
    ).iloc[0]
    assert row["input_with_path"] == 3  # a, shared, conflict (a duplicate counted once)
    assert row["output_with_path"] == 2  # conflict removed, shared retained
    assert row["difference"] == 1
    assert row["count_basis"] == "unique_urls"


def test_quarantine_and_ungrouped_explanations(pipeline):
    report = module.audit(pipeline)["excluded"]
    assert len(report.loc[report["reason"] == "unsupported_url_format"]) == 1
    assert len(report.loc[report["reason"] == "conflicting_label"]) == 2
    assert len(report.loc[report["reason"] == "missing_group_key"]) == 1
    invalid = report.query('source == "legitphish" and reason == "raw_invalid_label_total"')
    assert invalid.iloc[0]["records"] == 1


def test_heldout_train_does_not_mean_deleted(pipeline):
    rows = module.audit(pipeline)["transitions"]
    chosen = rows.query('transition == "eligible_to_train"')
    assert chosen["interpretation"].eq("reserved_or_other_split_not_lost").all()


def test_v2_path_mismatch_detected(pipeline):
    file = pipeline / "data/splits/v2/train.csv"
    frame = pd.read_csv(file)
    frame.loc[frame["url"] == "https://a.example/path", "v2_path_length"] = 0
    frame.to_csv(file, index=False)
    integrity = module.audit(pipeline)["integrity"]
    mismatches = integrity.loc[
        integrity["check"] == "v2_path_length_matches_parsed_train_url", "anomalies"
    ]
    assert mismatches.iloc[0] == 1


def test_metadata_inconsistency_rejected(pipeline):
    file = pipeline / "data/processed/domain_groups.csv"
    frame = pd.read_csv(file, keep_default_na=False)
    frame.loc[0, "label"] = 1
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="metadata change"):
        module.audit(pipeline)


def test_unexpected_record_rejected(pipeline):
    file = pipeline / "data/splits/train.csv"
    frame = pd.read_csv(file)
    frame.loc[0, "url"] = "https://unknown.example/path"
    frame.to_csv(file, index=False)
    # TRAIN and V2 metadata disagree, triggering a fail-closed audit.
    with pytest.raises(ValueError):
        module.audit(pipeline)


def test_read_only_no_reserved_split_access(pipeline, monkeypatch):
    original = module.pd.read_csv
    observed = []

    def monitored_read(path, *args, **kwargs):
        name = str(path).lower().replace("\\", "/")
        assert "validation.csv" not in name
        assert "test.csv" not in name
        observed.append(name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.pd, "read_csv", monitored_read)
    reports = module.audit(pipeline)
    assert reports["integrity"].iloc[-1]["anomalies"] == 0
    assert len(observed) == 4 + 7


def test_no_url_or_group_identifier_exported(pipeline, tmp_path):
    reports = module.audit(pipeline)
    directory = tmp_path / "aggregate_reports"
    module.save_reports(reports, directory)
    files = sorted(directory.glob("*.csv"))
    assert len(files) == 4
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert "https://" not in content
        assert "domain:" not in content
        assert "shared.example" not in content
        assert "group_key" not in pd.read_csv(path, nrows=0).columns


def test_disallow_sensitive_report_columns(pipeline, tmp_path):
    reports = module.audit(pipeline)
    reports["coverage"]["url"] = "https://private.example"
    dest = tmp_path / "reports"
    with pytest.raises(ValueError, match="identifying"):
        module.save_reports(reports, dest)
    assert not dest.exists()


def test_missing_required_input_fails(pipeline):
    (pipeline / "data/splits/train.csv").unlink()
    with pytest.raises(FileNotFoundError, match="train.csv"):
        module.audit(pipeline)


def test_inconsistent_raw_clean_totals_fails(pipeline):
    file = pipeline / "data/processed/cleaned_urls.csv"
    frame = pd.read_csv(file)
    frame = frame.iloc[1:]
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="totals inconsistent"):
        module.audit(pipeline)


def test_duplicate_membership_rejected():
    frame = pd.DataFrame({
        "url": ["https://a.example/x"], "label": [0],
        "sources": ["legitphish|legitphish"],
    })
    with pytest.raises(ValueError, match="duplicated source"):
        module.expand_stage(frame, single_source=False)
