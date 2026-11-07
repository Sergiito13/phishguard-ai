"""
PhishGuard AI - Dataset Auditing

Audits the original phishing URL datasets without modifying them.

Checks:
- Dataset dimensions
- Column names and data types
- Missing values
- Label distribution
- Duplicate URLs
- URL format
- Cross-dataset duplicates
- Conflicting labels

Output:
    ml/reports/data_audit.json
"""

import json

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "legitphish": {
        "file": "url_features_extracted1.csv",
        "url": "URL",
        "label": "ClassLabel",
        "phishing_value": 0,
    },

    "phiusiil": {
        "file": "PhiUSIIL_Phishing_URL_Dataset.csv",
        "url": "URL",
        "label": "label",
        "phishing_value": 0,
    },

    "url_phish": {
        "file": "Dataset.csv",
        "url": "url",
        "label": "label",
        "phishing_value": 1,
    },

    "phishbd": {
        "file": "PhishBD_2026_73_features.csv",
        "url": "url",
        "label": "label",
        "phishing_value": 1,
    },
}


# ============================================================
# 3. AUDIT INDIVIDUAL DATASET
# ============================================================

def audit_dataset(name: str, config: dict):

    file_path = RAW_DIR / config["file"]

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print(f"{'=' * 60}")

    # Load original CSV
    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    # Validate required columns
    required_columns = {
        config["url"],
        config["label"]
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {name}: {missing_columns}"
        )

    # Extract URL and label columns
    urls = (
        df[config["url"]]
        .astype("string")
        .str.strip()
    )

    labels = pd.to_numeric(
        df[config["label"]],
        errors="coerce"
    )

    # Validate URL values
    has_url = urls.notna() & urls.ne("")

    # Validate labels
    valid_label = labels.isin([0, 1])

    # Check HTTP / HTTPS scheme
    has_http = urls.str.match(
        r"^https?://",
        case=False,
        na=False
    )

    # Dataset statistics
    report = {
        "file": config["file"],

        "rows": int(len(df)),

        "columns": int(len(df.columns)),

        "column_names": df.columns.tolist(),

        "column_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "nulls_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },

        "missing_or_blank_urls": int(
            (~has_url).sum()
        ),

        "urls_without_http_scheme": int(
            (has_url & ~has_http).sum()
        ),

        "duplicated_urls_within_dataset": int(
            urls[has_url].duplicated().sum()
        ),

        "original_labels": {
            str(label): int(count)
            for label, count in (
                df[config["label"]]
                .value_counts(dropna=False)
                .items()
            )
        },

        "invalid_or_missing_labels": int(
            (~valid_label).sum()
        ),
    }

    # Harmonize labels in memory for comparison
    # 0 = legitimate
    # 1 = phishing

    if config["phishing_value"] == 0:
        normalized_labels = 1 - labels
    else:
        normalized_labels = labels

    valid_records = has_url & valid_label

    # Temporary DataFrame for cross-dataset auditing
    audit_pairs = pd.DataFrame({
        "url": urls[valid_records],

        "label": normalized_labels[
            valid_records
        ].astype(int),

        "source": name,
    })

    report["valid_url_label_pairs"] = int(
        valid_records.sum()
    )

    # Display summary
    print(f"Rows: {report['rows']:,}")
    print(f"Columns: {report['columns']}")

    print("\nOriginal label distribution:")
    print(report["original_labels"])

    print("\nData quality:")
    print(
        f"Missing URLs: "
        f"{report['missing_or_blank_urls']:,}"
    )

    print(
        f"Invalid labels: "
        f"{report['invalid_or_missing_labels']:,}"
    )

    print(
        f"Duplicate URLs: "
        f"{report['duplicated_urls_within_dataset']:,}"
    )

    print(
        f"URLs without HTTP(S): "
        f"{report['urls_without_http_scheme']:,}"
    )

    print(f"\nAudit completed: {name}")

    return report, audit_pairs


# ============================================================
# 4. AUDIT ALL DATASETS
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - DATASET AUDIT")
    print("=" * 60)

    reports = {}
    datasets = []

    # Audit each dataset
    for name, config in DATASETS.items():

        report, audit_pairs = audit_dataset(
            name,
            config
        )

        reports[name] = report
        datasets.append(audit_pairs)

    # Combine temporary URL-label pairs
    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    # Find conflicting labels across datasets
    label_counts = (
        combined.groupby("url")["label"]
        .nunique()
    )

    conflicting_urls = int(
        (label_counts > 1).sum()
    )

    # Count URLs appearing in multiple sources
    source_counts = (
        combined.groupby("url")["source"]
        .nunique()
    )

    shared_urls = int(
        (source_counts > 1).sum()
    )

    # General statistics
    overall = {
        "total_rows": sum(
            report["rows"]
            for report in reports.values()
        ),

        "valid_url_label_pairs": int(
            len(combined)
        ),

        "distinct_urls_exact_after_trimming": int(
            combined["url"].nunique()
        ),

        "duplicate_occurrences_across_all_datasets": int(
            combined.duplicated("url").sum()
        ),

        "urls_present_in_multiple_sources": shared_urls,

        "urls_with_conflicting_labels": conflicting_urls,

        "normalized_label_counts_all_rows": {
            str(label): int(count)
            for label, count in (
                combined["label"]
                .value_counts()
                .items()
            )
        },
    }

    # Create JSON report
    final_report = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "datasets": reports,

        "overall": overall,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        REPORT_DIR / "data_audit.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Display final results
    print("\n" + "=" * 60)
    print("OVERALL DATASET STATISTICS")
    print("=" * 60)

    for key, value in overall.items():
        print(f"{key}: {value}")

    print(
        f"\nAudit report saved: {output_file}"
    )

    print("\nDataset audit completed successfully.")


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()