"""
PhishGuard AI - Initial Data Cleaning

Cleans and harmonizes the four original URL datasets.

Does not modify raw datasets.
Does not remove duplicate URLs.

Outputs:
    ml/data/processed/cleaned_urls.csv
    ml/data/processed/quarantine_urls.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "legitphish": {
        "file": "url_features_extracted1.csv",
        "url_column": "URL",
        "label_column": "ClassLabel",
        "invert_labels": True,
    },
    "phiusiil": {
        "file": "PhiUSIIL_Phishing_URL_Dataset.csv",
        "url_column": "URL",
        "label_column": "label",
        "invert_labels": True,
    },
    "url_phish": {
        "file": "Dataset.csv",
        "url_column": "url",
        "label_column": "label",
        "invert_labels": False,
    },
    "phishbd": {
        "file": "PhishBD_2026_73_features.csv",
        "url_column": "url",
        "label_column": "label",
        "invert_labels": False,
    },
}


# Domain-like URL without a required HTTP(S) scheme
DOMAIN_PATTERN = (
    r"^(?:[a-z0-9-]+\.)+"
    r"[a-z0-9-]+(?:$|[/:?#])"
)


# ============================================================
# 3. CLEAN INDIVIDUAL DATASET
# ============================================================

def clean_dataset(name: str, config: dict):

    file_path = RAW_DIR / config["file"]

    print(f"\nCleaning dataset: {name}")

    df = pd.read_csv(
        file_path,
        usecols=[
            config["url_column"],
            config["label_column"],
        ],
        low_memory=False,
    )

    # Preserve original URLs for traceability
    original_urls = df[
        config["url_column"]
    ].astype("string")

    # Remove exterior whitespace
    urls = original_urls.str.strip().copy()

    # Detect recoverable leading apostrophes
    without_quote = urls.str.slice(start=1)

    recoverable_quote = (
        urls.str.startswith("'", na=False)
        & without_quote.str.match(
            DOMAIN_PATTERN,
            case=False,
            na=False,
        )
    )

    urls.loc[recoverable_quote] = (
        without_quote.loc[recoverable_quote]
    )

    # Read original labels
    labels = pd.to_numeric(
        df[config["label_column"]],
        errors="coerce",
    )

    # Standard convention:
    # 0 = legitimate
    # 1 = phishing
    if config["invert_labels"]:
        labels = 1 - labels

    labels = labels.astype("Int64")

    # Validate URL presence
    missing_url = (
        urls.isna()
        | urls.eq("").fillna(False)
    )

    # Accept HTTP(S) or recognizable scheme-less domains
    has_http = urls.str.match(
        r"^https?://",
        case=False,
        na=False,
    )

    domain_like = urls.str.match(
        DOMAIN_PATTERN,
        case=False,
        na=False,
    )

    supported_format = has_http | domain_like

    # Validate labels
    invalid_label = ~labels.isin([0, 1])

    # Explain why a record is rejected
    reasons = pd.Series(
        "",
        index=df.index,
        dtype="string",
    )

    reasons.loc[
        ~missing_url & ~supported_format
    ] = "unsupported_url_format"

    reasons.loc[
        missing_url
    ] = "missing_url"

    reasons.loc[
        invalid_label
    ] = "invalid_label"

    # Build traceable dataset
    result = pd.DataFrame({
        "source": name,
        "source_row": df.index + 2,
        "original_url": original_urls,
        "url": urls,
        "label": labels,
        "cleaning_action": recoverable_quote.map({
            True: "remove_leading_apostrophe",
            False: "trim_only",
        }),
        "reason": reasons,
    })

    # Separate accepted and rejected records
    accepted = result[
        result["reason"] == ""
    ].copy()

    rejected = result[
        result["reason"] != ""
    ].copy()

    print(f"Original rows: {len(df):,}")
    print(f"Accepted: {len(accepted):,}")
    print(f"Quarantined: {len(rejected):,}")

    print(
        "Recovered leading apostrophes: "
        f"{recoverable_quote.sum():,}"
    )

    return accepted, rejected


# ============================================================
# 4. PROCESS ALL DATASETS
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - DATA CLEANING")
    print("=" * 60)

    cleaned_datasets = []
    quarantined_datasets = []

    for name, config in DATASETS.items():

        accepted, rejected = clean_dataset(
            name,
            config,
        )

        cleaned_datasets.append(accepted)
        quarantined_datasets.append(rejected)

    # Combine all four sources
    cleaned = pd.concat(
        cleaned_datasets,
        ignore_index=True,
    )

    quarantine = pd.concat(
        quarantined_datasets,
        ignore_index=True,
    )

    # Create output directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save cleaned records
    cleaned.to_csv(
        PROCESSED_DIR / "cleaned_urls.csv",
        index=False,
    )

    # Save rejected records for review
    quarantine.to_csv(
        PROCESSED_DIR / "quarantine_urls.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"Accepted records: {len(cleaned):,}")
    print(f"Quarantined records: {len(quarantine):,}")

    print("\nNormalized labels:")
    print(cleaned["label"].value_counts())

    print("\nQuarantine reasons:")
    print(quarantine["reason"].value_counts())

    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()