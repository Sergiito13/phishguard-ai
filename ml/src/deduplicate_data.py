"""
PhishGuard AI - URL Deduplication

Removes duplicate URLs from cleaned datasets while preserving
their provenance.

Input:
    ml/data/processed/cleaned_urls.csv

Outputs:
    ml/data/processed/deduplicated_urls.csv
    ml/data/processed/conflicting_labels.csv
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "cleaned_urls.csv"

OUTPUT_FILE = PROCESSED_DIR / "deduplicated_urls.csv"

CONFLICTS_FILE = PROCESSED_DIR / "conflicting_labels.csv"


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

def load_data():

    print("\nLoading cleaned dataset...")

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "url",
            "label",
            "source",
            "source_row",
        ],
        dtype={
            "url": "string",
            "source": "string",
        },
    )

    # Ensure input meets the expected contract
    if df["url"].isna().any() or df["url"].eq("").any():
        raise ValueError("Input contains missing URLs")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Input contains invalid labels")

    print(f"Loaded records: {len(df):,}")

    return df


# ============================================================
# 3. DEDUPLICATE URLs
# ============================================================

def deduplicate_urls(df):

    print("\nAnalyzing duplicate URLs...")

    # Aggregate records by exact URL
    summary = (
        df.groupby("url", sort=False)
        .agg(
            label=("label", "first"),
            occurrences=("label", "size"),
            different_labels=("label", "nunique"),
            source_count=("source", "nunique"),
            first_source=("source", "first"),
        )
        .reset_index()
    )

    # Identify conflicting labels
    conflict_urls = summary.loc[
        summary["different_labels"] > 1,
        "url",
    ]

    # Preserve all original records involved in conflicts
    conflicts = df[
        df["url"].isin(conflict_urls)
    ].copy()

    # Exclude conflicting URLs from the final dataset
    unique = summary[
        summary["different_labels"] == 1
    ].copy()

    # Identify URLs that appear in multiple sources
    source_pairs = df[
        ["url", "source"]
    ].drop_duplicates()

    multiple_sources = source_pairs[
        source_pairs.duplicated(
            "url",
            keep=False,
        )
    ]

    # Join source names only for URLs shared by datasets
    source_lists = (
        multiple_sources.groupby("url")["source"]
        .agg(lambda values: "|".join(sorted(values)))
    )

    # Single-source URLs retain their original source name
    unique["sources"] = (
        unique["url"]
        .map(source_lists)
        .fillna(unique["first_source"])
    )

    # Select final output columns
    unique = unique[
        [
            "url",
            "label",
            "occurrences",
            "source_count",
            "sources",
        ]
    ]

    print(f"Original records: {len(df):,}")
    print(f"Unique URLs detected: {len(summary):,}")

    print(
        "Duplicate occurrences: "
        f"{len(df) - len(summary):,}"
    )

    print(
        "Conflicting URLs: "
        f"{len(conflict_urls):,}"
    )

    print(
        "Final unique records: "
        f"{len(unique):,}"
    )

    return unique, conflicts


# ============================================================
# 4. SAVE RESULTS
# ============================================================

def save_results(unique, conflicts):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    conflicts.to_csv(
        CONFLICTS_FILE,
        index=False,
    )

    print("\nFiles generated:")
    print(f"- {OUTPUT_FILE}")
    print(f"- {CONFLICTS_FILE}")


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - URL DEDUPLICATION")
    print("=" * 60)

    df = load_data()

    unique, conflicts = deduplicate_urls(df)

    save_results(unique, conflicts)

    print("\nFinal label distribution:")
    print(unique["label"].value_counts())

    print("\nDeduplication completed successfully.")


if __name__ == "__main__":
    main()