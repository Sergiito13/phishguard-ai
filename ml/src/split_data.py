"""
PhishGuard AI - Group-Aware Dataset Splitting

Splits processed URL data into:

- Training: approximately 70%
- Validation: approximately 15%
- Test: approximately 15%

URLs belonging to the same group_key are always assigned
to the same dataset split.

URLs without a grouping key are excluded from this first
training experiment and saved separately.

The output percentages are approximate because splitting
is performed by groups rather than individual URLs.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "url_features.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "splits"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

EXCLUDED_FILE = OUTPUT_DIR / "excluded_ungrouped.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

REQUIRED_COLUMNS = [
    "url",
    "label",
    "sources",
    "group_key",
    "domain_status",
]


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
        dtype={
            "url": "string",
            "sources": "string",
            "group_key": "string",
            "domain_status": "string",
        },
    )

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 or 1"
        )

    if df["url"].duplicated().any():
        raise ValueError(
            "Duplicate URLs found in processed dataset"
        )

    return df


# ============================================================
# 4. SEPARATE UNGROUPED URLS
# ============================================================

def separate_ungrouped(df: pd.DataFrame):

    has_group = df["group_key"].ne("")

    eligible = (
        df.loc[has_group]
        .reset_index(drop=True)
    )

    excluded = (
        df.loc[~has_group]
        .reset_index(drop=True)
    )

    return eligible, excluded


# ============================================================
# 5. GROUP-AWARE SPLITTING
# ============================================================

def split_dataset(df: pd.DataFrame):

    # --------------------------------------------------------
    # FIRST SPLIT: TRAIN + VALIDATION VS TEST
    # --------------------------------------------------------

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_validation_idx, test_idx = next(
        first_split.split(
            X=df,
            y=df["label"],
            groups=df["group_key"],
        )
    )

    train_validation = (
        df.iloc[train_validation_idx]
        .reset_index(drop=True)
    )

    test = (
        df.iloc[test_idx]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # SECOND SPLIT: TRAIN VS VALIDATION
    # --------------------------------------------------------

    # Validation should represent approximately 15%
    # of the original dataset, not 15% of the remainder.

    relative_validation_size = (
        VALIDATION_SIZE / (1 - TEST_SIZE)
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE + 1,
    )

    train_idx, validation_idx = next(
        second_split.split(
            X=train_validation,
            y=train_validation["label"],
            groups=train_validation["group_key"],
        )
    )

    train = (
        train_validation.iloc[train_idx]
        .reset_index(drop=True)
    )

    validation = (
        train_validation.iloc[validation_idx]
        .reset_index(drop=True)
    )

    return train, validation, test


# ============================================================
# 6. VERIFY GROUP ISOLATION
# ============================================================

def verify_group_isolation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):

    train_groups = set(train["group_key"])
    validation_groups = set(validation["group_key"])
    test_groups = set(test["group_key"])

    if train_groups & validation_groups:
        raise ValueError(
            "Group leakage between train and validation"
        )

    if train_groups & test_groups:
        raise ValueError(
            "Group leakage between train and test"
        )

    if validation_groups & test_groups:
        raise ValueError(
            "Group leakage between validation and test"
        )

    print("\nGroup isolation verification: PASSED")

    print(
        "No group_key appears in more than one split."
    )


# ============================================================
# 7. DISPLAY CLASS DISTRIBUTION
# ============================================================

def print_split_statistics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):

    datasets = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    total = sum(
        len(dataset)
        for dataset in datasets.values()
    )

    print("\n1. SPLIT DISTRIBUTION")

    print(
        f"{'Split':<15}"
        f"{'URLs':>12}"
        f"{'Percent':>12}"
        f"{'Groups':>12}"
    )

    print("-" * 51)

    for name, dataset in datasets.items():

        percentage = (
            len(dataset) / total * 100
        )

        groups = dataset["group_key"].nunique()

        print(
            f"{name:<15}"
            f"{len(dataset):>12,}"
            f"{percentage:>11.2f}%"
            f"{groups:>12,}"
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n2. CLASS DISTRIBUTION")

    print(
        f"{'Split':<15}"
        f"{'Legitimate':>14}"
        f"{'Phishing':>14}"
        f"{'Phishing %':>14}"
    )

    print("-" * 57)

    for name, dataset in datasets.items():

        legitimate = (
            dataset["label"] == 0
        ).sum()

        phishing = (
            dataset["label"] == 1
        ).sum()

        phishing_pct = (
            phishing / len(dataset) * 100
        )

        print(
            f"{name:<15}"
            f"{legitimate:>14,}"
            f"{phishing:>14,}"
            f"{phishing_pct:>13.2f}%"
        )

    # --------------------------------------------------------
    # SOURCE DISTRIBUTION
    # --------------------------------------------------------

    print("\n3. DATASET SOURCE DISTRIBUTION")

    for name, dataset in datasets.items():

        expanded = (
            dataset["sources"]
            .str.split("|", regex=False)
            .explode()
        )

        print(f"\n{name.upper()}")

        print(
            expanded.value_counts().to_string()
        )


# ============================================================
# 8. SAVE DATASETS
# ============================================================

def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    excluded: pd.DataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    test.to_csv(
        TEST_FILE,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_FILE,
        index=False,
    )

    print("\n4. OUTPUT FILES")

    for file in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
        EXCLUDED_FILE,
    ]:
        print(file)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHISHGUARD AI - GROUP-AWARE DATASET SPLITTING")
    print("=" * 65)

    df = load_data()

    print(f"\nTotal URLs loaded: {len(df):,}")

    # --------------------------------------------------------
    # REMOVE UNGROUPED URLS FROM FIRST EXPERIMENT
    # --------------------------------------------------------

    eligible, excluded = separate_ungrouped(df)

    print(
        f"URLs eligible for splitting: {len(eligible):,}"
    )

    print(
        f"URLs excluded without group: {len(excluded):,}"
    )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    train, validation, test = split_dataset(eligible)

    # --------------------------------------------------------
    # VERIFY ROW COUNTS
    # --------------------------------------------------------

    split_total = (
        len(train)
        + len(validation)
        + len(test)
    )

    if split_total != len(eligible):
        raise ValueError(
            "The split changed the number of eligible URLs"
        )

    # --------------------------------------------------------
    # VERIFY GROUP ISOLATION
    # --------------------------------------------------------

    verify_group_isolation(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # PRINT STATISTICS
    # --------------------------------------------------------

    print_split_statistics(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_splits(
        train,
        validation,
        test,
        excluded,
    )

    print("\nDataset splitting completed successfully.")


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()