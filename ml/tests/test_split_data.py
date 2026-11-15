"""
PhishGuard AI - Dataset Splitting Unit Tests

Tests:
- Separation of ungrouped URLs
- Group isolation between splits
- Preservation of all eligible URLs
- Reproducible splitting
- Approximate split proportions
- Group leakage detection
- Invalid label detection
- Duplicate URL detection
- Missing required columns
"""

import sys
from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# 1. IMPORT PROJECT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import split_data


# ============================================================
# 2. TEST DATA FIXTURE
# ============================================================

@pytest.fixture
def sample_dataset():
    """
    Create 240 synthetic URLs distributed across 120 groups.

    Each group contains two URLs belonging to the same domain.
    """

    records = []

    for group_id in range(120):

        domain = f"site{group_id}.com"

        label = group_id % 2

        for variant in range(2):

            records.append(
                {
                    "url": (
                        f"https://{domain}/path{variant}"
                    ),
                    "label": label,
                    "sources": "test_dataset",
                    "group_key": f"domain:{domain}",
                    "domain_status": "domain",
                    "url_length": 30,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# 3. TEST UNGROUPED URL SEPARATION
# ============================================================

def test_separate_ungrouped(sample_dataset):

    df = sample_dataset.copy()

    df.loc[0, "group_key"] = ""
    df.loc[1, "group_key"] = ""

    eligible, excluded = split_data.separate_ungrouped(df)

    assert len(eligible) == 238
    assert len(excluded) == 2

    assert eligible["group_key"].ne("").all()
    assert excluded["group_key"].eq("").all()

    assert len(eligible) + len(excluded) == len(df)


# ============================================================
# 4. TEST GROUP ISOLATION AND ROW PRESERVATION
# ============================================================

def test_split_preserves_rows_and_groups(sample_dataset):

    train, validation, test = split_data.split_dataset(
        sample_dataset
    )

    # Every URL must appear in exactly one split
    all_urls = (
        train["url"].tolist()
        + validation["url"].tolist()
        + test["url"].tolist()
    )

    assert len(all_urls) == len(sample_dataset)

    assert len(set(all_urls)) == len(sample_dataset)

    assert set(all_urls) == set(sample_dataset["url"])

    # Group keys must not overlap
    train_groups = set(train["group_key"])
    validation_groups = set(validation["group_key"])
    test_groups = set(test["group_key"])

    assert train_groups.isdisjoint(validation_groups)

    assert train_groups.isdisjoint(test_groups)

    assert validation_groups.isdisjoint(test_groups)

    # Verify using the project's own validation function
    split_data.verify_group_isolation(
        train,
        validation,
        test,
    )


# ============================================================
# 5. TEST REPRODUCIBILITY
# ============================================================

def test_split_is_reproducible(sample_dataset):
    """
    Repeating the split with the same random_state
    must produce identical results.
    """

    first = split_data.split_dataset(sample_dataset)

    second = split_data.split_dataset(sample_dataset)

    for first_df, second_df in zip(first, second):

        assert (
            first_df["url"].tolist()
            == second_df["url"].tolist()
        )


# ============================================================
# 6. TEST SPLIT PROPORTIONS
# ============================================================

def test_split_proportions(sample_dataset):

    train, validation, test = split_data.split_dataset(
        sample_dataset
    )

    total = len(sample_dataset)

    train_ratio = len(train) / total

    validation_ratio = len(validation) / total

    test_ratio = len(test) / total

    # Allow a small tolerance for group-aware splitting
    assert abs(train_ratio - 0.70) <= 0.02

    assert abs(validation_ratio - 0.15) <= 0.02

    assert abs(test_ratio - 0.15) <= 0.02


# ============================================================
# 7. TEST GROUP LEAKAGE DETECTION
# ============================================================

def test_group_leakage_detection():

    train = pd.DataFrame(
        {
            "group_key": ["domain:example.com"]
        }
    )

    validation = pd.DataFrame(
        {
            "group_key": ["domain:example.com"]
        }
    )

    test = pd.DataFrame(
        {
            "group_key": ["domain:other.com"]
        }
    )

    with pytest.raises(
        ValueError,
        match="Group leakage",
    ):

        split_data.verify_group_isolation(
            train,
            validation,
            test,
        )


# ============================================================
# 8. TEST INVALID LABELS
# ============================================================

def test_load_data_rejects_invalid_labels(
    tmp_path,
    monkeypatch,
):

    df = pd.DataFrame(
        {
            "url": [
                "https://example.com",
                "https://other.com",
            ],
            "label": [0, 2],
            "sources": ["test", "test"],
            "group_key": [
                "domain:example.com",
                "domain:other.com",
            ],
            "domain_status": ["domain", "domain"],
        }
    )

    file_path = tmp_path / "invalid_labels.csv"

    df.to_csv(file_path, index=False)

    monkeypatch.setattr(
        split_data,
        "INPUT_FILE",
        file_path,
    )

    with pytest.raises(
        ValueError,
        match="Labels",
    ):

        split_data.load_data()


# ============================================================
# 9. TEST DUPLICATE URL DETECTION
# ============================================================

def test_load_data_rejects_duplicates(
    tmp_path,
    monkeypatch,
):

    df = pd.DataFrame(
        {
            "url": [
                "https://example.com",
                "https://example.com",
            ],
            "label": [0, 1],
            "sources": ["test", "test"],
            "group_key": [
                "domain:example.com",
                "domain:example.com",
            ],
            "domain_status": ["domain", "domain"],
        }
    )

    file_path = tmp_path / "duplicates.csv"

    df.to_csv(file_path, index=False)

    monkeypatch.setattr(
        split_data,
        "INPUT_FILE",
        file_path,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate URLs",
    ):

        split_data.load_data()


# ============================================================
# 10. TEST MISSING REQUIRED COLUMNS
# ============================================================

def test_load_data_rejects_missing_columns(
    tmp_path,
    monkeypatch,
):

    df = pd.DataFrame(
        {
            "url": ["https://example.com"],
            "label": [0],
            "group_key": ["domain:example.com"],
            "domain_status": ["domain"],
        }
    )

    # The required 'sources' column is missing

    file_path = tmp_path / "missing_columns.csv"

    df.to_csv(file_path, index=False)

    monkeypatch.setattr(
        split_data,
        "INPUT_FILE",
        file_path,
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):

        split_data.load_data()