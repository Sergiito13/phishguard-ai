import sys
from pathlib import Path

import pandas as pd

# Allow imports from ml/src
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from deduplicate_data import deduplicate_urls


def test_duplicate_urls_are_merged():
    """Identical URLs should be grouped into one record."""

    df = pd.DataFrame({
        "url": [
            "https://example.com",
            "https://example.com",
        ],
        "label": [0, 0],
        "source": ["legitphish", "phiusiil"],
        "source_row": [2, 100],
    })

    unique, conflicts = deduplicate_urls(df)

    assert len(unique) == 1
    assert len(conflicts) == 0

    assert unique.iloc[0]["occurrences"] == 2
    assert unique.iloc[0]["source_count"] == 2
    assert unique.iloc[0]["sources"] == "legitphish|phiusiil"


def test_conflicting_labels_are_quarantined():
    """Conflicting labels must be excluded from the final dataset."""

    df = pd.DataFrame({
        "url": [
            "https://example.com",
            "https://example.com",
        ],
        "label": [0, 1],
        "source": ["legitphish", "phishbd"],
        "source_row": [2, 200],
    })

    unique, conflicts = deduplicate_urls(df)

    assert len(unique) == 0
    assert len(conflicts) == 2


def test_different_urls_are_preserved():
    """Different URL paths and protocols must remain separate."""

    df = pd.DataFrame({
        "url": [
            "https://example.com/Login",
            "https://example.com/login",
            "http://example.com/login",
        ],
        "label": [0, 0, 0],
        "source": ["legitphish"] * 3,
        "source_row": [2, 3, 4],
    })

    unique, conflicts = deduplicate_urls(df)

    assert len(unique) == 3
    assert len(conflicts) == 0