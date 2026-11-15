"""
PhishGuard AI - URL Validation

Performs initial structural validation of deduplicated URLs.

This script does not:
- Modify URLs
- Remove records
- Resolve DNS
- Visit websites
"""

from pathlib import Path
from urllib.parse import urlsplit
from ipaddress import ip_address

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "deduplicated_urls.csv"
)


# ============================================================
# 2. VALIDATE INDIVIDUAL URL
# ============================================================

def validate_url(url: str) -> tuple[str, bool]:

    if not isinstance(url, str) or not url.strip():
        return "empty_url", False

    url = url.strip()

    # Only HTTP and HTTPS URLs are supported in this audit
    if not url.lower().startswith(("http://", "https://")):
        return "missing_http_scheme", False

    # Detect malformed URL structure
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname

        # Accessing port also validates its syntax
        _ = parsed.port

    except ValueError:
        return "malformed_url", False

    # Validate hostname presence
    if not hostname:
        return "missing_hostname", False

    # Detect whitespace in hostname
    if any(char.isspace() for char in hostname):
        return "hostname_contains_whitespace", False

    # Detect obvious invalid domain structures
    if hostname.startswith(".") or ".." in hostname:
        return "invalid_hostname_structure", False

    # Check whether hostname is an IP address
    try:
        ip_address(hostname)
        return "valid_structure", True

    except ValueError:
        pass

    # Identify single-label hostnames
    if "." not in hostname:
        return "single_label_hostname", False

    return "valid_structure", False


# ============================================================
# 3. ANALYZE DATASET
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - URL VALIDATION")
    print("=" * 60)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=["url", "label", "sources"],
        dtype={"url": "string"},
    )

    print(f"\nTotal URLs: {len(df):,}")

    # Validate each URL
    results = df["url"].map(validate_url)

    df["validation_status"] = results.map(
        lambda result: result[0]
    )

    df["is_ip_address"] = results.map(
        lambda result: result[1]
    )

    # General results
    print("\n1. URL VALIDATION RESULTS")

    print(
        df["validation_status"]
        .value_counts()
    )

    # IP analysis
    print("\n2. IP ADDRESS ANALYSIS")

    print(
        "URLs using IP addresses:",
        f"{df['is_ip_address'].sum():,}"
    )

    print("\nIP address label distribution:")

    print(
        df.loc[
            df["is_ip_address"],
            "label"
        ].value_counts()
    )

    # URL lengths
    print("\n3. URL LENGTH ANALYSIS")

    lengths = df["url"].str.len()

    print(f"Minimum length: {lengths.min()}")
    print(f"Maximum length: {lengths.max()}")
    print(f"Average length: {lengths.mean():.2f}")

    print(
        "URLs longer than 2048 characters:",
        f"{(lengths > 2048).sum():,}"
    )

    # Validation status by label
    print("\n4. VALIDATION STATUS BY LABEL")

    print(
        pd.crosstab(
            df["validation_status"],
            df["label"]
        )
    )

    print("\nURL validation completed successfully.")


if __name__ == "__main__":
    main()