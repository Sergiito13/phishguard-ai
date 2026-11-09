"""
PhishGuard AI - Domain Extraction

Extracts registrable domains and creates grouping keys
for future train/validation/test splitting.

Input:
    ml/data/processed/deduplicated_urls.csv

Output:
    ml/data/processed/domain_groups.csv

Does not visit websites or modify original datasets.
"""

from pathlib import Path
from urllib.parse import urlsplit
from ipaddress import ip_address

import pandas as pd
import tldextract

from validate_urls import validate_url


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "deduplicated_urls.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "domain_groups.csv"
)


# ============================================================
# 2. DOMAIN EXTRACTOR CONFIGURATION
# ============================================================

EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    include_psl_private_domains=True,
)


# ============================================================
# 3. EXTRACT DOMAIN INFORMATION
# ============================================================

def extract_domain_info(url: str) -> tuple:

    # Reuse our existing structural validator
    status, is_ip = validate_url(url)

    if status != "valid_structure":
        return ("", "", "", status)

    # Parse URL
    parsed = urlsplit(url)

    hostname = parsed.hostname

    if not hostname:
        return ("", "", "", "missing_hostname")

    hostname = hostname.lower()

    # Detect IP addresses
    if is_ip:

        normalized_ip = ip_address(hostname).compressed

        return (
            hostname,
            "",
            f"ip:{normalized_ip}",
            "ip",
        )

    # Extract registrable domain
    result = EXTRACTOR(hostname)

    registrable_domain = (
        result.top_domain_under_public_suffix.lower()
    )

    # Domain cannot be reliably determined
    if not registrable_domain:

        return (
            hostname,
            "",
            "",
            "unresolved_domain",
        )

    # Create grouping key
    group_key = f"domain:{registrable_domain}"

    return (
        hostname,
        registrable_domain,
        group_key,
        "domain",
    )


# ============================================================
# 4. PROCESS DATASET
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - DOMAIN EXTRACTION")
    print("=" * 60)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=["url", "label", "sources"],
        dtype={"url": "string"},
        keep_default_na=False,
    )

    print(f"\nTotal URLs: {len(df):,}")

    # Extract domain information
    results = [
        extract_domain_info(url)
        for url in df["url"]
    ]

    domain_data = pd.DataFrame(
        results,
        columns=[
            "hostname",
            "registrable_domain",
            "group_key",
            "domain_status",
        ],
    )

    # Combine original data and extracted information
    df = pd.concat(
        [
            df.reset_index(drop=True),
            domain_data,
        ],
        axis=1,
    )

    # Display domain status distribution
    print("\n1. DOMAIN EXTRACTION RESULTS")

    print(
        df["domain_status"].value_counts()
    )

    # Only URLs with a reliable grouping key
    grouped_urls = df[
        df["group_key"] != ""
    ]

    # Analyze domain groups
    groups = (
        grouped_urls.groupby("group_key")
        .agg(
            url_count=("url", "size"),
            label_types=("label", "nunique"),
        )
    )

    print("\n2. DOMAIN GROUP STATISTICS")

    print(f"Total groups: {len(groups):,}")

    print(
        "Groups containing multiple URLs:",
        f"{(groups['url_count'] > 1).sum():,}"
    )

    print(
        "Groups containing both labels:",
        f"{(groups['label_types'] > 1).sum():,}"
    )

    print(
        "Largest group:",
        f"{groups['url_count'].max():,}"
    )

    print("\n3. URLs WITHOUT RELIABLE GROUPING KEY")

    print(
        f"{(df['group_key'] == '').sum():,}"
    )

    # Save processed dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nOutput saved: {OUTPUT_FILE}")

    print("\nDomain extraction completed successfully.")


if __name__ == "__main__":
    main()