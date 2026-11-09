"""
PhishGuard AI - Domain Extraction

Extracts registrable domains and creates grouping keys
for future train/validation/test splitting.

Handles:
- Standard registrable domains
- Private Public Suffix List domains
- Private suffix roots
- IPv4 and IPv6 addresses
- Invalid or unsupported URL formats

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
    """
    Extract hostname, registrable domain and grouping key.

    Returns:
        tuple:
            hostname
            registrable_domain
            group_key
            domain_status
    """

    # Reuse existing URL structural validator
    status, is_ip = validate_url(url)

    if status != "valid_structure":
        return ("", "", "", status)

    # Parse URL
    parsed = urlsplit(url)

    hostname = parsed.hostname

    if not hostname:
        return ("", "", "", "missing_hostname")

    hostname = hostname.lower()

    # --------------------------------------------------------
    # HANDLE IP ADDRESSES
    # --------------------------------------------------------

    if is_ip:

        normalized_ip = ip_address(hostname).compressed

        return (
            hostname,
            "",
            f"ip:{normalized_ip}",
            "ip",
        )

    # --------------------------------------------------------
    # EXTRACT REGISTRABLE DOMAIN
    # --------------------------------------------------------

    result = EXTRACTOR(hostname)

    registrable_domain = (
        result.top_domain_under_public_suffix.lower()
    )

    # --------------------------------------------------------
    # HANDLE PRIVATE SUFFIX ROOTS
    # --------------------------------------------------------

    if not registrable_domain:

        # Example:
        # s3.amazonaws.com
        # cloudflare-ipfs.com
        #
        # These hostnames can be private PSL suffixes.
        # They do not have a separate registrable domain.
        #
        # Use the complete hostname as a grouping key.

        if result.is_private and not result.domain:

            return (
                hostname,
                "",
                f"host:{hostname}",
                "private_suffix_root",
            )

        # Unknown or public suffix roots remain unresolved
        return (
            hostname,
            "",
            "",
            "unresolved_domain",
        )

    # --------------------------------------------------------
    # CREATE DOMAIN GROUPING KEY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXTRACT DOMAIN INFORMATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COMBINE ORIGINAL DATA WITH DOMAIN INFORMATION
    # --------------------------------------------------------

    df = pd.concat(
        [
            df.reset_index(drop=True),
            domain_data,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # DOMAIN STATUS DISTRIBUTION
    # --------------------------------------------------------

    print("\n1. DOMAIN EXTRACTION RESULTS")

    print(
        df["domain_status"].value_counts()
    )

    # --------------------------------------------------------
    # ANALYZE DOMAIN GROUPS
    # --------------------------------------------------------

    grouped_urls = df[
        df["group_key"] != ""
    ]

    groups = (
        grouped_urls.groupby("group_key")
        .agg(
            url_count=("url", "size"),
            label_types=("label", "nunique"),
        )
    )

    print("\n2. DOMAIN GROUP STATISTICS")

    print(
        f"Total groups: {len(groups):,}"
    )

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

    # --------------------------------------------------------
    # URLS WITHOUT RELIABLE GROUPING KEY
    # --------------------------------------------------------

    print("\n3. URLs WITHOUT RELIABLE GROUPING KEY")

    print(
        f"{(df['group_key'] == '').sum():,}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

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


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()