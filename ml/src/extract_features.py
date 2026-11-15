"""
PhishGuard AI - URL Feature Engineering

Extracts numerical features from URL strings.

Input:
    ml/data/processed/domain_groups.csv

Output:
    ml/data/processed/url_features.csv

Features are calculated exclusively from URL text.

No HTTP requests, DNS resolution or website access.
"""

import math
import re

from collections import Counter
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

from extract_domains import EXTRACTOR


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "domain_groups.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "url_features.csv"
)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "digit_count",
    "letter_count",
    "dot_count",
    "hyphen_count",
    "at_count",
    "percent_count",
    "slash_count",
    "question_mark_count",
    "query_param_count",
    "subdomain_count",
    "is_https",
    "has_ip",
    "has_punycode",
    "url_entropy",
    "digit_ratio",
    "special_char_ratio",
]


# ============================================================
# 3. URL ENTROPY
# ============================================================

def calculate_entropy(value: str) -> float:
    """
    Calculate Shannon entropy of a string.

    Higher entropy means a more varied character distribution.
    """

    if not value:
        return 0.0

    counts = Counter(value)
    length = len(value)

    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

    return entropy


# ============================================================
# 4. URL PARSING
# ============================================================

def parse_url(url: str):
    """
    Parse a URL without making network requests.

    Supports HTTP(S) URLs and URLs without a scheme.
    """

    has_explicit_scheme = re.match(
        r"^[a-z][a-z0-9+.-]*://",
        url,
        flags=re.IGNORECASE,
    )

    candidate = (
        url if has_explicit_scheme else f"//{url}"
    )

    try:
        return urlsplit(candidate)

    except ValueError:
        return None


# ============================================================
# 5. EXTRACT NUMERICAL FEATURES
# ============================================================

def extract_url_features(url: str) -> dict:
    """
    Extract numerical features from a single URL.

    Returns a dictionary containing 20 features.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()

    parsed = parse_url(url)

    hostname = ""
    path = ""
    query = ""
    is_https = 0

    if parsed is not None:
        try:
            hostname = (parsed.hostname or "").lower()
        except ValueError:
            hostname = ""

        path = parsed.path
        query = parsed.query

        is_https = int(
            parsed.scheme.lower() == "https"
        )

    # --------------------------------------------------------
    # CHARACTER COUNTS
    # --------------------------------------------------------

    url_length = len(url)

    digit_count = sum(
        char.isdigit()
        for char in url
    )

    letter_count = sum(
        char.isalpha()
        for char in url
    )

    special_char_count = sum(
        not char.isalnum()
        for char in url
    )

    # --------------------------------------------------------
    # IP ADDRESS DETECTION
    # --------------------------------------------------------

    has_ip = 0

    if hostname:
        try:
            ip_address(hostname)
            has_ip = 1
        except ValueError:
            pass

    # --------------------------------------------------------
    # SUBDOMAIN COUNT
    # --------------------------------------------------------

    subdomain_count = 0

    if hostname and not has_ip:
        extracted = EXTRACTOR(hostname)

        if extracted.subdomain:
            subdomain_count = len(
                extracted.subdomain.split(".")
            )

    # --------------------------------------------------------
    # QUERY PARAMETERS
    # --------------------------------------------------------

    query_param_count = len(
        parse_qsl(
            query,
            keep_blank_values=True,
        )
    )

    # --------------------------------------------------------
    # CHARACTER RATIOS
    # --------------------------------------------------------

    digit_ratio = digit_count / url_length

    special_char_ratio = (
        special_char_count / url_length
    )

    # --------------------------------------------------------
    # BUILD FEATURE DICTIONARY
    # --------------------------------------------------------

    features = {
        "url_length": url_length,

        "hostname_length": len(hostname),

        "path_length": len(path),

        "query_length": len(query),

        "digit_count": digit_count,

        "letter_count": letter_count,

        "dot_count": url.count("."),

        "hyphen_count": url.count("-"),

        "at_count": url.count("@"),

        "percent_count": url.count("%"),

        "slash_count": url.count("/"),

        "question_mark_count": url.count("?"),

        "query_param_count": query_param_count,

        "subdomain_count": subdomain_count,

        "is_https": is_https,

        "has_ip": has_ip,

        "has_punycode": int(
            "xn--" in hostname
        ),

        "url_entropy": calculate_entropy(url),

        "digit_ratio": digit_ratio,

        "special_char_ratio": special_char_ratio,
    }

    return features


# ============================================================
# 6. PROCESS DATASET
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - FEATURE ENGINEERING")
    print("=" * 60)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "url",
            "label",
            "sources",
            "group_key",
            "domain_status",
        ],
        dtype={"url": "string"},
        keep_default_na=False,
    )

    print(f"\nTotal URLs: {len(df):,}")

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    print("\nExtracting URL features...")

    records = [
        extract_url_features(url)
        for url in df["url"]
    ]

    features_df = pd.DataFrame.from_records(
        records,
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------------
    # COMBINE METADATA AND FEATURES
    # --------------------------------------------------------

    result = pd.concat(
        [
            df.reset_index(drop=True),
            features_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # VALIDATE RESULTS
    # --------------------------------------------------------

    if len(result) != len(df):
        raise ValueError(
            "Feature extraction changed the number of rows"
        )

    if features_df.isna().any().any():
        raise ValueError(
            "Extracted features contain missing values"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\nFEATURE EXTRACTION SUMMARY")

    print(f"Processed URLs: {len(result):,}")

    print(
        f"Extracted features: {len(FEATURE_COLUMNS)}"
    )

    print(
        "URLs with grouping key:",
        f"{result['group_key'].ne('').sum():,}"
    )

    print(
        "URLs without grouping key:",
        f"{result['group_key'].eq('').sum():,}"
    )

    print("\nFeature statistics:")

    print(
        features_df[
            [
                "url_length",
                "digit_count",
                "subdomain_count",
                "url_entropy",
            ]
        ].describe().round(2)
    )

    print(f"\nOutput saved: {OUTPUT_FILE}")

    print("\nFeature engineering completed successfully.")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()