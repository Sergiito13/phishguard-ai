"""Experimental, protocol-independent URL features for PhishGuard AI.

V3 extends the existing 31 V2 features with 6 additional features.

No network requests, dataset writes or model updates.
"""

from urllib.parse import parse_qsl

import numpy as np
import pandas as pd

from extract_features import calculate_entropy

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    count_auth_tokens,
    count_digits,
    extract_url_features_v2,
    parse_url_v2,
    safe_ratio,
)


# ============================================================
# 1. FEATURE CONFIGURATION
# ============================================================

ADDITIONAL_FEATURES_V3 = [
    "v3_hostname_auth_tokens",
    "v3_path_auth_tokens",
    "v3_avg_path_segment_length",
    "v3_max_path_segment_length",
    "v3_path_digit_ratio",
    "v3_query_value_entropy",
]

FEATURE_COLUMNS_V3 = list(
    ADDITIONAL_FEATURES_V3
)

FEATURE_COLUMNS_EXTENDED_V3 = (
    list(FEATURE_COLUMNS_V2)
    + FEATURE_COLUMNS_V3
)


# ============================================================
# 2. V3 FEATURE EXTRACTION
# ============================================================

def extract_url_features_v3(
    url: str,
) -> dict[str, int | float]:
    """
    Extract six additional numerical features.

    Properties:
    - Independent of HTTP/HTTPS.
    - No network requests.
    - Authentication tokens use V2 tokenization.
    - Empty paths produce zero-valued statistics.
    - Query entropy uses decoded parameter values.
    """

    parsed = parse_url_v2(url)

    hostname = parsed.hostname.lower()

    path = (
        ""
        if parsed.path == "/"
        else parsed.path
    )

    # --------------------------------------------------------
    # PATH SEGMENTS
    # --------------------------------------------------------

    segments = [
        part
        for part in path.split("/")
        if part
    ]

    segment_lengths = [
        len(part)
        for part in segments
    ]

    # --------------------------------------------------------
    # QUERY VALUES
    # --------------------------------------------------------

    query_values = "".join(
        value
        for _, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = {
        "v3_hostname_auth_tokens": (
            count_auth_tokens(hostname)
        ),

        "v3_path_auth_tokens": (
            count_auth_tokens(path)
        ),

        "v3_avg_path_segment_length": (
            sum(segment_lengths) / len(segment_lengths)
            if segments
            else 0.0
        ),

        "v3_max_path_segment_length": (
            max(segment_lengths)
            if segments
            else 0
        ),

        "v3_path_digit_ratio": (
            safe_ratio(
                count_digits(path),
                len(path),
            )
        ),

        "v3_query_value_entropy": (
            calculate_entropy(query_values)
        ),
    }

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if list(features) != FEATURE_COLUMNS_V3:
        raise RuntimeError(
            "Unexpected V3 feature order"
        )

    values = np.array(
        list(features.values()),
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "V3 features contain non-finite values"
        )

    return features


# ============================================================
# 3. EXTENDED V3 FEATURES
# ============================================================

def extract_extended_features_v3(
    url: str,
) -> dict[str, int | float]:
    """
    Return 31 V2 features and 6 additional V3 features.

    Existing V2 feature values are preserved.
    """

    v2_features = extract_url_features_v2(
        url
    )

    v3_features = extract_url_features_v3(
        url
    )

    return {
        **v2_features,
        **v3_features,
    }


# ============================================================
# 4. DATAFRAME TRANSFORMATION
# ============================================================

def transform_dataframe_v3(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append V3 features to an existing V2 DataFrame.

    The original DataFrame is not modified.
    No files are written.
    """

    if frame.empty:
        raise ValueError(
            "Dataset cannot be empty"
        )

    if "url" not in frame.columns:
        raise ValueError(
            "Missing URL column"
        )

    # --------------------------------------------------------
    # VALIDATE EXISTING V2 FEATURES
    # --------------------------------------------------------

    missing_v2 = (
        set(FEATURE_COLUMNS_V2)
        - set(frame.columns)
    )

    if missing_v2:
        raise ValueError(
            f"Missing V2 features: {sorted(missing_v2)}"
        )

    existing_v3 = (
        set(FEATURE_COLUMNS_V3)
        & set(frame.columns)
    )

    if existing_v3:
        raise ValueError(
            f"V3 features already exist: {sorted(existing_v3)}"
        )

    existing_values = frame[
        FEATURE_COLUMNS_V2
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(existing_values).all():
        raise ValueError(
            "Existing V2 features contain non-finite values"
        )

    # --------------------------------------------------------
    # GENERATE V3 FEATURES
    # --------------------------------------------------------

    records = [
        extract_url_features_v3(url)
        for url in frame["url"]
    ]

    features_df = pd.DataFrame.from_records(
        records,
        columns=FEATURE_COLUMNS_V3,
    )

    # --------------------------------------------------------
    # MERGE FEATURES
    # --------------------------------------------------------

    result = pd.concat(
        [
            frame.reset_index(drop=True),
            features_df,
        ],
        axis=1,
    )

    if len(result) != len(frame):
        raise ValueError(
            "V3 extraction changed the row count"
        )

    return result


# ============================================================
# 5. STANDALONE SMOKE TEST
# ============================================================

if __name__ == "__main__":

    example = (
        "https://secure-login.example.com/"
        "account/reset?session=abcd"
    )

    print(
        "PHISHGUARD AI - FEATURE ENGINEERING V3"
    )

    print("=" * 60)

    features = extract_url_features_v3(
        example
    )

    for feature, value in features.items():
        print(
            f"{feature}: {value}"
        )

    print("=" * 60)

    print(
        f"V2 features: {len(FEATURE_COLUMNS_V2)}"
    )

    print(
        f"V3 new features: {len(FEATURE_COLUMNS_V3)}"
    )

    print(
        "Total features: "
        f"{len(FEATURE_COLUMNS_EXTENDED_V3)}"
    )