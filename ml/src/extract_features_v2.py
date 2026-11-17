"""
PhishGuard AI - Feature Engineering V2

Experimental URL feature extraction designed to improve
cross-dataset generalization.

Feature sets:
    - Protocol-independent features.
    - Extended structural and lexical features.

Input:
    ml/data/splits/train.csv
    ml/data/splits/validation.csv

Output:
    ml/data/splits/v2/train.csv
    ml/data/splits/v2/validation.csv

IMPORTANT:
    Original features and dataset splits are preserved.
    The test dataset is never accessed.
    No trained models are overwritten.
    No network requests are performed.
"""

import re

from pathlib import Path
from ipaddress import ip_address
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd

from extract_domains import EXTRACTOR

from extract_features import (
    FEATURE_COLUMNS,
    calculate_entropy,
)

from train_model import (
    TRAIN_FILE,
    VALIDATION_FILE,
)

from experiment_training_strategies import (
    verify_group_isolation,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "data" / "splits" / "v2"

CHUNK_SIZE = 10_000

METADATA_COLUMNS = [
    "url",
    "label",
    "sources",
    "group_key",
    "domain_status",
]


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

PROTOCOL_INDEPENDENT_FEATURES = [
    f"v2_{feature}"
    for feature in FEATURE_COLUMNS
    if feature != "is_https"
]

ADDITIONAL_FEATURES = [
    "v2_registered_domain_length",
    "v2_hostname_entropy",
    "v2_domain_label_entropy",
    "v2_path_entropy",
    "v2_path_depth",
    "v2_hostname_digit_ratio",
    "v2_hostname_hyphen_count",
    "v2_domain_token_count",
    "v2_auth_token_count",
    "v2_percent_encoded_count",
    "v2_has_userinfo",
    "v2_has_explicit_port",
]

FEATURE_COLUMNS_V2 = (
    PROTOCOL_INDEPENDENT_FEATURES
    + ADDITIONAL_FEATURES
)

AUTH_TOKENS = {
    "login",
    "signin",
    "account",
    "verify",
    "verification",
    "secure",
    "security",
    "password",
    "reset",
    "confirm",
    "authentication",
    "auth",
    "credential",
    "credentials",
    "update",
    "validate",
    "validation",
    "recover",
    "recovery",
    "unlock",
    "session",
    "token",
}

SCHEME_PATTERN = re.compile(
    r"^[a-z][a-z0-9+.-]*://",
    flags=re.IGNORECASE,
)

PERCENT_ENCODED_PATTERN = re.compile(
    r"%[0-9a-fA-F]{2}"
)

TOKEN_PATTERN = re.compile(
    r"[a-z]+"
)


# ============================================================
# 3. URL PARSING
# ============================================================

def parse_url_v2(url: str):
    """
    Parse HTTP(S) URLs and URLs without a scheme.

    Parsing does not perform network requests.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            "URL must be a non-empty string"
        )

    url = url.strip()

    if SCHEME_PATTERN.match(url):
        candidate = url

    elif url.startswith("//"):
        candidate = url

    else:
        candidate = f"//{url}"

    try:

        parsed = urlsplit(candidate)

        hostname = parsed.hostname

        # Accessing port validates its format and range.
        _ = parsed.port

    except ValueError as exc:

        raise ValueError(
            "Malformed URL"
        ) from exc

    if parsed.scheme.lower() not in [
        "",
        "http",
        "https",
    ]:

        raise ValueError(
            "Unsupported URL scheme"
        )

    if not hostname:
        raise ValueError(
            "URL must contain a hostname"
        )

    return parsed


# ============================================================
# 4. PROTOCOL-INDEPENDENT URL REPRESENTATION
# ============================================================

def build_protocol_independent_url(parsed) -> str:
    """
    Build a URL representation without HTTP or HTTPS.

    An empty root path and '/' are treated equally.

    Non-root paths, query strings and fragments
    are preserved.
    """

    path = (
        ""
        if parsed.path == "/"
        else parsed.path
    )

    result = parsed.netloc + path

    if parsed.query:
        result += f"?{parsed.query}"

    if parsed.fragment:
        result += f"#{parsed.fragment}"

    return result


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def count_digits(value: str) -> int:

    return sum(
        char.isdigit()
        for char in value
    )


def count_letters(value: str) -> int:

    return sum(
        char.isalpha()
        for char in value
    )


def detect_ip(hostname: str) -> int:

    try:

        ip_address(hostname)

        return 1

    except ValueError:

        return 0


def count_path_depth(path: str) -> int:
    """
    Count non-empty path segments.
    """

    return len([
        segment
        for segment in path.split("/")
        if segment
    ])


def count_auth_tokens(value: str) -> int:
    """
    Count authentication-related words.

    Matches complete alphabetic tokens rather
    than arbitrary substrings.
    """

    tokens = TOKEN_PATTERN.findall(
        value.lower()
    )

    return sum(
        token in AUTH_TOKENS
        for token in tokens
    )


# ============================================================
# 6. DOMAIN FEATURE EXTRACTION
# ============================================================

def extract_domain_features(
    hostname: str,
    has_ip: int,
) -> dict:
    """
    Extract domain components using the existing
    offline Public Suffix List configuration.
    """

    if has_ip:

        return {
            "registered_domain": "",
            "domain_label": "",
            "subdomain_count": 0,
        }

    extracted = EXTRACTOR(hostname)

    registered_domain = (
        extracted.top_domain_under_public_suffix
        .lower()
    )

    domain_label = extracted.domain.lower()

    subdomain_count = (
        len(extracted.subdomain.split("."))
        if extracted.subdomain
        else 0
    )

    return {
        "registered_domain": registered_domain,
        "domain_label": domain_label,
        "subdomain_count": subdomain_count,
    }


# ============================================================
# 7. MAIN FEATURE EXTRACTOR
# ============================================================

def extract_url_features_v2(url: str) -> dict:
    """
    Extract protocol-independent and extended
    numerical features from a URL.
    """

    parsed = parse_url_v2(url)

    hostname = parsed.hostname.lower()

    path = (
        ""
        if parsed.path == "/"
        else parsed.path
    )

    query = parsed.query

    normalized_url = (
        build_protocol_independent_url(parsed)
    )

    # --------------------------------------------------------
    # PROTOCOL-INDEPENDENT CHARACTER COUNTS
    # --------------------------------------------------------

    url_length = len(normalized_url)

    digit_count = count_digits(
        normalized_url
    )

    letter_count = count_letters(
        normalized_url
    )

    special_char_count = sum(
        not char.isalnum()
        for char in normalized_url
    )

    # --------------------------------------------------------
    # DOMAIN STRUCTURE
    # --------------------------------------------------------

    has_ip = detect_ip(hostname)

    domain_features = extract_domain_features(
        hostname,
        has_ip,
    )

    registered_domain = domain_features[
        "registered_domain"
    ]

    domain_label = domain_features[
        "domain_label"
    ]

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
    # PROTOCOL-INDEPENDENT FEATURES
    # --------------------------------------------------------

    features = {
        "v2_url_length": url_length,

        "v2_hostname_length": len(hostname),

        "v2_path_length": len(path),

        "v2_query_length": len(query),

        "v2_digit_count": digit_count,

        "v2_letter_count": letter_count,

        "v2_dot_count": normalized_url.count("."),

        "v2_hyphen_count": normalized_url.count("-"),

        "v2_at_count": normalized_url.count("@"),

        "v2_percent_count": normalized_url.count("%"),

        "v2_slash_count": normalized_url.count("/"),

        "v2_question_mark_count":
            normalized_url.count("?"),

        "v2_query_param_count": query_param_count,

        "v2_subdomain_count":
            domain_features["subdomain_count"],

        "v2_has_ip": has_ip,

        "v2_has_punycode": int(
            "xn--" in hostname
        ),

        "v2_url_entropy": calculate_entropy(
            normalized_url
        ),

        "v2_digit_ratio": safe_ratio(
            digit_count,
            url_length,
        ),

        "v2_special_char_ratio": safe_ratio(
            special_char_count,
            url_length,
        ),
    }

    # --------------------------------------------------------
    # EXTENDED FEATURES
    # --------------------------------------------------------

    features.update({
        "v2_registered_domain_length": len(
            registered_domain
        ),

        "v2_hostname_entropy": calculate_entropy(
            hostname
        ),

        "v2_domain_label_entropy": calculate_entropy(
            domain_label
        ),

        "v2_path_entropy": calculate_entropy(
            path
        ),

        "v2_path_depth": count_path_depth(
            path
        ),

        "v2_hostname_digit_ratio": safe_ratio(
            count_digits(hostname),
            len(hostname),
        ),

        "v2_hostname_hyphen_count":
            hostname.count("-"),

        "v2_domain_token_count": len(
            re.findall(
                r"[a-z0-9]+",
                domain_label,
            )
        ),

        "v2_auth_token_count": count_auth_tokens(
            f"{hostname} {path} {query}"
        ),

        "v2_percent_encoded_count": len(
            PERCENT_ENCODED_PATTERN.findall(
                normalized_url
            )
        ),

        "v2_has_userinfo": int(
            parsed.username is not None
        ),

        "v2_has_explicit_port": int(
            parsed.port is not None
        ),
    })

    return features


# ============================================================
# 8. VALIDATE ORIGINAL DATASET SPLITS
# ============================================================

def validate_original_splits():
    """
    Verify that existing training and validation splits
    remain isolated by domain group.

    The test dataset is intentionally excluded.
    """

    metadata = []

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
    ]:

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path,
            usecols=METADATA_COLUMNS,
            keep_default_na=False,
            dtype={
                "url": "string",
                "sources": "string",
                "group_key": "string",
                "domain_status": "string",
            },
        )

        if dataframe.empty:
            raise ValueError(
                "Dataset cannot be empty"
            )

        if not dataframe["label"].isin([0, 1]).all():
            raise ValueError(
                "Invalid labels"
            )

        if dataframe["group_key"].eq("").any():
            raise ValueError(
                "Missing group keys"
            )

        if dataframe["url"].duplicated().any():
            raise ValueError(
                "Duplicate URLs found"
            )

        metadata.append(dataframe)

    train_df, validation_df = metadata

    verify_group_isolation(
        train_df,
        validation_df,
    )

    if not set(train_df["url"]).isdisjoint(
        validation_df["url"]
    ):

        raise ValueError(
            "URL leakage detected"
        )

    return (
        len(train_df),
        len(validation_df),
    )


# ============================================================
# 9. EXTRACT FEATURES FROM DATAFRAME
# ============================================================

def transform_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append V2 features while preserving all
    original metadata and V1 features.
    """

    if dataframe.empty:
        raise ValueError(
            "Dataset cannot be empty"
        )

    if "url" not in dataframe.columns:
        raise ValueError(
            "Missing URL column"
        )

    existing_features = (
        set(FEATURE_COLUMNS_V2)
        & set(dataframe.columns)
    )

    if existing_features:
        raise ValueError(
            "V2 features already exist"
        )

    records = [
        extract_url_features_v2(url)
        for url in dataframe["url"]
    ]

    features_df = pd.DataFrame.from_records(
        records,
        columns=FEATURE_COLUMNS_V2,
    )

    numerical_values = features_df.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numerical_values).all():
        raise ValueError(
            "V2 features contain invalid numerical values"
        )

    result = pd.concat(
        [
            dataframe.reset_index(drop=True),
            features_df,
        ],
        axis=1,
    )

    if len(result) != len(dataframe):
        raise ValueError(
            "Feature extraction changed the row count"
        )

    return result


# ============================================================
# 10. PROCESS DATASET IN CHUNKS
# ============================================================

def process_dataset(
    input_file: Path,
    output_file: Path,
    expected_rows: int,
) -> int:
    """
    Process large datasets in chunks to reduce
    memory usage.

    A temporary file prevents incomplete output
    from replacing an existing dataset.
    """

    print("\n" + "=" * 65)

    print(f"PROCESSING: {input_file.name}")

    print("=" * 65)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = output_file.with_suffix(
        ".csv.tmp"
    )

    processed_rows = 0

    try:

        for chunk in pd.read_csv(
            input_file,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            dtype={
                "url": "string",
                "sources": "string",
                "group_key": "string",
                "domain_status": "string",
            },
        ):

            transformed = transform_dataframe(
                chunk
            )

            transformed.to_csv(
                temp_file,
                mode="w" if processed_rows == 0 else "a",
                header=processed_rows == 0,
                index=False,
            )

            processed_rows += len(transformed)

            print(
                f"Processed: "
                f"{processed_rows:,}/{expected_rows:,}"
            )

        if processed_rows != expected_rows:
            raise ValueError(
                "Unexpected number of processed rows"
            )

        temp_file.replace(output_file)

    finally:

        if temp_file.exists():
            temp_file.unlink()

    print("\nFeature extraction completed.")

    print(f"Output: {output_file}")

    return processed_rows


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 65)

    print("PHISHGUARD AI - FEATURE ENGINEERING V2")

    print("=" * 65)

    print("\nValidating original dataset splits...")

    train_rows, validation_rows = (
        validate_original_splits()
    )

    print(f"\nTraining URLs: {train_rows:,}")

    print(f"Validation URLs: {validation_rows:,}")

    print(
        "Protocol-independent features:",
        len(PROTOCOL_INDEPENDENT_FEATURES),
    )

    print(
        "Additional features:",
        len(ADDITIONAL_FEATURES),
    )

    print(
        "Total V2 features:",
        len(FEATURE_COLUMNS_V2),
    )

    # --------------------------------------------------------
    # PROCESS TRAINING DATA
    # --------------------------------------------------------

    process_dataset(
        TRAIN_FILE,
        OUTPUT_DIR / "train.csv",
        train_rows,
    )

    # --------------------------------------------------------
    # PROCESS VALIDATION DATA
    # --------------------------------------------------------

    process_dataset(
        VALIDATION_FILE,
        OUTPUT_DIR / "validation.csv",
        validation_rows,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("FEATURE ENGINEERING V2 COMPLETED")

    print("=" * 65)

    print(f"\nTraining URLs: {train_rows:,}")

    print(f"Validation URLs: {validation_rows:,}")

    print(
        "Original V1 features:",
        len(FEATURE_COLUMNS),
    )

    print(
        "New V2 features:",
        len(FEATURE_COLUMNS_V2),
    )

    print("\nOriginal dataset splits were preserved.")

    print("No trained models were overwritten.")

    print("The test dataset was not accessed.")


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()