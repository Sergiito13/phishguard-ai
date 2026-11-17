"""
PhishGuard AI - Feature Engineering V2 Tests

Validate protocol-independent URL features,
extended features and dataset transformation.

All tests use synthetic URLs and temporary datasets.
No real datasets or trained models are modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import extract_features_v2 as v2

from extract_features import FEATURE_COLUMNS


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

def make_record(
    url: str,
    label: int,
    group_key: str,
) -> dict:

    record = {
        "url": url,
        "label": label,
        "sources": "synthetic",
        "group_key": group_key,
        "domain_status": "domain",
    }

    record.update({
        feature: 0
        for feature in FEATURE_COLUMNS
    })

    return record


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame([
        make_record(
            "https://example.com",
            0,
            "domain:example.com",
        ),
        make_record(
            "https://example.org/login",
            1,
            "domain:example.org",
        ),
    ])


# ============================================================
# 3. FEATURE CONFIGURATION
# ============================================================

def test_feature_configuration():

    assert len(
        v2.PROTOCOL_INDEPENDENT_FEATURES
    ) == 19

    assert len(
        v2.ADDITIONAL_FEATURES
    ) == 12

    assert len(
        v2.FEATURE_COLUMNS_V2
    ) == 31

    assert len(
        set(v2.FEATURE_COLUMNS_V2)
    ) == 31

    assert "v2_is_https" not in (
        v2.FEATURE_COLUMNS_V2
    )

    assert set(
        FEATURE_COLUMNS
    ).isdisjoint(
        v2.FEATURE_COLUMNS_V2
    )


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def test_extract_url_features_v2():

    features = v2.extract_url_features_v2(
        "https://example.com/login"
    )

    assert len(features) == 31

    assert list(features) == v2.FEATURE_COLUMNS_V2

    assert features["v2_hostname_length"] == 11

    assert features["v2_path_length"] == 6

    assert features["v2_path_depth"] == 1

    assert features["v2_auth_token_count"] == 1

    assert features["v2_has_ip"] == 0

    assert np.isfinite(
        list(features.values())
    ).all()


# ============================================================
# 5. PROTOCOL INDEPENDENCE
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "example.com",
        "example.com/login",
        "shop.example.com/products?id=123",
        "example.com/path#section",
        "192.0.2.10/login",
    ],
)
def test_protocol_independence(url):

    http_features = v2.extract_url_features_v2(
        f"http://{url}"
    )

    https_features = v2.extract_url_features_v2(
        f"https://{url}"
    )

    no_scheme_features = v2.extract_url_features_v2(
        url
    )

    assert http_features == https_features

    assert https_features == no_scheme_features


# ============================================================
# 6. ROOT PATH NORMALIZATION
# ============================================================

@pytest.mark.parametrize(
    "url_a,url_b",
    [
        (
            "https://example.com",
            "https://example.com/",
        ),
        (
            "https://example.com?id=1",
            "https://example.com/?id=1",
        ),
        (
            "https://example.com#section",
            "https://example.com/#section",
        ),
    ],
)
def test_root_path_normalization(url_a, url_b):

    features_a = v2.extract_url_features_v2(
        url_a
    )

    features_b = v2.extract_url_features_v2(
        url_b
    )

    assert features_a == features_b


def test_non_root_path_is_preserved():

    root = v2.extract_url_features_v2(
        "https://example.com"
    )

    path = v2.extract_url_features_v2(
        "https://example.com/login"
    )

    assert root != path

    assert root["v2_path_length"] == 0

    assert path["v2_path_length"] == 6


# ============================================================
# 7. DOMAIN FEATURES
# ============================================================

def test_registered_domain_features():

    features = v2.extract_url_features_v2(
        "https://shop.example.co.uk/products"
    )

    assert (
        features["v2_registered_domain_length"]
        == len("example.co.uk")
    )

    assert features["v2_subdomain_count"] == 1

    assert features["v2_domain_token_count"] == 1

    assert features["v2_hostname_entropy"] > 0

    assert features["v2_domain_label_entropy"] > 0


def test_private_suffix_domain():

    features = v2.extract_url_features_v2(
        "https://usuario.github.io/project"
    )

    assert features["v2_subdomain_count"] == 0

    assert (
        features["v2_registered_domain_length"]
        == len("usuario.github.io")
    )


def test_hostname_numeric_features():

    hostname = "shop-123.example.com"

    features = v2.extract_url_features_v2(
        f"https://{hostname}"
    )

    assert features[
        "v2_hostname_digit_ratio"
    ] == pytest.approx(
        3 / len(hostname)
    )

    assert features[
        "v2_hostname_hyphen_count"
    ] == 1


def test_domain_token_count():

    features = v2.extract_url_features_v2(
        "https://my-bank42.com"
    )

    assert features[
        "v2_domain_token_count"
    ] == 2


# ============================================================
# 8. IP ADDRESS DETECTION
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.2.10/login",
        "http://[2001:db8::1]/login",
    ],
)
def test_ip_address_features(url):

    features = v2.extract_url_features_v2(
        url
    )

    assert features["v2_has_ip"] == 1

    assert features["v2_subdomain_count"] == 0

    assert features[
        "v2_registered_domain_length"
    ] == 0


# ============================================================
# 9. PUNYCODE DETECTION
# ============================================================

def test_punycode_detection():

    features = v2.extract_url_features_v2(
        "https://xn--bcher-kva.example"
    )

    assert features["v2_has_punycode"] == 1


# ============================================================
# 10. PATH AND QUERY FEATURES
# ============================================================

def test_path_features():

    features = v2.extract_url_features_v2(
        "https://example.com/account/login/verify"
    )

    assert features["v2_path_depth"] == 3

    assert features["v2_path_entropy"] > 0

    assert features["v2_path_length"] == len(
        "/account/login/verify"
    )


def test_empty_path_features():

    features = v2.extract_url_features_v2(
        "https://example.com/"
    )

    assert features["v2_path_depth"] == 0

    assert features["v2_path_entropy"] == 0

    assert features["v2_path_length"] == 0


def test_query_parameter_count():

    features = v2.extract_url_features_v2(
        "https://example.com/path?a=1&b=&a=2"
    )

    assert features["v2_query_param_count"] == 3

    assert features["v2_query_length"] == len(
        "a=1&b=&a=2"
    )


# ============================================================
# 11. AUTHENTICATION TOKENS
# ============================================================

def test_auth_token_count():

    features = v2.extract_url_features_v2(
        "https://secure.example.com/account/login/verify"
    )

    assert features["v2_auth_token_count"] == 4


def test_auth_tokens_require_complete_words():

    assert v2.count_auth_tokens(
        "catalogin"
    ) == 0

    assert v2.count_auth_tokens(
        "login-verification"
    ) == 2


# ============================================================
# 12. URL ENCODING
# ============================================================

def test_percent_encoded_count():

    features = v2.extract_url_features_v2(
        "https://example.com/a%20b%2Fc"
    )

    assert features[
        "v2_percent_encoded_count"
    ] == 2

    assert features["v2_percent_count"] == 2


# ============================================================
# 13. USER INFORMATION AND PORTS
# ============================================================

def test_userinfo_and_explicit_port():

    features = v2.extract_url_features_v2(
        "https://user@example.com:8443/login"
    )

    assert features["v2_has_userinfo"] == 1

    assert features["v2_has_explicit_port"] == 1

    assert features["v2_at_count"] == 1


def test_url_without_userinfo_or_port():

    features = v2.extract_url_features_v2(
        "https://example.com"
    )

    assert features["v2_has_userinfo"] == 0

    assert features["v2_has_explicit_port"] == 0


# ============================================================
# 14. INVALID INPUT
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
        "https:///login",
        "ftp://example.com",
        "https://example.com:abc",
        "https://example.com:99999",
        "http://[invalid",
    ],
)
def test_invalid_url_input(url):

    with pytest.raises(ValueError):

        v2.extract_url_features_v2(url)


def test_url_whitespace():

    original = v2.extract_url_features_v2(
        "https://example.com/login"
    )

    whitespace = v2.extract_url_features_v2(
        "  https://example.com/login  "
    )

    assert original == whitespace


# ============================================================
# 15. SAFE NUMERICAL OPERATIONS
# ============================================================

def test_safe_ratio():

    assert v2.safe_ratio(0, 0) == 0.0

    assert v2.safe_ratio(2, 4) == pytest.approx(
        0.5
    )


# ============================================================
# 16. DATAFRAME TRANSFORMATION
# ============================================================

def test_transform_dataframe(sample_dataframe):

    transformed = v2.transform_dataframe(
        sample_dataframe
    )

    assert len(transformed) == len(
        sample_dataframe
    )

    assert len(transformed.columns) == (
        len(sample_dataframe.columns) + 31
    )

    assert set(
        v2.FEATURE_COLUMNS_V2
    ).issubset(
        transformed.columns
    )

    # Original metadata and V1 features
    # must remain unchanged.

    pd.testing.assert_frame_equal(
        transformed[sample_dataframe.columns],
        sample_dataframe,
    )

    assert np.isfinite(
        transformed[
            v2.FEATURE_COLUMNS_V2
        ].to_numpy()
    ).all()

    # Original dataframe must remain unchanged.

    assert "v2_url_length" not in (
        sample_dataframe.columns
    )


def test_transform_empty_dataframe():

    with pytest.raises(
        ValueError,
        match="Dataset cannot be empty",
    ):

        v2.transform_dataframe(
            pd.DataFrame()
        )


def test_transform_missing_url_column():

    dataframe = pd.DataFrame({
        "label": [0, 1],
    })

    with pytest.raises(
        ValueError,
        match="Missing URL column",
    ):

        v2.transform_dataframe(dataframe)


def test_transform_rejects_existing_v2_features(
    sample_dataframe,
):

    transformed = v2.transform_dataframe(
        sample_dataframe
    )

    with pytest.raises(
        ValueError,
        match="V2 features already exist",
    ):

        v2.transform_dataframe(
            transformed
        )


# ============================================================
# 17. DATASET SPLIT VALIDATION
# ============================================================

def test_validate_original_splits(
    tmp_path,
    monkeypatch,
):

    train_file = tmp_path / "train.csv"

    validation_file = tmp_path / "validation.csv"

    train_df = pd.DataFrame([
        make_record(
            "https://example.com",
            0,
            "domain:example.com",
        ),
    ])

    validation_df = pd.DataFrame([
        make_record(
            "https://example.org/login",
            1,
            "domain:example.org",
        ),
    ])

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        v2,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        v2,
        "VALIDATION_FILE",
        validation_file,
    )

    train_rows, validation_rows = (
        v2.validate_original_splits()
    )

    assert train_rows == 1

    assert validation_rows == 1


def test_validate_original_splits_detects_leakage(
    tmp_path,
    monkeypatch,
):

    train_file = tmp_path / "train.csv"

    validation_file = tmp_path / "validation.csv"

    train_df = pd.DataFrame([
        make_record(
            "https://example.com",
            0,
            "domain:example.com",
        ),
    ])

    validation_df = pd.DataFrame([
        make_record(
            "https://example.com/login",
            1,
            "domain:example.com",
        ),
    ])

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        v2,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        v2,
        "VALIDATION_FILE",
        validation_file,
    )

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        v2.validate_original_splits()


# ============================================================
# 18. CHUNKED DATASET PROCESSING
# ============================================================

def test_process_dataset(
    tmp_path,
    monkeypatch,
    sample_dataframe,
):

    input_file = tmp_path / "train.csv"

    output_file = (
        tmp_path / "v2" / "train.csv"
    )

    sample_dataframe.to_csv(
        input_file,
        index=False,
    )

    # Force processing in multiple chunks.

    monkeypatch.setattr(
        v2,
        "CHUNK_SIZE",
        1,
    )

    processed_rows = v2.process_dataset(
        input_file,
        output_file,
        expected_rows=len(sample_dataframe),
    )

    assert processed_rows == len(
        sample_dataframe
    )

    assert output_file.is_file()

    result = pd.read_csv(
        output_file,
        keep_default_na=False,
    )

    assert len(result) == len(
        sample_dataframe
    )

    assert len(result.columns) == (
        len(sample_dataframe.columns) + 31
    )

    pd.testing.assert_frame_equal(
        result[sample_dataframe.columns],
        sample_dataframe,
    )

    # Original input dataset must remain unchanged.

    original = pd.read_csv(input_file)

    pd.testing.assert_frame_equal(
        original,
        sample_dataframe,
    )

    # Temporary files must be removed.

    assert not output_file.with_suffix(
        ".csv.tmp"
    ).exists()


def test_process_dataset_row_mismatch(
    tmp_path,
    sample_dataframe,
):

    input_file = tmp_path / "train.csv"

    output_file = (
        tmp_path / "v2" / "train.csv"
    )

    sample_dataframe.to_csv(
        input_file,
        index=False,
    )

    # Simulate an existing valid output.

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        "previous_output",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unexpected number of processed rows",
    ):

        v2.process_dataset(
            input_file,
            output_file,
            expected_rows=999,
        )

    # Existing output must not be overwritten.

    assert output_file.read_text(
        encoding="utf-8"
    ) == "previous_output"

    assert not output_file.with_suffix(
        ".csv.tmp"
    ).exists()