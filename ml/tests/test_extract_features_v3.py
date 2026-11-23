"""Unit tests for PhishGuard AI Feature Engineering V3.

All tests use synthetic URLs.
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

SRC_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR),
)

import extract_features_v2 as v2
import extract_features_v3 as v3


# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

def test_configuration():

    assert len(
        v2.FEATURE_COLUMNS_V2
    ) == 31

    assert len(
        v3.FEATURE_COLUMNS_V3
    ) == 6

    assert len(
        v3.FEATURE_COLUMNS_EXTENDED_V3
    ) == 37

    assert len(
        set(v3.FEATURE_COLUMNS_EXTENDED_V3)
    ) == 37

    assert (
        v3.FEATURE_COLUMNS_EXTENDED_V3[:31]
        == v2.FEATURE_COLUMNS_V2
    )


# ============================================================
# 3. BASIC EXTRACTION
# ============================================================

def test_feature_order_and_values():

    features = v3.extract_url_features_v3(
        "https://example.com/login"
    )

    assert list(features) == (
        v3.FEATURE_COLUMNS_V3
    )

    assert len(features) == 6

    assert np.isfinite(
        list(features.values())
    ).all()


# ============================================================
# 4. HOSTNAME AUTHENTICATION TOKENS
# ============================================================

def test_hostname_auth_tokens():

    features = v3.extract_url_features_v3(
        "https://secure-login.example.com/home"
    )

    assert features[
        "v3_hostname_auth_tokens"
    ] == 2

    assert features[
        "v3_path_auth_tokens"
    ] == 0


def test_auth_requires_complete_tokens():

    features = v3.extract_url_features_v3(
        "https://loginfoo.example.com/accounts"
    )

    assert features[
        "v3_hostname_auth_tokens"
    ] == 0

    assert features[
        "v3_path_auth_tokens"
    ] == 0


# ============================================================
# 5. PATH AUTHENTICATION TOKENS
# ============================================================

def test_path_auth_tokens():

    features = v3.extract_url_features_v3(
        "https://example.com/account/reset/profile"
    )

    assert features[
        "v3_hostname_auth_tokens"
    ] == 0

    assert features[
        "v3_path_auth_tokens"
    ] == 2


def test_repeated_auth_tokens_count():

    features = v3.extract_url_features_v3(
        "https://login.example.com/login/login"
    )

    assert features[
        "v3_hostname_auth_tokens"
    ] == 1

    assert features[
        "v3_path_auth_tokens"
    ] == 2


# ============================================================
# 6. PATH SEGMENT LENGTHS
# ============================================================

def test_path_segment_lengths():

    features = v3.extract_url_features_v3(
        "https://example.com/a/bbbb/ccc"
    )

    assert features[
        "v3_avg_path_segment_length"
    ] == pytest.approx(
        8 / 3
    )

    assert features[
        "v3_max_path_segment_length"
    ] == 4


def test_empty_path_segments_ignored():

    features = v3.extract_url_features_v3(
        "https://example.com//a///bbb/"
    )

    assert features[
        "v3_avg_path_segment_length"
    ] == pytest.approx(2.0)

    assert features[
        "v3_max_path_segment_length"
    ] == 3


# ============================================================
# 7. EMPTY PATH
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://example.com/",
        "example.com?empty=",
    ],
)
def test_empty_path(url):

    features = v3.extract_url_features_v3(
        url
    )

    assert features[
        "v3_avg_path_segment_length"
    ] == 0

    assert features[
        "v3_max_path_segment_length"
    ] == 0

    assert features[
        "v3_path_digit_ratio"
    ] == 0


# ============================================================
# 8. PATH DIGIT RATIO
# ============================================================

def test_path_digit_ratio():

    features = v3.extract_url_features_v3(
        "https://example.com/a1/b23"
    )

    assert features[
        "v3_path_digit_ratio"
    ] == pytest.approx(
        3 / 7
    )


# ============================================================
# 9. QUERY VALUE ENTROPY
# ============================================================

def test_query_value_entropy():

    uniform = v3.extract_url_features_v3(
        "https://example.com/?session=aaaa"
    )

    varied = v3.extract_url_features_v3(
        "https://example.com/?session=abcd"
    )

    assert uniform[
        "v3_query_value_entropy"
    ] == pytest.approx(0.0)

    assert varied[
        "v3_query_value_entropy"
    ] == pytest.approx(2.0)


def test_query_parameter_names_ignored():

    first = v3.extract_url_features_v3(
        "https://example.com/?token=abcd"
    )

    second = v3.extract_url_features_v3(
        "https://example.com/?different_name=abcd"
    )

    assert first[
        "v3_query_value_entropy"
    ] == second[
        "v3_query_value_entropy"
    ]


def test_query_values_percent_decoded():

    escaped = v3.extract_url_features_v3(
        "https://example.com/?v=a%62cd"
    )

    plain = v3.extract_url_features_v3(
        "https://example.com/?v=abcd"
    )

    assert escaped[
        "v3_query_value_entropy"
    ] == plain[
        "v3_query_value_entropy"
    ]


def test_empty_query_value_entropy():

    urls = [
        "https://example.com/",
        "https://example.com/?x=&y=",
    ]

    for url in urls:

        features = v3.extract_url_features_v3(
            url
        )

        assert features[
            "v3_query_value_entropy"
        ] == 0


# ============================================================
# 10. HTTP/HTTPS INDEPENDENCE
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "example.com/login",
        "secure.example.com/account/reset?session=abc123",
        "example.com/a1/b23?x=abcd&y=efgh",
        "192.0.2.1/login",
        "example.com/path#fragment",
    ],
)
def test_protocol_independence(url):

    http = v3.extract_url_features_v3(
        f"http://{url}"
    )

    https = v3.extract_url_features_v3(
        f"https://{url}"
    )

    no_scheme = v3.extract_url_features_v3(
        url
    )

    assert http == https == no_scheme


# ============================================================
# 11. ROOT PATH NORMALIZATION
# ============================================================

@pytest.mark.parametrize(
    "left,right",
    [
        (
            "https://example.com",
            "https://example.com/",
        ),
        (
            "https://example.com?x=abcd",
            "https://example.com/?x=abcd",
        ),
    ],
)
def test_root_path_normalization(
    left,
    right,
):

    assert (
        v3.extract_url_features_v3(left)
        == v3.extract_url_features_v3(right)
    )


# ============================================================
# 12. INVALID URLS
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
        "https:///login",
        "ftp://example.com",
        "https://example.com:99999",
    ],
)
def test_invalid_urls(url):

    with pytest.raises(ValueError):

        v3.extract_url_features_v3(
            url
        )


# ============================================================
# 13. V2 COMPATIBILITY
# ============================================================

def test_composite_preserves_v2():

    url = (
        "https://secure-login.example.com/"
        "account?token=abcd"
    )

    v2_values = v2.extract_url_features_v2(
        url
    )

    combined = v3.extract_extended_features_v3(
        url
    )

    assert list(combined) == (
        v3.FEATURE_COLUMNS_EXTENDED_V3
    )

    assert {
        key: combined[key]
        for key in v2_values
    } == v2_values

    assert len(combined) == 37


# ============================================================
# 14. SYNTHETIC DATAFRAME
# ============================================================

@pytest.fixture
def sample_frame():

    urls = [
        "https://example.com/login",
        "https://shop.example.org/a1/b23?x=abcd",
    ]

    v2_rows = [
        v2.extract_url_features_v2(url)
        for url in urls
    ]

    metadata = pd.DataFrame({
        "url": urls,
        "label": [0, 1],
    })

    features = pd.DataFrame.from_records(
        v2_rows
    )

    return pd.concat(
        [
            metadata,
            features,
        ],
        axis=1,
    )


# ============================================================
# 15. DATAFRAME TRANSFORMATION
# ============================================================

def test_transform_dataframe_v3(
    sample_frame,
):

    original = sample_frame.copy(
        deep=True
    )

    transformed = v3.transform_dataframe_v3(
        sample_frame
    )

    assert len(transformed) == len(
        sample_frame
    )

    assert set(
        v3.FEATURE_COLUMNS_V3
    ).issubset(
        transformed.columns
    )

    assert transformed[
        "url"
    ].tolist() == sample_frame[
        "url"
    ].tolist()

    assert transformed[
        "label"
    ].tolist() == [0, 1]

    pd.testing.assert_frame_equal(
        sample_frame,
        original,
    )

    assert np.isfinite(
        transformed[
            v3.FEATURE_COLUMNS_V3
        ].to_numpy()
    ).all()


def test_transform_empty(
    sample_frame,
):

    with pytest.raises(
        ValueError,
        match="empty",
    ):

        v3.transform_dataframe_v3(
            sample_frame.iloc[:0]
        )


def test_transform_missing_url(
    sample_frame,
):

    with pytest.raises(
        ValueError,
        match="URL",
    ):

        v3.transform_dataframe_v3(
            sample_frame.drop(
                columns=["url"]
            )
        )


def test_transform_missing_v2(
    sample_frame,
):

    with pytest.raises(
        ValueError,
        match="Missing V2",
    ):

        v3.transform_dataframe_v3(
            sample_frame.drop(
                columns=[
                    v2.FEATURE_COLUMNS_V2[0]
                ]
            )
        )


def test_transform_duplicate_v3(
    sample_frame,
):

    frame = sample_frame.copy()

    frame[
        v3.FEATURE_COLUMNS_V3[0]
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="already exist",
    ):

        v3.transform_dataframe_v3(
            frame
        )


# ============================================================
# 16. INVALID V2 FEATURE VALIDATION
# ============================================================

def test_transform_invalid_v2(
    sample_frame,
):

    frame = sample_frame.copy()

    # Convert the column to float before assigning np.inf.
    # This avoids pandas FutureWarning about incompatible dtypes.
    feature = v2.FEATURE_COLUMNS_V2[0]

    frame[feature] = frame[feature].astype(float)

    frame.loc[0, feature] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):

        v3.transform_dataframe_v3(
            frame
        )