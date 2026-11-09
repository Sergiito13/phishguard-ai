"""
PhishGuard AI - URL Feature Engineering Unit Tests

Tests:
- Extraction of 20 numerical features
- Feature values and data types
- HTTP and HTTPS detection
- IPv4 and IPv6 detection
- Query parameter counting
- Subdomain counting
- Private suffix handling
- Punycode detection
- Shannon entropy calculation
- Whitespace normalization
- Special character counting
- Invalid input handling
"""

import math
import sys
from pathlib import Path

import pytest


# ============================================================
# 1. IMPORT PROJECT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

from extract_features import (
    FEATURE_COLUMNS,
    calculate_entropy,
    extract_url_features,
)


# ============================================================
# 2. TEST FEATURE EXTRACTION
# ============================================================

def test_extract_url_features():
    """
    Verify the numerical features extracted from a known URL.
    """

    url = (
        "https://secure-login123.example.com/"
        "verify?account=456"
    )

    features = extract_url_features(url)

    assert len(features) == 20

    assert set(features.keys()) == set(FEATURE_COLUMNS)

    assert features["url_length"] == 54
    assert features["hostname_length"] == 27

    assert features["path_length"] == 7
    assert features["query_length"] == 11

    assert features["digit_count"] == 6
    assert features["letter_count"] == 39

    assert features["dot_count"] == 2
    assert features["hyphen_count"] == 1

    assert features["at_count"] == 0
    assert features["percent_count"] == 0

    assert features["slash_count"] == 3
    assert features["question_mark_count"] == 1

    assert features["query_param_count"] == 1
    assert features["subdomain_count"] == 1

    assert features["is_https"] == 1
    assert features["has_ip"] == 0
    assert features["has_punycode"] == 0

    assert features["url_entropy"] == pytest.approx(
        4.75721522473889
    )

    assert features["digit_ratio"] == pytest.approx(
        6 / 54
    )

    assert features["special_char_ratio"] == pytest.approx(
        9 / 54
    )

    # Every feature must be numeric and finite
    for value in features.values():

        assert isinstance(value, (int, float))
        assert math.isfinite(value)


# ============================================================
# 3. TEST HTTP, HTTPS AND IP ADDRESSES
# ============================================================

@pytest.mark.parametrize(
    "url, expected_https, expected_ip",
    [
        (
            "https://example.com",
            1,
            0,
        ),
        (
            "http://192.168.1.1/login",
            0,
            1,
        ),
        (
            "http://[2001:db8::1]/",
            0,
            1,
        ),
    ],
)
def test_protocol_and_ip_detection(
    url,
    expected_https,
    expected_ip,
):
    """
    Verify protocol and IP address features.
    """

    features = extract_url_features(url)

    assert features["is_https"] == expected_https

    assert features["has_ip"] == expected_ip


# ============================================================
# 4. TEST QUERY PARAMETERS
# ============================================================

@pytest.mark.parametrize(
    "url, expected_count",
    [
        (
            "https://example.com/path?a=1&b=&a=2",
            3,
        ),
        (
            "https://example.com/path",
            0,
        ),
    ],
)
def test_query_parameter_count(url, expected_count):
    """
    Verify query parameters are counted correctly.
    """

    features = extract_url_features(url)

    assert features["query_param_count"] == expected_count


# ============================================================
# 5. TEST SUBDOMAIN COUNTING
# ============================================================

@pytest.mark.parametrize(
    "url, expected_count",
    [
        (
            "https://shop.example.co.uk",
            1,
        ),
        (
            "https://a.b.example.com",
            2,
        ),
        (
            "https://usuario.github.io",
            0,
        ),
    ],
)
def test_subdomain_count(url, expected_count):
    """
    Verify subdomain counting using the Public Suffix List.
    """

    features = extract_url_features(url)

    assert features["subdomain_count"] == expected_count


# ============================================================
# 6. TEST PUNYCODE DETECTION
# ============================================================

@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://xn--bcher-kva.example",
            1,
        ),
        (
            "https://example.com",
            0,
        ),
    ],
)
def test_punycode_detection(url, expected):
    """
    Verify Punycode detection.
    """

    features = extract_url_features(url)

    assert features["has_punycode"] == expected


# ============================================================
# 7. TEST SHANNON ENTROPY
# ============================================================

@pytest.mark.parametrize(
    "value, expected",
    [
        ("", 0.0),
        ("aaaa", 0.0),
        ("ab", 1.0),
    ],
)
def test_entropy_calculation(value, expected):
    """
    Verify Shannon entropy calculations.
    """

    result = calculate_entropy(value)

    assert result == pytest.approx(expected)


# ============================================================
# 8. TEST INVALID INPUTS
# ============================================================

@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        None,
    ],
)
def test_invalid_url_input(url):
    """
    Empty or invalid inputs must raise ValueError.
    """

    with pytest.raises(ValueError):

        extract_url_features(url)


# ============================================================
# 9. TEST WHITESPACE HANDLING
# ============================================================

def test_url_whitespace():
    """
    Exterior whitespace must not affect the features.
    """

    original = "https://example.com/login"

    with_spaces = "  https://example.com/login  "

    original_features = extract_url_features(original)

    spaced_features = extract_url_features(with_spaces)

    assert original_features == spaced_features


# ============================================================
# 10. TEST SPECIAL CHARACTERS
# ============================================================

def test_special_character_counts():
    """
    Verify URL special character counting.
    """

    url = "https://example.com/a-b%20?x=1&y="

    features = extract_url_features(url)

    assert features["hyphen_count"] == 1
    assert features["percent_count"] == 1
    assert features["question_mark_count"] == 1
    assert features["slash_count"] == 3

    assert features["query_param_count"] == 2