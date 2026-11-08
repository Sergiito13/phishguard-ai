"""
Unit tests for URL structural validation.
"""

import sys
from pathlib import Path

import pytest

# Allow imports from ml/src
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from validate_urls import validate_url


@pytest.mark.parametrize(
    "url, expected_status, expected_ip",
    [
        (
            "https://example.com/login",
            "valid_structure",
            False,
        ),
        (
            "http://192.168.1.1/login",
            "valid_structure",
            True,
        ),
        (
            "example.com/login",
            "missing_http_scheme",
            False,
        ),
        (
            "https:///login",
            "missing_hostname",
            False,
        ),
        (
            "http://localhost",
            "single_label_hostname",
            False,
        ),
        (
            "http://[invalid",
            "malformed_url",
            False,
        ),
        (
            "https://example.com:abc",
            "malformed_url",
            False,
        ),
        (
            "https://bad..example.com",
            "invalid_hostname_structure",
            False,
        ),
    ],
)
def test_url_validation(url, expected_status, expected_ip):

    status, is_ip = validate_url(url)

    assert status == expected_status
    assert is_ip == expected_ip