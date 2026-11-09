"""
Unit tests for registrable domain extraction.
"""

import sys
from pathlib import Path

import pytest


# Allow imports from ml/src
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from extract_domains import extract_domain_info


@pytest.mark.parametrize(
    "url, expected",
    [
        (
            "https://login.example.com/path",
            (
                "login.example.com",
                "example.com",
                "domain:example.com",
                "domain",
            ),
        ),
        (
            "https://accounts.example.com",
            (
                "accounts.example.com",
                "example.com",
                "domain:example.com",
                "domain",
            ),
        ),
        (
            "https://shop.example.co.uk",
            (
                "shop.example.co.uk",
                "example.co.uk",
                "domain:example.co.uk",
                "domain",
            ),
        ),
        (
            "https://usuario.github.io",
            (
                "usuario.github.io",
                "usuario.github.io",
                "domain:usuario.github.io",
                "domain",
            ),
        ),
        (
            "http://192.168.1.1",
            (
                "192.168.1.1",
                "",
                "ip:192.168.1.1",
                "ip",
            ),
        ),
        (
            "http://[2001:db8::1]/",
            (
                "2001:db8::1",
                "",
                "ip:2001:db8::1",
                "ip",
            ),
        ),
        (
            "example.com/login",
            (
                "",
                "",
                "",
                "missing_http_scheme",
            ),
        ),
        (
            "http://localhost",
            (
                "",
                "",
                "",
                "single_label_hostname",
            ),
        ),
    ],
)
def test_extract_domain_info(url, expected):

    result = extract_domain_info(url)

    assert result == expected