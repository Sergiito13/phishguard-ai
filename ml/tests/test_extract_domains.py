"""
PhishGuard AI - Domain Extraction Unit Tests

Tests:
- Standard registrable domains
- Subdomains
- Compound public suffixes
- Private suffix domains
- Private suffix roots
- IPv4 and IPv6 addresses
- Unsupported URL formats
"""

import sys
from pathlib import Path

import pytest


# ============================================================
# 1. IMPORT PROJECT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from extract_domains import extract_domain_info


# ============================================================
# 2. STANDARD DOMAIN EXTRACTION TESTS
# ============================================================

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
    """
    Verify standard domain extraction and grouping keys.
    """

    result = extract_domain_info(url)

    assert result == expected


# ============================================================
# 3. PRIVATE SUFFIX ROOT TESTS
# ============================================================

@pytest.mark.parametrize(
    "url, expected_hostname",
    [
        (
            "https://s3.amazonaws.com/bucket/file.html",
            "s3.amazonaws.com",
        ),
        (
            "https://cloudflare-ipfs.com/ipfs/example",
            "cloudflare-ipfs.com",
        ),
    ],
)
def test_private_suffix_root(url, expected_hostname):
    """
    Verify private suffix roots are grouped by hostname.
    """

    hostname, domain, group_key, status = extract_domain_info(url)

    assert hostname == expected_hostname
    assert domain == ""
    assert group_key == f"host:{expected_hostname}"
    assert status == "private_suffix_root"