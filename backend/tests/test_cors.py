"""
PhishGuard AI - CORS Tests

Validate frontend and Firefox extension origins.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

ENDPOINT = "/api/v1/analyze"


# ============================================================
# TEST HELPERS
# ============================================================

def preflight(origin: str):

    return client.options(
        ENDPOINT,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


# ============================================================
# REACT FRONTEND
# ============================================================

def test_localhost_cors():

    origin = "http://localhost:5173"

    response = preflight(origin)

    assert response.status_code == 200

    assert (
        response.headers["access-control-allow-origin"]
        == origin
    )


def test_local_ip_cors():

    origin = "http://127.0.0.1:5173"

    response = preflight(origin)

    assert response.status_code == 200

    assert (
        response.headers["access-control-allow-origin"]
        == origin
    )


# ============================================================
# FIREFOX EXTENSION
# ============================================================

def test_firefox_extension_cors():

    origin = (
        "moz-extension://"
        "a31f8b20-5b52-44e1-a518-2c789a124abc"
    )

    response = preflight(origin)

    assert response.status_code == 200

    assert (
        response.headers["access-control-allow-origin"]
        == origin
    )


def test_firefox_extension_different_uuid():

    origin = (
        "moz-extension://"
        "b45a3120-7c32-4e51-8c11-6d243a751abc"
    )

    response = preflight(origin)

    assert response.status_code == 200


# ============================================================
# DISALLOWED ORIGINS
# ============================================================

def test_external_origin_rejected():

    response = preflight(
        "https://untrusted.example"
    )

    assert response.status_code == 400

    assert (
        "access-control-allow-origin"
        not in response.headers
    )


def test_invalid_extension_origin_rejected():

    response = preflight(
        "moz-extension://invalid-origin"
    )

    assert response.status_code == 400