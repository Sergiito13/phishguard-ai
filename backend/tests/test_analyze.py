"""
PhishGuard AI - URL Analysis Endpoint Tests

Tests for phishing URL analysis API.
"""

from importlib import import_module

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

analyze_route = import_module("app.api.routes.analyze")


# ============================================================
# SUCCESSFUL ANALYSIS
# ============================================================

def test_analyze_valid_url(monkeypatch):

    def mock_analyze(url):
        return {
            "url": url,
            "prediction": "not_flagged",
            "phishing_score": 0.01,
            "threshold": 0.5,
            "calibrated": False,
            "model_version": "0.1.0-experimental",
            "features_analyzed": 20,
        }

    monkeypatch.setattr(
        analyze_route,
        "analyze_url",
        mock_analyze,
    )

    response = client.post(
        "/api/v1/analyze",
        json={"url": "https://example.com"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "not_flagged"
    assert data["phishing_score"] == 0.01
    assert data["features_analyzed"] == 20


# ============================================================
# EMPTY URL
# ============================================================

def test_analyze_empty_url():

    response = client.post(
        "/api/v1/analyze",
        json={"url": ""},
    )

    assert response.status_code == 422


# ============================================================
# MISSING URL
# ============================================================

def test_analyze_missing_url():

    response = client.post(
        "/api/v1/analyze",
        json={},
    )

    assert response.status_code == 422


# ============================================================
# INVALID URL
# ============================================================

def test_analyze_invalid_url(monkeypatch):

    def mock_analyze(url):
        raise ValueError("Invalid URL")

    monkeypatch.setattr(
        analyze_route,
        "analyze_url",
        mock_analyze,
    )

    response = client.post(
        "/api/v1/analyze",
        json={"url": "invalid-url"},
    )

    assert response.status_code == 422


# ============================================================
# MODEL UNAVAILABLE
# ============================================================

def test_model_unavailable(monkeypatch):

    def mock_analyze(url):
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr(
        analyze_route,
        "analyze_url",
        mock_analyze,
    )

    response = client.post(
        "/api/v1/analyze",
        json={"url": "https://example.com"},
    )

    assert response.status_code == 503


# ============================================================
# INTERNAL ERROR
# ============================================================

def test_internal_error(monkeypatch):

    def mock_analyze(url):
        raise RuntimeError("Prediction failed")

    monkeypatch.setattr(
        analyze_route,
        "analyze_url",
        mock_analyze,
    )

    response = client.post(
        "/api/v1/analyze",
        json={"url": "https://example.com"},
    )

    assert response.status_code == 500