"""
PhishGuard AI - Health Endpoint Tests

Automated tests for API availability and root endpoints.
"""

from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# HEALTH ENDPOINT TEST
# ============================================================

def test_health_check():

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "service": "PhishGuard AI API",
        "version": "0.1.0",
    }


# ============================================================
# ROOT ENDPOINT TEST
# ============================================================

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "PhishGuard AI API"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"
    assert data["documentation"] == "/docs"