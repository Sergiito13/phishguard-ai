"""
PhishGuard AI - Health Check Endpoint

Provides basic API availability information.
"""

from fastapi import APIRouter


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
        "service": "PhishGuard AI API",
        "version": "0.1.0",
    }