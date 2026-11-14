"""
PhishGuard AI - FastAPI Application

Main entry point for the phishing detection REST API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    health,
    analyze,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="PhishGuard AI API",
    description=(
        "REST API for phishing URL detection "
        "powered by Machine Learning."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# API ROUTERS
# ============================================================

# Health check endpoint

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["Health"],
)


# Phishing analysis endpoint

app.include_router(
    analyze.router,
    prefix="/api/v1",
    tags=["Analysis"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["Root"])
def root():
    """
    Return basic API information.
    """

    return {
        "name": "PhishGuard AI API",
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs",
    }