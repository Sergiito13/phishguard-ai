"""
PhishGuard AI - URL Analysis Endpoint

REST endpoint for phishing URL detection using
the trained Machine Learning model.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
)

from app.services.model_service import analyze_url


# ============================================================
# ROUTER CONFIGURATION
# ============================================================

router = APIRouter()

logger = logging.getLogger(__name__)


# ============================================================
# URL ANALYSIS ENDPOINT
# ============================================================

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze URL",
    description=(
        "Analyze an HTTP or HTTPS URL using the "
        "experimental PhishGuard AI Machine Learning model. "
        "No target website is visited."
    ),
)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """
    Analyze a URL for potential phishing.

    Returns:
        AnalysisResponse: Model prediction and metadata.

    Raises:
        HTTPException:
            422 - Invalid URL.
            503 - Model unavailable.
            500 - Internal prediction error.
    """

    # --------------------------------------------------------
    # EXECUTE MODEL PREDICTION
    # --------------------------------------------------------

    try:

        result = analyze_url(
            url=request.url
        )

    # --------------------------------------------------------
    # INVALID URL
    # --------------------------------------------------------

    except ValueError as exc:

        logger.warning(
            "URL analysis validation or model error: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=422,
            detail="Invalid URL or incompatible model configuration.",
        ) from exc

    # --------------------------------------------------------
    # MODEL NOT FOUND
    # --------------------------------------------------------

    except FileNotFoundError as exc:

        logger.error(
            "Machine Learning model artifact not found."
        )

        raise HTTPException(
            status_code=503,
            detail="Machine Learning model is unavailable.",
        ) from exc

    # --------------------------------------------------------
    # UNEXPECTED ERRORS
    # --------------------------------------------------------

    except Exception as exc:

        logger.exception(
            "Unexpected error during URL analysis."
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected analysis error occurred.",
        ) from exc

    # --------------------------------------------------------
    # RETURN PREDICTION
    # --------------------------------------------------------

    return AnalysisResponse(**result)