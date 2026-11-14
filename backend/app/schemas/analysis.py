"""
PhishGuard AI - Analysis Schemas

Pydantic models for URL analysis requests and responses.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AnalysisRequest(BaseModel):
    """
    Request body for phishing URL analysis.
    """

    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Complete HTTP or HTTPS URL to analyze.",
        examples=[
            "https://example.com"
        ],
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class AnalysisResponse(BaseModel):
    """
    Response returned by the phishing detection model.
    """

    url: str = Field(
        description="Analyzed URL."
    )

    prediction: Literal[
        "potential_phishing",
        "not_flagged",
    ] = Field(
        description="Classification returned by the ML model."
    )

    phishing_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Experimental phishing score. "
            "This is not a calibrated probability."
        ),
    )

    threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification decision threshold."
    )

    calibrated: bool = Field(
        description="Whether model scores are calibrated."
    )

    model_version: str = Field(
        description="Version of the Machine Learning model."
    )

    features_analyzed: int = Field(
        ge=0,
        description="Number of URL features analyzed."
    )