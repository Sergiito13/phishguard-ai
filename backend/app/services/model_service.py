"""
PhishGuard AI - Machine Learning Service

Service responsible for loading the trained model
and performing URL phishing predictions.

The model is loaded once and cached for subsequent requests.
"""

import sys

from functools import lru_cache
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_SRC = PROJECT_ROOT / "ml" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


# ============================================================
# MACHINE LEARNING IMPORTS
# ============================================================

from predict_url import load_model, predict_url  # noqa: E402


# ============================================================
# MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def get_model() -> dict:
    """
    Load the trained ML model.

    The model is cached to avoid loading it from disk
    for every incoming request.

    Returns:
        dict: Trained model package and metadata.
    """

    return load_model()


# ============================================================
# URL PREDICTION
# ============================================================

def analyze_url(url: str) -> dict:
    """
    Analyze a URL using the trained phishing detection model.

    Args:
        url (str): HTTP or HTTPS URL to analyze.

    Returns:
        dict: Prediction result and model metadata.

    Raises:
        ValueError: If the URL is invalid.
        FileNotFoundError: If the model artifact is missing.
    """

    model_package = get_model()

    result = predict_url(
        url=url,
        package=model_package,
    )

    return result