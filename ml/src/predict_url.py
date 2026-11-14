"""
PhishGuard AI - Experimental URL Prediction

Predicts whether a URL is potentially phishing
using a previously trained ML model.

Usage:
    python ml/src/predict_url.py "https://example.com"

WARNING:
    Predictions are experimental.
    Model scores are not calibrated probabilities.
    No website is visited during analysis.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from extract_features import (
    FEATURE_COLUMNS,
    extract_url_features,
)

from validate_urls import validate_url


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR / "models" / "hgb_baseline.joblib"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.is_file():

        raise FileNotFoundError(
            "Trained model not found. "
            "Run save_model.py first."
        )

    package = joblib.load(MODEL_FILE)

    if not isinstance(package, dict):
        raise ValueError("Invalid model package")

    required_keys = {
        "model",
        "feature_columns",
        "threshold",
        "calibrated",
    }

    if not required_keys.issubset(package):
        raise ValueError(
            "Model package is missing required metadata"
        )

    if package["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError(
            "Feature mismatch between model and extractor"
        )

    return package


# ============================================================
# 3. PREDICT URL
# ============================================================

def predict_url(url: str, package: dict) -> dict:

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not isinstance(url, str) or not url.strip():

        raise ValueError(
            "URL must be a non-empty string"
        )

    url = url.strip()

    status, _ = validate_url(url)

    if status != "valid_structure":

        raise ValueError(
            f"Unsupported URL structure: {status}. "
            "Provide a complete HTTP or HTTPS URL."
        )

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    features = extract_url_features(url)

    feature_columns = package["feature_columns"]

    values = np.array(
        [
            [
                features[name]
                for name in feature_columns
            ]
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "URL generated invalid numerical features"
        )

    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    model = package["model"]

    probability = float(
        model.predict_proba(values)[0, 1]
    )

    threshold = float(
        package["threshold"]
    )

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = {
        "url": url,
        "prediction": (
            "potential_phishing"
            if prediction == 1
            else "not_flagged"
        ),
        "phishing_score": round(
            probability,
            6,
        ),
        "threshold": threshold,
        "calibrated": bool(
            package["calibrated"]
        ),
        "model_version": package.get(
            "model_version",
            "unknown",
        ),
        "features_analyzed": len(feature_columns),
    }

    return result


# ============================================================
# 4. COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PhishGuard AI - Experimental URL Detector"
        )
    )

    parser.add_argument(
        "url",
        help="Complete HTTP or HTTPS URL to analyze",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PHISHGUARD AI - URL ANALYSIS")
    print("=" * 60)

    try:

        package = load_model()

        result = predict_url(
            args.url,
            package,
        )

    except (ValueError, FileNotFoundError) as error:

        parser.exit(
            status=1,
            message=f"Error: {error}\n",
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(f"\nURL: {result['url']}")

    print(
        f"Model version: {result['model_version']}"
    )

    print(
        f"Features analyzed: {result['features_analyzed']}"
    )

    print("\nRESULT")

    if result["prediction"] == "potential_phishing":

        print("Classification: POTENTIAL PHISHING")

    else:

        print("Classification: NOT FLAGGED")

    print(
        "Experimental phishing score: "
        f"{result['phishing_score'] * 100:.2f}%"
    )

    print(
        f"Decision threshold: {result['threshold']:.2f}"
    )

    print("\nIMPORTANT")

    print(
        "This score is not a calibrated probability."
    )

    print(
        "A URL not flagged by this model is not "
        "necessarily safe."
    )

    print(
        "The model has known cross-dataset "
        "generalization limitations."
    )

    print("\nNo website was visited.")


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()