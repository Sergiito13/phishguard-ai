"""
PhishGuard AI - Save Experimental ML Model

Trains a HistGradientBoosting classifier and saves it
for future URL predictions.

Training dataset:
    ml/data/splits/train.csv

Validation dataset:
    ml/data/splits/validation.csv

Output:
    ml/models/hgb_baseline.joblib

The test dataset is never used.

WARNING:
    This is an experimental, uncalibrated model.
    It is not ready for production security decisions.
"""

from pathlib import Path

import joblib
import numpy as np

from compare_models import create_models, calculate_metrics
from train_model import load_dataset, TRAIN_FILE, VALIDATION_FILE
from extract_features import FEATURE_COLUMNS


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "hgb_baseline.joblib"

MODEL_NAME = "HistGradientBoosting"

MODEL_VERSION = "0.1.0-experimental"

THRESHOLD = 0.50


# ============================================================
# 2. TRAIN AND SAVE MODEL
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - SAVE EXPERIMENTAL MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading datasets...")

    X_train, y_train = load_dataset(TRAIN_FILE)

    X_validation, y_validation = load_dataset(
        VALIDATION_FILE
    )

    print(f"Training URLs: {len(X_train):,}")

    print(
        f"Validation URLs: {len(X_validation):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # CREATE AND TRAIN MODEL
    # --------------------------------------------------------

    print("\nCreating model...")

    model = create_models()[MODEL_NAME]

    print("Training HistGradientBoosting...")

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    print("\nEvaluating model...")

    predictions = model.predict(X_validation)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    metrics = calculate_metrics(
        y_validation,
        predictions,
        probabilities,
    )

    print("\nVALIDATION METRICS")

    for name, value in metrics.items():

        if isinstance(value, (int, np.integer)):
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value:.4f}")

    # --------------------------------------------------------
    # MODEL PACKAGE
    # --------------------------------------------------------

    model_package = {
        "model": model,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": THRESHOLD,
        "calibrated": False,
        "production_ready": False,
        "training_samples": len(X_train),
        "validation_samples": len(X_validation),
        "validation_metrics": metrics,
        "label_mapping": {
            0: "legitimate",
            1: "phishing",
        },
    }

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_package,
        MODEL_FILE,
        compress=3,
    )

    print("\nMODEL SAVED")

    print(f"Path: {MODEL_FILE}")

    print(f"Version: {MODEL_VERSION}")

    print("\nWARNING: Model probabilities are not calibrated.")

    print("The model is experimental.")

    print("\nTest dataset has not been used.")


# ============================================================
# 3. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()