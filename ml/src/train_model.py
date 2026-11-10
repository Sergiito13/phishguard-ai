"""
PhishGuard AI - Baseline Model Training

Train and validate a Logistic Regression classifier.

Labels:
    0 = Legitimate
    1 = Phishing

The test dataset is intentionally not used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import FEATURE_COLUMNS


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data" / "splits" / "train.csv"

VALIDATION_FILE = (
    BASE_DIR / "data" / "splits" / "validation.csv"
)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_dataset(file_path: Path):

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        usecols=["label", *FEATURE_COLUMNS],
    )

    if df.empty:
        raise ValueError("Dataset is empty")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid labels found")

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )

    y = df["label"].astype(int).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain invalid numerical values"
        )

    return X, y


# ============================================================
# 3. CREATE MODEL
# ============================================================

def create_model():

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# ============================================================
# 4. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "Average Precision": average_precision_score(
            y, probabilities
        ),
    }

    for name, value in metrics.items():
        print(f"{name:<20}: {value:.4f}")

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            digits=4,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")

    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
    )

    return metrics


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHISHGUARD AI - BASELINE MODEL TRAINING")
    print("=" * 60)

    print("\nModel: Logistic Regression")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nLoading datasets...")

    X_train, y_train = load_dataset(
        TRAIN_FILE
    )

    X_validation, y_validation = load_dataset(
        VALIDATION_FILE
    )

    print(f"Training samples: {len(X_train):,}")

    print(
        f"Validation samples: {len(X_validation):,}"
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print("\nTraining model...")

    model = create_model()

    model.fit(X_train, y_train)

    print("Training completed successfully.")

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print("\nBaseline experiment completed.")
    print("Test dataset has not been used.")


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()