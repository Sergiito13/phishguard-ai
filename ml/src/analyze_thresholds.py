"""
PhishGuard AI - Classification Threshold Analysis

Compare different classification thresholds for two models:

1. Baseline model with 20 features.
2. Model without the HTTPS feature (19 features).

Training:
    All sources except URL_PHISH.

Validation:
    URL_PHISH only.

This is an exploratory experiment.

The test dataset is never accessed.
No trained models are saved or overwritten.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from ablation_https import (
    load_data,
    EXPERIMENTS,
)

from compare_models import create_models


# ============================================================
# 1. CONFIGURATION
# ============================================================

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]


# ============================================================
# 2. ANALYZE SCORE DISTRIBUTION
# ============================================================

def analyze_score_distribution(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    """
    Analyze predicted score distributions for legitimate
    and phishing URLs.
    """

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION")
    print("=" * 70)

    for label, name in [
        (0, "LEGITIMATE"),
        (1, "PHISHING"),
    ]:

        scores = probabilities[y_true == label]

        if len(scores) == 0:
            continue

        print(f"\n{name} URLS")
        print("-" * 40)

        print(f"Samples: {len(scores):,}")

        print(f"Mean:    {np.mean(scores):.4f}")
        print(f"Median:  {np.median(scores):.4f}")

        print(
            f"Minimum: {np.min(scores):.4f}"
        )

        print(
            f"Maximum: {np.max(scores):.4f}"
        )

        print(
            f"P10:     {np.percentile(scores, 10):.4f}"
        )

        print(
            f"P25:     {np.percentile(scores, 25):.4f}"
        )

        print(
            f"P75:     {np.percentile(scores, 75):.4f}"
        )

        print(
            f"P90:     {np.percentile(scores, 90):.4f}"
        )


# ============================================================
# 3. EVALUATE CLASSIFICATION THRESHOLDS
# ============================================================

def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, F1 and classification
    errors for each configured threshold.
    """

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": false_positive_rate,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

    return pd.DataFrame(results)


# ============================================================
# 4. PRINT THRESHOLD RESULTS
# ============================================================

def print_threshold_results(
    results_df: pd.DataFrame,
):
    """
    Print a formatted threshold comparison table.
    """

    print("\n" + "=" * 90)
    print("THRESHOLD COMPARISON")
    print("=" * 90)

    columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "fpr",
        "false_positives",
        "false_negatives",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "fpr": "{:.4f}".format,
                "false_positives": "{:,}".format,
                "false_negatives": "{:,}".format,
            },
        )
    )


# ============================================================
# 5. TRAIN AND ANALYZE MODEL
# ============================================================

def analyze_model(
    experiment_name: str,
    features: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Train a HistGradientBoosting model and analyze
    its classification thresholds.
    """

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)

    print(f"Features: {len(features)}")

    # Prepare training data.

    X_train = train_df[
        features
    ].to_numpy(dtype=np.float64)

    y_train = train_df[
        "label"
    ].to_numpy(dtype=int)

    # Prepare validation data.

    X_validation = validation_df[
        features
    ].to_numpy(dtype=np.float64)

    y_validation = validation_df[
        "label"
    ].to_numpy(dtype=int)

    # Create a new model.

    model = create_models()[
        "HistGradientBoosting"
    ]

    # Train model.

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed successfully.")

    # Predict scores.

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Evaluate ranking quality.

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    average_precision = average_precision_score(
        y_validation,
        probabilities,
    )

    print("\nRANKING METRICS")
    print("-" * 40)

    print(f"ROC-AUC:           {roc_auc:.4f}")

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    # Analyze score distribution.

    analyze_score_distribution(
        y_validation,
        probabilities,
    )

    # Evaluate thresholds.

    results_df = evaluate_thresholds(
        y_validation,
        probabilities,
    )

    print_threshold_results(
        results_df
    )

    return results_df


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHISHGUARD AI - THRESHOLD ANALYSIS")
    print("=" * 70)

    # Load previously prepared datasets.

    train_df, validation_df = load_data()

    print(f"\nTraining URLs:   {len(train_df):,}")
    print(f"Validation URLs: {len(validation_df):,}")

    # Analyze each experiment.

    for experiment_name, features in EXPERIMENTS.items():

        analyze_model(
            experiment_name,
            features,
            train_df,
            validation_df,
        )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nNo model artifacts were modified.")
    print("The test dataset was not accessed.")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()