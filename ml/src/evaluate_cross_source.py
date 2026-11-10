"""
PhishGuard AI - Cross-Source Generalization Evaluation

Evaluates HistGradientBoosting using a leave-one-source-out
strategy on the existing training and validation splits.

For each experiment:
- Exclude the target source from training.
- Train using the remaining sources.
- Evaluate on validation URLs belonging to the target source.

The test dataset is never accessed.
"""

import numpy as np
import pandas as pd

from compare_models import (
    create_models,
    calculate_metrics,
    TRAIN_FILE,
    VALIDATION_FILE,
)

from extract_features import FEATURE_COLUMNS


# ============================================================
# CONFIGURATION
# ============================================================

SOURCES = [
    "legitphish",
    "phiusiil",
    "phishbd",
    "url_phish",
]

COLUMNS = [
    "label",
    "sources",
    "group_key",
    *FEATURE_COLUMNS,
]


# ============================================================
# SOURCE DETECTION
# ============================================================

def belongs_to_source(df, source):
    """
    Return a boolean mask identifying rows belonging
    to the specified dataset source.
    """

    return (
        df["sources"]
        .str.split("|", regex=False)
        .apply(lambda values: source in values)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHISHGUARD AI - LEAVE-ONE-SOURCE-OUT EVALUATION")
    print("=" * 70)

    train_df = pd.read_csv(
        TRAIN_FILE,
        usecols=COLUMNS,
        keep_default_na=False,
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE,
        usecols=COLUMNS,
        keep_default_na=False,
    )

    results = []

    # ========================================================
    # EVALUATE EACH SOURCE
    # ========================================================

    for source in SOURCES:

        print("\n" + "=" * 70)
        print(f"EXCLUDED SOURCE: {source.upper()}")
        print("=" * 70)

        # Remove every training URL associated with
        # the source currently being evaluated.

        train_mask = ~belongs_to_source(
            train_df,
            source,
        )

        filtered_train = train_df.loc[
            train_mask
        ].copy()

        # Evaluate only URLs from the excluded source.

        validation_mask = belongs_to_source(
            validation_df,
            source,
        )

        filtered_validation = validation_df.loc[
            validation_mask
        ].copy()

        # ----------------------------------------------------
        # VERIFY GROUP ISOLATION
        # ----------------------------------------------------

        train_groups = set(
            filtered_train["group_key"]
        )

        validation_groups = set(
            filtered_validation["group_key"]
        )

        if not train_groups.isdisjoint(validation_groups):
            raise ValueError(
                f"Group leakage detected: {source}"
            )

        # ----------------------------------------------------
        # PREPARE FEATURES
        # ----------------------------------------------------

        X_train = filtered_train[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float64)

        y_train = filtered_train[
            "label"
        ].to_numpy(dtype=int)

        X_validation = filtered_validation[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float64)

        y_validation = filtered_validation[
            "label"
        ].to_numpy(dtype=int)

        print(f"Training URLs: {len(X_train):,}")
        print(f"Validation URLs: {len(X_validation):,}")

        print(
            f"Training phishing: "
            f"{y_train.mean() * 100:.2f}%"
        )

        print(
            f"Validation phishing: "
            f"{y_validation.mean() * 100:.2f}%"
        )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model = create_models()["HistGradientBoosting"]

        print("\nTraining model...")

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # EVALUATE MODEL
        # ----------------------------------------------------

        predictions = model.predict(
            X_validation
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            predictions,
            probabilities,
        )

        print("\nRESULTS")

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"F1-score:  {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   {metrics['roc_auc']:.4f}"
        )

        print(
            f"False positives: "
            f"{metrics['false_positives']:,}"
        )

        print(
            f"False negatives: "
            f"{metrics['false_negatives']:,}"
        )

        results.append({
            "source": source,
            "train_urls": len(X_train),
            "validation_urls": len(X_validation),
            **metrics,
        })

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("FINAL CROSS-SOURCE COMPARISON")
    print("=" * 70)

    columns = [
        "source",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positives",
        "false_negatives",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )

    print("\nEvaluation completed.")
    print("Test dataset has not been used.")


if __name__ == "__main__":
    main()