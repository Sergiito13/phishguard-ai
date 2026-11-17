"""
PhishGuard AI - HTTPS Feature Ablation Experiment

Compare HistGradientBoosting models with and without
the is_https feature.

Training:
    All sources except URL_PHISH.

Validation:
    URL_PHISH only.

The test dataset is never used.
No trained models are saved or overwritten.
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
# 1. CONFIGURATION
# ============================================================

TARGET_SOURCE = "url_phish"

RANDOM_STATE = 42

EXPERIMENTS = {
    "Baseline (20 features)": list(FEATURE_COLUMNS),

    "Without HTTPS (19 features)": [
        feature
        for feature in FEATURE_COLUMNS
        if feature != "is_https"
    ],
}


# ============================================================
# 2. SOURCE FILTERING
# ============================================================

def belongs_to_source(
    dataframe: pd.DataFrame,
    source: str,
) -> pd.Series:
    """
    Identify rows associated with a dataset source.
    """

    return (
        dataframe["sources"]
        .str.split("|", regex=False)
        .apply(lambda values: source in values)
    )


# ============================================================
# 3. LOAD DATASETS
# ============================================================

def load_data():
    """
    Load existing train and validation splits.

    Exclude URL_PHISH from training and evaluate
    exclusively on URL_PHISH validation samples.
    """

    columns = [
        "label",
        "sources",
        "group_key",
        *FEATURE_COLUMNS,
    ]

    train_df = pd.read_csv(
        TRAIN_FILE,
        usecols=columns,
        keep_default_na=False,
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE,
        usecols=columns,
        keep_default_na=False,
    )

    # Exclude target source from training.
    train_mask = ~belongs_to_source(
        train_df,
        TARGET_SOURCE,
    )

    # Select target source for validation.
    validation_mask = belongs_to_source(
        validation_df,
        TARGET_SOURCE,
    )

    train_df = train_df.loc[
        train_mask
    ].copy()

    validation_df = validation_df.loc[
        validation_mask
    ].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    # Prevent leakage between domain groups.
    train_groups = set(train_df["group_key"])
    validation_groups = set(validation_df["group_key"])

    if not train_groups.isdisjoint(validation_groups):
        raise ValueError(
            "Group leakage detected between datasets."
        )

    return train_df, validation_df


# ============================================================
# 4. TRAIN AND EVALUATE
# ============================================================

def run_experiment(
    name: str,
    features: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict:

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {name}")
    print("=" * 65)

    print(f"Features: {len(features)}")

    # Training data.
    X_train = train_df[
        features
    ].to_numpy(dtype=np.float64)

    y_train = train_df[
        "label"
    ].to_numpy(dtype=int)

    # Validation data.
    X_validation = validation_df[
        features
    ].to_numpy(dtype=np.float64)

    y_validation = validation_df[
        "label"
    ].to_numpy(dtype=int)

    # Create a fresh model for each experiment.
    model = create_models()[
        "HistGradientBoosting"
    ]

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # Predictions.
    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Calculate metrics.
    metrics = calculate_metrics(
        y_validation,
        predictions,
        probabilities,
    )

    print("\nVALIDATION RESULTS")
    print("-" * 40)

    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-score:          {metrics['f1']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")

    print(
        f"False positives:   "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False negatives:   "
        f"{metrics['false_negatives']:,}"
    )

    return {
        "experiment": name,
        "features": len(features),
        **metrics,
    }


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHISHGUARD AI - HTTPS ABLATION EXPERIMENT")
    print("=" * 65)

    print(f"\nTarget source: {TARGET_SOURCE}")

    # Load datasets.
    train_df, validation_df = load_data()

    print(f"Training URLs:   {len(train_df):,}")
    print(f"Validation URLs: {len(validation_df):,}")

    print("\nTRAINING LABEL DISTRIBUTION")

    print(
        train_df["label"].value_counts().sort_index()
    )

    print("\nVALIDATION LABEL DISTRIBUTION")

    print(
        validation_df["label"].value_counts().sort_index()
    )

    # Run experiments.
    results = []

    for name, features in EXPERIMENTS.items():

        result = run_experiment(
            name,
            features,
            train_df,
            validation_df,
        )

        results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    columns = [
        "experiment",
        "features",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positives",
        "false_negatives",
    ]

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

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

    print("\nExperiment completed successfully.")
    print("No model artifacts were modified.")
    print("Test dataset was not accessed.")


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()