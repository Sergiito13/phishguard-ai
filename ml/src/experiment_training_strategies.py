"""
PhishGuard AI - Training Strategies Experiment

Compare different HistGradientBoosting training strategies
to improve cross-dataset generalization.

Strategies:
    1. Baseline
    2. Class-balanced training
    3. Source-and-class-balanced training
    4. Source-and-class-balanced training without HTTPS

Evaluations:
    - Overall validation
    - Leave-one-source-out validation

IMPORTANT:
    The test dataset is never accessed.
    No trained model artifacts are overwritten.
    Results are saved as CSV reports.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

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

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

SOURCES = [
    "legitphish",
    "phiusiil",
    "phishbd",
    "url_phish",
]

RANDOM_STATE = 42

THRESHOLD = 0.50

EXPERIMENTS = {
    "baseline": {
        "features": list(FEATURE_COLUMNS),
        "weighting": "none",
    },

    "class_balanced": {
        "features": list(FEATURE_COLUMNS),
        "weighting": "class",
    },

    "source_class_balanced": {
        "features": list(FEATURE_COLUMNS),
        "weighting": "source_class",
    },

    "source_class_no_https": {
        "features": [
            feature
            for feature in FEATURE_COLUMNS
            if feature != "is_https"
        ],
        "weighting": "source_class",
    },
}


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_datasets():
    """
    Load the existing training and validation splits.

    The test dataset is intentionally excluded.
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

    if train_df.empty or validation_df.empty:
        raise ValueError("Dataset cannot be empty")

    for name, df in [
        ("training", train_df),
        ("validation", validation_df),
    ]:

        if not df["label"].isin([0, 1]).all():
            raise ValueError(
                f"Invalid labels in {name} dataset"
            )

        if df["group_key"].eq("").any():
            raise ValueError(
                f"Missing group keys in {name} dataset"
            )

        values = df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid features in {name} dataset"
            )

    verify_group_isolation(
        train_df,
        validation_df,
    )

    return train_df, validation_df


# ============================================================
# 3. SOURCE IDENTIFICATION
# ============================================================

def belongs_to_source(
    dataframe: pd.DataFrame,
    source: str,
) -> pd.Series:
    """
    Identify URLs associated with a specific source.

    A URL may belong to multiple datasets.
    """

    return dataframe["sources"].str.split(
        "|",
        regex=False,
    ).apply(
        lambda sources: source in sources
    )


# ============================================================
# 4. GROUP LEAKAGE PREVENTION
# ============================================================

def verify_group_isolation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Ensure that training and validation do not
    contain overlapping domain groups.
    """

    train_groups = set(
        train_df["group_key"]
    )

    validation_groups = set(
        validation_df["group_key"]
    )

    if not train_groups.isdisjoint(
        validation_groups
    ):
        raise ValueError(
            "Group leakage detected"
        )


# ============================================================
# 5. TRAINING SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(
    dataframe: pd.DataFrame,
    strategy: str,
):
    """
    Calculate training weights for each experiment.

    none:
        All URLs have equal weight.

    class:
        Balance legitimate and phishing samples.

    source_class:
        Balance combinations of dataset source
        and classification label.

    For URLs belonging to multiple sources,
    the first recorded source is used for
    weight assignment.

    The complete source membership remains
    available for cross-source filtering.
    """

    if strategy == "none":
        return None

    labels = dataframe["label"].to_numpy(
        dtype=int
    )

    if strategy == "class":

        return compute_sample_weight(
            class_weight="balanced",
            y=labels,
        )

    if strategy == "source_class":

        primary_sources = (
            dataframe["sources"]
            .str.split("|", regex=False)
            .apply(lambda values: values[0])
        )

        # Build source/label groups.
        group_keys = (
            primary_sources.astype(str)
            + "|"
            + dataframe["label"].astype(str)
        )

        group_counts = group_keys.value_counts()

        # Give each source/label group
        # approximately equal total weight.
        weights = (
            1.0
            / group_keys.map(group_counts).to_numpy(
                dtype=np.float64
            )
        )

        # Normalize so average weight is 1.
        weights *= len(weights) / weights.sum()

        return weights

    raise ValueError(
        f"Unknown weighting strategy: {strategy}"
    )


# ============================================================
# 6. CREATE MODEL
# ============================================================

def create_experiment_model():
    """
    Create a new HistGradientBoosting model.

    All experiments use identical hyperparameters
    so changes in performance can be attributed
    to the training strategy.
    """

    return create_models()[
        "HistGradientBoosting"
    ]


# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================

def run_experiment(
    experiment_name: str,
    config: dict,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    evaluation_source: str,
):
    """
    Train and evaluate one model configuration.
    """

    features = config["features"]

    weighting = config["weighting"]

    print("\n" + "=" * 70)

    print(f"EXPERIMENT: {experiment_name}")

    print(f"EVALUATION: {evaluation_source}")

    print("=" * 70)

    print(f"Features: {len(features)}")
    print(f"Weighting: {weighting}")

    print(f"Training URLs: {len(train_df):,}")

    print(
        f"Validation URLs: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    X_train = train_df[
        features
    ].to_numpy(dtype=np.float64)

    y_train = train_df[
        "label"
    ].to_numpy(dtype=int)

    X_validation = validation_df[
        features
    ].to_numpy(dtype=np.float64)

    y_validation = validation_df[
        "label"
    ].to_numpy(dtype=int)

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "Training requires both classes"
        )

    if len(np.unique(y_validation)) != 2:
        raise ValueError(
            "Validation requires both classes"
        )

    # --------------------------------------------------------
    # CALCULATE SAMPLE WEIGHTS
    # --------------------------------------------------------

    sample_weights = calculate_sample_weights(
        train_df,
        weighting,
    )

    # --------------------------------------------------------
    # CREATE AND TRAIN MODEL
    # --------------------------------------------------------

    model = create_experiment_model()

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        predictions,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    metrics["fpr"] = false_positive_rate

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nRESULTS")

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:         "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:            "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:          "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['fpr']:.4f}"
    )

    print(
        f"False positives:   "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False negatives:   "
        f"{metrics['false_negatives']:,}"
    )

    return {
        "experiment": experiment_name,
        "evaluation_source": evaluation_source,
        "training_samples": len(train_df),
        "validation_samples": len(validation_df),
        "feature_count": len(features),
        "weighting": weighting,
        **metrics,
    }


# ============================================================
# 8. OVERALL VALIDATION
# ============================================================

def evaluate_overall(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Evaluate models using all available sources.
    """

    print("\n" + "#" * 70)

    print("OVERALL VALIDATION")

    print("#" * 70)

    results = []

    for experiment_name, config in EXPERIMENTS.items():

        result = run_experiment(
            experiment_name,
            config,
            train_df,
            validation_df,
            "overall",
        )

        results.append(result)

    return results


# ============================================================
# 9. CROSS-SOURCE VALIDATION
# ============================================================

def evaluate_cross_source(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Leave-one-source-out evaluation.

    For each source:
        - Exclude that source from training.
        - Evaluate only on validation URLs
          belonging to the excluded source.
    """

    results = []

    for source in SOURCES:

        print("\n" + "#" * 70)

        print(f"EXCLUDED SOURCE: {source.upper()}")

        print("#" * 70)

        # Remove all URLs associated with
        # the excluded source.
        train_mask = ~belongs_to_source(
            train_df,
            source,
        )

        filtered_train = train_df.loc[
            train_mask
        ].copy()

        # Evaluate on excluded source only.
        validation_mask = belongs_to_source(
            validation_df,
            source,
        )

        filtered_validation = validation_df.loc[
            validation_mask
        ].copy()

        if filtered_train.empty:
            raise ValueError(
                f"Empty training set for {source}"
            )

        if filtered_validation.empty:
            raise ValueError(
                f"Empty validation set for {source}"
            )

        # Prevent domain group leakage.
        verify_group_isolation(
            filtered_train,
            filtered_validation,
        )

        # Train and evaluate all strategies.
        for experiment_name, config in EXPERIMENTS.items():

            result = run_experiment(
                experiment_name,
                config,
                filtered_train,
                filtered_validation,
                source,
            )

            results.append(result)

    return results


# ============================================================
# 10. GENERATE COMPARISON REPORTS
# ============================================================

def generate_reports(results: list[dict]):
    """
    Save detailed results and macro-averaged
    cross-source comparison.

    All sources receive equal weight when
    calculating macro metrics.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # DETAILED REPORT
    # --------------------------------------------------------

    detailed_file = (
        REPORT_DIR
        / "training_strategies_results.csv"
    )

    results_df.to_csv(
        detailed_file,
        index=False,
    )

    # --------------------------------------------------------
    # CROSS-SOURCE SUMMARY
    # --------------------------------------------------------

    cross_source_df = results_df.loc[
        results_df["evaluation_source"] != "overall"
    ]

    summary_df = (
        cross_source_df
        .groupby("experiment", as_index=False)
        .agg(
            macro_f1=("f1", "mean"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_roc_auc=("roc_auc", "mean"),
            macro_fpr=("fpr", "mean"),
            worst_fpr=("fpr", "max"),
        )
    )

    summary_df = summary_df.sort_values(
        by="macro_roc_auc",
        ascending=False,
    )

    summary_file = (
        REPORT_DIR
        / "training_strategies_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 90)

    print("FINAL CROSS-SOURCE COMPARISON")

    print("=" * 90)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "macro_f1": "{:.4f}".format,
                "macro_precision": "{:.4f}".format,
                "macro_recall": "{:.4f}".format,
                "macro_roc_auc": "{:.4f}".format,
                "macro_fpr": "{:.4f}".format,
                "worst_fpr": "{:.4f}".format,
            },
        )
    )

    print("\nREPORTS GENERATED")

    print(detailed_file)

    print(summary_file)

    print(
        "\nImportant: the best ROC-AUC does not "
        "necessarily identify the best model."
    )

    print(
        "Review false positive rates, recall "
        "and results for every source."
    )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("PHISHGUARD AI - TRAINING STRATEGIES")

    print("=" * 70)

    print("\nLoading datasets...")

    train_df, validation_df = load_datasets()

    print(
        f"Training URLs: {len(train_df):,}"
    )

    print(
        f"Validation URLs: "
        f"{len(validation_df):,}"
    )

    print(
        f"Experiments: {len(EXPERIMENTS)}"
    )

    print(
        f"Cross-source datasets: {len(SOURCES)}"
    )

    results = []

    # --------------------------------------------------------
    # OVERALL VALIDATION
    # --------------------------------------------------------

    results.extend(
        evaluate_overall(
            train_df,
            validation_df,
        )
    )

    # --------------------------------------------------------
    # CROSS-SOURCE VALIDATION
    # --------------------------------------------------------

    results.extend(
        evaluate_cross_source(
            train_df,
            validation_df,
        )
    )

    # --------------------------------------------------------
    # REPORT RESULTS
    # --------------------------------------------------------

    generate_reports(results)

    print("\n" + "=" * 70)

    print("TRAINING STRATEGIES COMPLETED")

    print("=" * 70)

    print("\nNo trained models were overwritten.")

    print("The test dataset was not accessed.")


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()