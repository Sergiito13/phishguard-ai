"""
PhishGuard AI - Dataset Bias Analysis

Investigate distribution differences between URL datasets
and identify features associated with classification errors.

Analyses:
    1. Source and label distributions.
    2. Feature distribution shifts.
    3. False positive profiles.
    4. Permutation feature importance.

Models:
    - HistGradientBoosting baseline (20 features).
    - HistGradientBoosting without HTTPS (19 features).

Training:
    All available sources except URL_PHISH.

Validation:
    URL_PHISH only.

IMPORTANT:
    The test dataset is never accessed.
    No trained model artifacts are overwritten.
    No URLs are visited or sent over the network.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from compare_models import (
    SOURCES,
    create_models,
    calculate_metrics,
)

from experiment_training_strategies import (
    load_datasets,
    belongs_to_source,
)

from extract_features import FEATURE_COLUMNS


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

TARGET_SOURCE = "url_phish"

RANDOM_STATE = 42

THRESHOLD = 0.50

PERMUTATION_SAMPLE_SIZE = 3000

PERMUTATION_REPEATS = 3

EXPERIMENTS = {
    "baseline": list(FEATURE_COLUMNS),

    "without_https": [
        feature
        for feature in FEATURE_COLUMNS
        if feature != "is_https"
    ],
}


# ============================================================
# 2. SOURCE DISTRIBUTION ANALYSIS
# ============================================================

def analyze_source_distributions(
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze feature distributions for each source
    and classification label.

    Rows associated with multiple sources contribute
    to each corresponding source summary.
    """

    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    results = []

    for source in SOURCES:

        source_mask = belongs_to_source(
            validation_df,
            source,
        )

        source_df = validation_df.loc[
            source_mask
        ]

        for label in [0, 1]:

            subset = source_df.loc[
                source_df["label"] == label
            ]

            if subset.empty:
                continue

            results.append({
                "source": source,
                "label": label,
                "samples": len(subset),

                "https_rate": subset[
                    "is_https"
                ].mean(),

                "ip_rate": subset[
                    "has_ip"
                ].mean(),

                "url_length_mean": subset[
                    "url_length"
                ].mean(),

                "hostname_length_mean": subset[
                    "hostname_length"
                ].mean(),

                "subdomain_count_mean": subset[
                    "subdomain_count"
                ].mean(),

                "url_entropy_mean": subset[
                    "url_entropy"
                ].mean(),

                "digit_ratio_mean": subset[
                    "digit_ratio"
                ].mean(),
            })

    results_df = pd.DataFrame(results)

    print("\nSOURCE SUMMARY")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return results_df


# ============================================================
# 3. FEATURE DISTRIBUTION SHIFT
# ============================================================

def analyze_feature_shift(
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare URL_PHISH with all other validation sources.

    Differences are calculated separately for legitimate
    and phishing URLs.

    Uses standardized mean differences to identify
    distribution shifts between sources.

    These are descriptive measurements and do not
    establish causality.
    """

    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION SHIFT")
    print("=" * 70)

    target_mask = belongs_to_source(
        validation_df,
        TARGET_SOURCE,
    )

    target_df = validation_df.loc[
        target_mask
    ]

    other_df = validation_df.loc[
        ~target_mask
    ]

    results = []

    for label in [0, 1]:

        target_label_df = target_df.loc[
            target_df["label"] == label
        ]

        other_label_df = other_df.loc[
            other_df["label"] == label
        ]

        if target_label_df.empty or other_label_df.empty:
            raise ValueError(
                f"Missing comparison samples for label {label}"
            )

        for feature in FEATURE_COLUMNS:

            target_values = target_label_df[
                feature
            ].to_numpy(dtype=np.float64)

            other_values = other_label_df[
                feature
            ].to_numpy(dtype=np.float64)

            target_mean = target_values.mean()

            other_mean = other_values.mean()

            target_variance = target_values.var()

            other_variance = other_values.var()

            pooled_std = np.sqrt(
                (
                    target_variance
                    + other_variance
                ) / 2.0
            )

            difference = (
                target_mean - other_mean
            )

            standardized_difference = (
                difference / pooled_std
                if pooled_std > 0
                else 0.0
            )

            results.append({
                "label": label,
                "feature": feature,
                "target_mean": target_mean,
                "other_mean": other_mean,
                "mean_difference": difference,
                "standardized_difference":
                    standardized_difference,
                "absolute_shift":
                    abs(standardized_difference),
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "label",
            "absolute_shift",
        ],
        ascending=[
            True,
            False,
        ],
    )

    for label, name in [
        (0, "LEGITIMATE"),
        (1, "PHISHING"),
    ]:

        print(f"\nTOP FEATURE SHIFTS: {name}")

        subset = results_df.loc[
            results_df["label"] == label
        ].head(10)

        print(
            subset[
                [
                    "feature",
                    "target_mean",
                    "other_mean",
                    "standardized_difference",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    return results_df


# ============================================================
# 4. FALSE POSITIVE ANALYSIS
# ============================================================

def analyze_false_positives(
    validation_df: pd.DataFrame,
    predictions: np.ndarray,
    experiment_name: str,
) -> pd.DataFrame:
    """
    Compare feature averages between:

    False positives:
        Legitimate URLs incorrectly flagged as phishing.

    True negatives:
        Legitimate URLs correctly classified.
    """

    print("\n" + "=" * 70)
    print(f"FALSE POSITIVE ANALYSIS: {experiment_name}")
    print("=" * 70)

    dataframe = validation_df.copy()

    dataframe["prediction"] = predictions

    false_positives = dataframe.loc[
        (dataframe["label"] == 0)
        & (dataframe["prediction"] == 1)
    ]

    true_negatives = dataframe.loc[
        (dataframe["label"] == 0)
        & (dataframe["prediction"] == 0)
    ]

    print(
        f"\nFalse positives: {len(false_positives):,}"
    )

    print(
        f"True negatives:  {len(true_negatives):,}"
    )

    results = []

    for feature in FEATURE_COLUMNS:

        fp_mean = (
            false_positives[feature].mean()
            if not false_positives.empty
            else np.nan
        )

        tn_mean = (
            true_negatives[feature].mean()
            if not true_negatives.empty
            else np.nan
        )

        results.append({
            "experiment": experiment_name,
            "feature": feature,
            "false_positive_mean": fp_mean,
            "true_negative_mean": tn_mean,
            "mean_difference": fp_mean - tn_mean,
        })

    results_df = pd.DataFrame(results)

    results_df["absolute_difference"] = (
        results_df["mean_difference"].abs()
    )

    results_df = results_df.sort_values(
        by="absolute_difference",
        ascending=False,
    )

    print("\nTOP DIFFERENCES")

    print(
        results_df[
            [
                "feature",
                "false_positive_mean",
                "true_negative_mean",
                "mean_difference",
            ]
        ].head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    return results_df


# ============================================================
# 5. PERMUTATION FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(
    model,
    validation_df: pd.DataFrame,
    features: list[str],
    experiment_name: str,
) -> pd.DataFrame:
    """
    Measure how permuting one feature affects ROC-AUC.

    Positive importance:
        Shuffling the feature reduces ROC-AUC.

    Negative importance:
        Shuffling the feature improves ROC-AUC.

    Results are specific to the dataset, model,
    scoring metric and sampled observations.
    """

    print("\n" + "=" * 70)
    print(f"PERMUTATION IMPORTANCE: {experiment_name}")
    print("=" * 70)

    sample_size = min(
        PERMUTATION_SAMPLE_SIZE,
        len(validation_df),
    )

    sample_df = validation_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    if sample_df["label"].nunique() != 2:
        raise ValueError(
            "Permutation analysis requires both classes"
        )

    X_sample = sample_df[
        features
    ].to_numpy(dtype=np.float64)

    y_sample = sample_df[
        "label"
    ].to_numpy(dtype=int)

    importance = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    results_df = pd.DataFrame({
        "experiment": experiment_name,
        "feature": features,
        "importance_mean":
            importance.importances_mean,
        "importance_std":
            importance.importances_std,
    })

    results_df = results_df.sort_values(
        by="importance_mean",
        ascending=False,
    )

    print("\nFEATURE IMPORTANCE RANKING")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    return results_df


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

def evaluate_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Train temporary models with URL_PHISH excluded.

    Compare their performance and inspect
    classification errors.
    """

    print("\n" + "=" * 70)
    print("CROSS-SOURCE MODEL ANALYSIS")
    print("=" * 70)

    train_mask = ~belongs_to_source(
        train_df,
        TARGET_SOURCE,
    )

    validation_mask = belongs_to_source(
        validation_df,
        TARGET_SOURCE,
    )

    filtered_train = train_df.loc[
        train_mask
    ].copy()

    filtered_validation = validation_df.loc[
        validation_mask
    ].copy()

    if filtered_train.empty or filtered_validation.empty:
        raise ValueError(
            "Training or validation data is empty"
        )

    # Group isolation is already validated
    # by load_datasets().

    y_train = filtered_train[
        "label"
    ].to_numpy(dtype=int)

    y_validation = filtered_validation[
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

    model_results = []
    error_results = []
    importance_results = []

    for experiment_name, features in EXPERIMENTS.items():

        print("\n" + "#" * 70)

        print(f"EXPERIMENT: {experiment_name}")

        print("#" * 70)

        X_train = filtered_train[
            features
        ].to_numpy(dtype=np.float64)

        X_validation = filtered_validation[
            features
        ].to_numpy(dtype=np.float64)

        model = create_models()[
            "HistGradientBoosting"
        ]

        print("\nTraining model...")

        model.fit(
            X_train,
            y_train,
        )

        print("Training completed.")

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        metrics = calculate_metrics(
            y_validation,
            predictions,
            probabilities,
        )

        print("\nVALIDATION METRICS")

        for name, value in metrics.items():

            if name in [
                "false_positives",
                "false_negatives",
            ]:

                print(f"{name}: {value:,}")

            else:

                print(f"{name}: {value:.4f}")

        model_results.append({
            "experiment": experiment_name,
            "features": len(features),
            **metrics,
        })

        # Analyze classification errors.

        error_df = analyze_false_positives(
            filtered_validation,
            predictions,
            experiment_name,
        )

        error_results.append(error_df)

        # Analyze permutation importance.

        importance_df = analyze_feature_importance(
            model,
            filtered_validation,
            features,
            experiment_name,
        )

        importance_results.append(importance_df)

    return (
        pd.DataFrame(model_results),
        pd.concat(
            error_results,
            ignore_index=True,
        ),
        pd.concat(
            importance_results,
            ignore_index=True,
        ),
    )


# ============================================================
# 7. SAVE REPORTS
# ============================================================

def save_reports(
    source_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    models_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    importance_df: pd.DataFrame,
):
    """
    Save analysis results without including raw URLs.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "dataset_bias_sources.csv": source_df,
        "dataset_bias_feature_shift.csv": shift_df,
        "dataset_bias_models.csv": models_df,
        "dataset_bias_false_positives.csv": errors_df,
        "dataset_bias_feature_importance.csv":
            importance_df,
    }

    print("\n" + "=" * 70)
    print("GENERATED REPORTS")
    print("=" * 70)

    for filename, dataframe in reports.items():

        output_file = REPORT_DIR / filename

        dataframe.to_csv(
            output_file,
            index=False,
        )

        print(output_file)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHISHGUARD AI - DATASET BIAS ANALYSIS")
    print("=" * 70)

    print("\nLoading datasets...")

    train_df, validation_df = load_datasets()

    print(f"Training URLs: {len(train_df):,}")

    print(
        f"Validation URLs: {len(validation_df):,}"
    )

    print(f"Target source: {TARGET_SOURCE}")

    # Analyze dataset distributions.

    source_df = analyze_source_distributions(
        validation_df
    )

    # Analyze feature differences.

    shift_df = analyze_feature_shift(
        validation_df
    )

    # Train and inspect temporary models.

    (
        models_df,
        errors_df,
        importance_df,
    ) = evaluate_models(
        train_df,
        validation_df,
    )

    # Save CSV reports.

    save_reports(
        source_df,
        shift_df,
        models_df,
        errors_df,
        importance_df,
    )

    print("\n" + "=" * 70)
    print("DATASET BIAS ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nNo trained models were overwritten.")
    print("The test dataset was not accessed.")


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()