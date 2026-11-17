"""
PhishGuard AI - Hybrid Feature Engineering Experiment

Compare hybrid combinations of original V1 features
and protocol-independent V2 features.

Experiments:
    1. V1 baseline.
    2. Extended V2.
    3. V1 + additional V2 features.
    4. V1 without HTTPS + additional V2 features.
    5. Full hybrid V1 + V2.
    6. Full hybrid without explicit HTTPS.
    7. Extended V2 with HTTPS.

Evaluation:
    - Overall validation.
    - Leave-one-source-out validation.

IMPORTANT:
    Original dataset splits are preserved.
    The test dataset is never accessed.
    No trained model artifacts are overwritten.
    No network requests are performed.
"""

from pathlib import Path

import pandas as pd

from extract_features import FEATURE_COLUMNS

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    ADDITIONAL_FEATURES,
)

from experiment_features_v2 import (
    load_datasets,
    generate_cross_source_summary,
    calculate_baseline_deltas,
)

from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
    run_experiment,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

V1_FEATURES = list(FEATURE_COLUMNS)

V1_WITHOUT_HTTPS = [
    feature
    for feature in V1_FEATURES
    if feature != "is_https"
]

V2_FEATURES = list(FEATURE_COLUMNS_V2)

V2_ADDITIONAL_FEATURES = list(
    ADDITIONAL_FEATURES
)

ALL_FEATURES = (
    V1_FEATURES
    + V2_FEATURES
)


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENTS = {

    "baseline_v1": {
        "features": V1_FEATURES,
        "weighting": "none",
    },

    "extended_v2": {
        "features": V2_FEATURES,
        "weighting": "none",
    },

    "hybrid_v1_extended": {
        "features": (
            V1_FEATURES
            + V2_ADDITIONAL_FEATURES
        ),
        "weighting": "none",
    },

    "hybrid_v1_extended_no_https": {
        "features": (
            V1_WITHOUT_HTTPS
            + V2_ADDITIONAL_FEATURES
        ),
        "weighting": "none",
    },

    "hybrid_full": {
        "features": (
            V1_FEATURES
            + V2_FEATURES
        ),
        "weighting": "none",
    },

    "hybrid_full_no_https": {
        "features": (
            V1_WITHOUT_HTTPS
            + V2_FEATURES
        ),
        "weighting": "none",
    },

    "extended_v2_with_https": {
        "features": (
            V2_FEATURES
            + ["is_https"]
        ),
        "weighting": "none",
    },
}


EXPECTED_FEATURE_COUNTS = {
    "baseline_v1": 20,
    "extended_v2": 31,
    "hybrid_v1_extended": 32,
    "hybrid_v1_extended_no_https": 31,
    "hybrid_full": 51,
    "hybrid_full_no_https": 50,
    "extended_v2_with_https": 32,
}


# ============================================================
# 4. VALIDATE EXPERIMENT CONFIGURATIONS
# ============================================================

def validate_experiment_configurations():
    """
    Verify that every experiment contains valid,
    unique and expected feature columns.
    """

    if set(EXPERIMENTS) != set(
        EXPECTED_FEATURE_COUNTS
    ):

        raise ValueError(
            "Unexpected experiment configurations"
        )

    available_features = set(
        ALL_FEATURES
    )

    for name, config in EXPERIMENTS.items():

        features = config["features"]

        if not features:

            raise ValueError(
                f"Empty feature configuration: {name}"
            )

        if len(features) != len(set(features)):

            raise ValueError(
                f"Duplicate features in {name}"
            )

        unknown_features = (
            set(features)
            - available_features
        )

        if unknown_features:

            raise ValueError(
                f"Unknown features in {name}: "
                f"{unknown_features}"
            )

        expected_count = (
            EXPECTED_FEATURE_COUNTS[name]
        )

        if len(features) != expected_count:

            raise ValueError(
                f"Unexpected feature count in {name}: "
                f"{len(features)} != {expected_count}"
            )

        if config["weighting"] != "none":

            raise ValueError(
                f"Unexpected weighting in {name}"
            )

    print("Experiment configurations validated.")


# ============================================================
# 5. VALIDATE DATASET FEATURE COLUMNS
# ============================================================

def validate_dataset_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Ensure all required feature columns are
    available in both datasets.
    """

    required_columns = {
        "url",
        "label",
        "sources",
        "group_key",
        *ALL_FEATURES,
    }

    for name, dataframe in [
        ("training", train_df),
        ("validation", validation_df),
    ]:

        if dataframe.empty:

            raise ValueError(
                f"Empty {name} dataset"
            )

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:

            raise ValueError(
                f"Missing columns in {name} dataset: "
                f"{sorted(missing_columns)}"
            )

    verify_group_isolation(
        train_df,
        validation_df,
    )

    if not set(train_df["url"]).isdisjoint(
        validation_df["url"]
    ):

        raise ValueError(
            "URL leakage detected"
        )


# ============================================================
# 6. OVERALL VALIDATION
# ============================================================

def evaluate_overall(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict]:
    """
    Evaluate every hybrid feature configuration
    using all available training sources.
    """

    print("\n" + "#" * 70)

    print("OVERALL HYBRID FEATURE COMPARISON")

    print("#" * 70)

    results = []

    for experiment_name, config in EXPERIMENTS.items():

        result = run_experiment(
            experiment_name=experiment_name,
            config=config,
            train_df=train_df,
            validation_df=validation_df,
            evaluation_source="overall",
        )

        results.append(result)

    return results


# ============================================================
# 7. CROSS-SOURCE VALIDATION
# ============================================================

def evaluate_cross_source(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict]:
    """
    Evaluate every hybrid feature configuration
    using leave-one-source-out validation.

    URLs belonging to the target source are excluded
    from training, including shared-source URLs.
    """

    results = []

    for source in SOURCES:

        print("\n" + "#" * 70)

        print(
            f"EXCLUDED SOURCE: {source.upper()}"
        )

        print("#" * 70)

        # ----------------------------------------------------
        # FILTER TRAINING DATA
        # ----------------------------------------------------

        train_mask = ~belongs_to_source(
            train_df,
            source,
        )

        filtered_train = train_df.loc[
            train_mask
        ].copy()

        # ----------------------------------------------------
        # FILTER VALIDATION DATA
        # ----------------------------------------------------

        validation_mask = belongs_to_source(
            validation_df,
            source,
        )

        filtered_validation = validation_df.loc[
            validation_mask
        ].copy()

        # ----------------------------------------------------
        # VALIDATE FILTERED DATASETS
        # ----------------------------------------------------

        if filtered_train.empty:

            raise ValueError(
                f"Empty training dataset for {source}"
            )

        if filtered_validation.empty:

            raise ValueError(
                f"Empty validation dataset for {source}"
            )

        if filtered_train["label"].nunique() != 2:

            raise ValueError(
                f"Training requires both classes: {source}"
            )

        if filtered_validation["label"].nunique() != 2:

            raise ValueError(
                f"Validation requires both classes: {source}"
            )

        verify_group_isolation(
            filtered_train,
            filtered_validation,
        )

        if belongs_to_source(
            filtered_train,
            source,
        ).any():

            raise ValueError(
                f"Source leakage detected: {source}"
            )

        # ----------------------------------------------------
        # RUN HYBRID EXPERIMENTS
        # ----------------------------------------------------

        for experiment_name, config in EXPERIMENTS.items():

            result = run_experiment(
                experiment_name=experiment_name,
                config=config,
                train_df=filtered_train,
                validation_df=filtered_validation,
                evaluation_source=source,
            )

            results.append(result)

    return results


# ============================================================
# 8. VALIDATE EXPERIMENT RESULTS
# ============================================================

def validate_results(
    results_df: pd.DataFrame,
):
    """
    Verify that every experiment has exactly one
    result for each evaluation source.
    """

    required_columns = {
        "experiment",
        "evaluation_source",
        "feature_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "fpr",
        "false_positives",
        "false_negatives",
    }

    missing_columns = (
        required_columns
        - set(results_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing result columns: {missing_columns}"
        )

    evaluation_sources = [
        "overall",
        *SOURCES,
    ]

    expected_pairs = {
        (experiment_name, source)

        for experiment_name in EXPERIMENTS

        for source in evaluation_sources
    }

    actual_pairs = set(
        zip(
            results_df["experiment"],
            results_df["evaluation_source"],
        )
    )

    if actual_pairs != expected_pairs:

        raise ValueError(
            "Incomplete or unexpected experiment results"
        )

    if results_df.duplicated(
        subset=[
            "experiment",
            "evaluation_source",
        ]
    ).any():

        raise ValueError(
            "Duplicate experiment results"
        )

    for name, expected_count in (
        EXPECTED_FEATURE_COUNTS.items()
    ):

        experiment_rows = results_df.loc[
            results_df["experiment"] == name
        ]

        if not experiment_rows[
            "feature_count"
        ].eq(expected_count).all():

            raise ValueError(
                f"Invalid feature count in results: {name}"
            )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "fpr",
    ]

    for column in metric_columns:

        if not results_df[column].between(
            0.0,
            1.0,
        ).all():

            raise ValueError(
                f"Invalid metric values: {column}"
            )


# ============================================================
# 9. COMPARE AGAINST EXTENDED V2
# ============================================================

def calculate_v2_deltas(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare every hybrid experiment against
    the previous Extended V2 model.
    """

    reference_df = summary_df.loc[
        summary_df["experiment"] == "extended_v2"
    ]

    if len(reference_df) != 1:

        raise ValueError(
            "Expected exactly one Extended V2 result"
        )

    reference = reference_df.iloc[0]

    comparison_df = summary_df.copy()

    metrics = [
        "macro_f1",
        "macro_roc_auc",
        "macro_recall",
        "macro_fpr",
        "worst_fpr",
        "worst_recall",
    ]

    for metric in metrics:

        comparison_df[f"delta_v2_{metric}"] = (
            comparison_df[metric]
            - reference[metric]
        )

    return comparison_df


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

def print_comparison(
    title: str,
    dataframe: pd.DataFrame,
    columns: list[str],
):
    """
    Display selected experiment metrics.
    """

    print("\n" + "=" * 110)

    print(title)

    print("=" * 110)

    print(
        dataframe[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# ============================================================
# 11. GENERATE REPORTS
# ============================================================

def generate_reports(
    results: list[dict],
):
    """
    Save detailed experiment results and comparisons
    against both V1 and Extended V2.
    """

    results_df = pd.DataFrame(results)

    validate_results(results_df)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # OVERALL VALIDATION RESULTS
    # --------------------------------------------------------

    overall_df = results_df.loc[
        results_df["evaluation_source"] == "overall"
    ].copy()

    overall_df = overall_df.sort_values(
        by="f1",
        ascending=False,
    )

    # --------------------------------------------------------
    # CROSS-SOURCE SUMMARY
    # --------------------------------------------------------

    summary_df = generate_cross_source_summary(
        results_df
    )

    # --------------------------------------------------------
    # BASELINE V1 COMPARISON
    # --------------------------------------------------------

    baseline_comparison_df = (
        calculate_baseline_deltas(
            summary_df
        )
    )

    # --------------------------------------------------------
    # EXTENDED V2 COMPARISON
    # --------------------------------------------------------

    v2_comparison_df = calculate_v2_deltas(
        summary_df
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    reports = {

        "hybrid_features_results.csv":
            results_df,

        "hybrid_features_overall.csv":
            overall_df,

        "hybrid_features_summary.csv":
            summary_df,

        "hybrid_features_baseline_comparison.csv":
            baseline_comparison_df,

        "hybrid_features_v2_comparison.csv":
            v2_comparison_df,
    }

    for filename, dataframe in reports.items():

        output_file = REPORT_DIR / filename

        dataframe.to_csv(
            output_file,
            index=False,
        )

    # --------------------------------------------------------
    # DISPLAY OVERALL VALIDATION
    # --------------------------------------------------------

    metric_columns = [
        "experiment",
        "feature_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "fpr",
        "false_positives",
        "false_negatives",
    ]

    print_comparison(
        "OVERALL VALIDATION COMPARISON",
        overall_df,
        metric_columns,
    )

    # --------------------------------------------------------
    # DISPLAY CROSS-SOURCE SUMMARY
    # --------------------------------------------------------

    summary_columns = [
        "experiment",
        "feature_count",
        "macro_f1",
        "macro_recall",
        "macro_roc_auc",
        "macro_fpr",
        "worst_fpr",
        "worst_recall",
    ]

    print_comparison(
        "FINAL CROSS-SOURCE HYBRID COMPARISON",
        summary_df,
        summary_columns,
    )

    # --------------------------------------------------------
    # DISPLAY INDIVIDUAL SOURCE RESULTS
    # --------------------------------------------------------

    for source in SOURCES:

        source_df = results_df.loc[
            results_df["evaluation_source"] == source
        ].copy()

        source_df = source_df.sort_values(
            by="f1",
            ascending=False,
        )

        print_comparison(
            f"{source.upper()} COMPARISON",
            source_df,
            metric_columns,
        )

    # --------------------------------------------------------
    # DISPLAY BASELINE V1 DELTAS
    # --------------------------------------------------------

    baseline_delta_columns = [
        "experiment",
        "delta_macro_f1",
        "delta_macro_roc_auc",
        "delta_macro_recall",
        "delta_macro_fpr",
        "delta_worst_fpr",
    ]

    print_comparison(
        "IMPROVEMENTS AGAINST V1 BASELINE",
        baseline_comparison_df,
        baseline_delta_columns,
    )

    # --------------------------------------------------------
    # DISPLAY EXTENDED V2 DELTAS
    # --------------------------------------------------------

    v2_delta_columns = [
        "experiment",
        "delta_v2_macro_f1",
        "delta_v2_macro_roc_auc",
        "delta_v2_macro_recall",
        "delta_v2_macro_fpr",
        "delta_v2_worst_fpr",
        "delta_v2_worst_recall",
    ]

    print_comparison(
        "IMPROVEMENTS AGAINST EXTENDED V2",
        v2_comparison_df,
        v2_delta_columns,
    )

    # --------------------------------------------------------
    # DISPLAY GENERATED REPORTS
    # --------------------------------------------------------

    print("\nGENERATED REPORTS")

    for filename in reports:

        print(REPORT_DIR / filename)

    print(
        "\nModel selection must consider recall, "
        "false positive rates and cross-source performance."
    )

    print(
        "The highest ROC-AUC does not necessarily "
        "identify the best deployment candidate."
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("PHISHGUARD AI - HYBRID FEATURE ENGINEERING")

    print("=" * 70)

    # --------------------------------------------------------
    # VALIDATE CONFIGURATIONS
    # --------------------------------------------------------

    validate_experiment_configurations()

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    print("\nLoading and validating datasets...")

    train_df, validation_df = load_datasets()

    validate_dataset_features(
        train_df,
        validation_df,
    )

    print(
        f"\nTraining URLs: {len(train_df):,}"
    )

    print(
        f"Validation URLs: {len(validation_df):,}"
    )

    print(
        f"Available features: {len(ALL_FEATURES)}"
    )

    print(
        f"Experiment configurations: {len(EXPERIMENTS)}"
    )

    print(
        f"Cross-source datasets: {len(SOURCES)}"
    )

    total_experiments = (
        len(EXPERIMENTS)
        * (len(SOURCES) + 1)
    )

    print(
        f"Total model trainings: {total_experiments}"
    )

    # --------------------------------------------------------
    # DISPLAY FEATURE CONFIGURATIONS
    # --------------------------------------------------------

    print("\nFEATURE CONFIGURATIONS")

    for name, config in EXPERIMENTS.items():

        features = config["features"]

        print(
            f"{name}: "
            f"{len(features)} features"
        )

    # --------------------------------------------------------
    # OVERALL EVALUATION
    # --------------------------------------------------------

    results = []

    results.extend(
        evaluate_overall(
            train_df,
            validation_df,
        )
    )

    # --------------------------------------------------------
    # CROSS-SOURCE EVALUATION
    # --------------------------------------------------------

    results.extend(
        evaluate_cross_source(
            train_df,
            validation_df,
        )
    )

    if len(results) != total_experiments:

        raise ValueError(
            "Unexpected number of experiment results"
        )

    # --------------------------------------------------------
    # GENERATE FINAL REPORTS
    # --------------------------------------------------------

    generate_reports(results)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("HYBRID FEATURE ENGINEERING COMPLETED")

    print("=" * 70)

    print(
        f"\nCompleted trainings: {len(results)}"
    )

    print(
        "Original dataset splits were preserved."
    )

    print(
        "No trained models were overwritten."
    )

    print(
        "The test dataset was not accessed."
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()