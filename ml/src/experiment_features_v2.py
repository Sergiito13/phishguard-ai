"""
PhishGuard AI - Feature Engineering V2 Experiment

Compare original V1 features against protocol-independent
and extended V2 features.

Experiments:
    1. V1 baseline (20 features).
    2. V1 without HTTPS (19 features).
    3. V2 protocol independent (19 features).
    4. V2 extended (31 features).

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

import numpy as np
import pandas as pd

from extract_features import FEATURE_COLUMNS

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    PROTOCOL_INDEPENDENT_FEATURES,
    METADATA_COLUMNS,
)

from train_model import (
    TRAIN_FILE,
    VALIDATION_FILE,
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

V2_DATA_DIR = BASE_DIR / "data" / "splits" / "v2"

V2_TRAIN_FILE = V2_DATA_DIR / "train.csv"

V2_VALIDATION_FILE = V2_DATA_DIR / "validation.csv"

REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENTS = {
    "baseline_v1": {
        "features": list(FEATURE_COLUMNS),
        "weighting": "none",
    },

    "without_https_v1": {
        "features": [
            feature
            for feature in FEATURE_COLUMNS
            if feature != "is_https"
        ],
        "weighting": "none",
    },

    "protocol_independent_v2": {
        "features": list(
            PROTOCOL_INDEPENDENT_FEATURES
        ),
        "weighting": "none",
    },

    "extended_v2": {
        "features": list(FEATURE_COLUMNS_V2),
        "weighting": "none",
    },
}


ALL_FEATURES = (
    list(FEATURE_COLUMNS)
    + list(FEATURE_COLUMNS_V2)
)


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset(
    original_file: Path,
    v2_file: Path,
) -> pd.DataFrame:
    """
    Load a V2 dataset and verify its alignment
    with the corresponding original V1 dataset.
    """

    if not original_file.is_file():
        raise FileNotFoundError(
            f"Original dataset not found: {original_file}"
        )

    if not v2_file.is_file():
        raise FileNotFoundError(
            f"V2 dataset not found: {v2_file}"
        )

    metadata_types = {
        "url": "string",
        "sources": "string",
        "group_key": "string",
        "domain_status": "string",
    }

    original_df = pd.read_csv(
        original_file,
        usecols=[
            *METADATA_COLUMNS,
            *FEATURE_COLUMNS,
        ],
        keep_default_na=False,
        dtype=metadata_types,
    )

    v2_df = pd.read_csv(
        v2_file,
        usecols=[
            *METADATA_COLUMNS,
            *ALL_FEATURES,
        ],
        keep_default_na=False,
        dtype=metadata_types,
    )

    if original_df.empty or v2_df.empty:
        raise ValueError(
            "Dataset cannot be empty"
        )

    # --------------------------------------------------------
    # CHECK ROW COUNT
    # --------------------------------------------------------

    if len(original_df) != len(v2_df):
        raise ValueError(
            "V1 and V2 datasets have different row counts"
        )

    # --------------------------------------------------------
    # CHECK METADATA ALIGNMENT
    # --------------------------------------------------------

    try:

        pd.testing.assert_frame_equal(
            original_df[METADATA_COLUMNS],
            v2_df[METADATA_COLUMNS],
            check_exact=True,
        )

    except AssertionError as exc:

        raise ValueError(
            "V1 and V2 dataset metadata do not match"
        ) from exc

    # --------------------------------------------------------
    # CHECK ORIGINAL V1 FEATURES
    # --------------------------------------------------------

    original_values = original_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    copied_values = v2_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    if not np.allclose(
        original_values,
        copied_values,
        rtol=1e-12,
        atol=1e-12,
    ):

        raise ValueError(
            "Original V1 features were modified"
        )

    # Use the original V1 values to ensure that
    # baseline experiments remain reproducible.

    v2_df[FEATURE_COLUMNS] = (
        original_df[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # VALIDATE LABELS AND METADATA
    # --------------------------------------------------------

    if not v2_df["label"].isin([0, 1]).all():
        raise ValueError(
            "Invalid labels found"
        )

    if v2_df["label"].nunique() != 2:
        raise ValueError(
            "Dataset requires both classes"
        )

    if v2_df["group_key"].eq("").any():
        raise ValueError(
            "Missing group keys"
        )

    if v2_df["sources"].eq("").any():
        raise ValueError(
            "Missing dataset sources"
        )

    if v2_df["url"].duplicated().any():
        raise ValueError(
            "Duplicate URLs found"
        )

    # --------------------------------------------------------
    # VALIDATE NUMERICAL FEATURES
    # --------------------------------------------------------

    feature_values = v2_df[
        ALL_FEATURES
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Dataset contains invalid numerical features"
        )

    print(
        f"Validated {v2_file.name}: "
        f"{len(v2_df):,} URLs"
    )

    return v2_df


# ============================================================
# 4. LOAD TRAINING AND VALIDATION DATASETS
# ============================================================

def load_datasets():
    """
    Load both V2 datasets without accessing the test set.
    """

    train_df = load_dataset(
        TRAIN_FILE,
        V2_TRAIN_FILE,
    )

    validation_df = load_dataset(
        VALIDATION_FILE,
        V2_VALIDATION_FILE,
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

    return train_df, validation_df


# ============================================================
# 5. OVERALL VALIDATION
# ============================================================

def evaluate_overall(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict]:
    """
    Compare all feature sets using every available source.
    """

    print("\n" + "#" * 70)

    print("OVERALL FEATURE COMPARISON")

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
# 6. CROSS-SOURCE VALIDATION
# ============================================================

def evaluate_cross_source(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict]:
    """
    Compare feature sets using leave-one-source-out
    validation.

    Each target source is completely excluded
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
        # COMPARE FEATURE SETS
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
# 7. GENERATE CROSS-SOURCE SUMMARY
# ============================================================

def generate_cross_source_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate macro-averaged cross-source metrics.

    Each evaluation source receives equal weight.
    Overall validation is excluded from these averages.
    """

    cross_source_df = results_df.loc[
        results_df["evaluation_source"] != "overall"
    ]

    summary_df = (
        cross_source_df
        .groupby(
            [
                "experiment",
                "feature_count",
            ],
            as_index=False,
        )
        .agg(
            evaluated_sources=(
                "evaluation_source",
                "nunique",
            ),

            macro_accuracy=(
                "accuracy",
                "mean",
            ),

            macro_precision=(
                "precision",
                "mean",
            ),

            macro_recall=(
                "recall",
                "mean",
            ),

            macro_f1=(
                "f1",
                "mean",
            ),

            macro_roc_auc=(
                "roc_auc",
                "mean",
            ),

            macro_average_precision=(
                "average_precision",
                "mean",
            ),

            macro_fpr=(
                "fpr",
                "mean",
            ),

            worst_fpr=(
                "fpr",
                "max",
            ),

            worst_recall=(
                "recall",
                "min",
            ),
        )
    )

    if not summary_df[
        "evaluated_sources"
    ].eq(len(SOURCES)).all():

        raise ValueError(
            "Incomplete cross-source evaluation"
        )

    return summary_df.sort_values(
        by="macro_roc_auc",
        ascending=False,
    )


# ============================================================
# 8. CALCULATE IMPROVEMENTS AGAINST BASELINE
# ============================================================

def calculate_baseline_deltas(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate changes in macro metrics relative to V1.

    Positive F1, ROC-AUC and recall deltas are favorable.
    Negative FPR deltas are favorable.
    """

    baseline_df = summary_df.loc[
        summary_df["experiment"] == "baseline_v1"
    ]

    if len(baseline_df) != 1:
        raise ValueError(
            "Expected exactly one baseline result"
        )

    baseline = baseline_df.iloc[0]

    comparison_df = summary_df.copy()

    metrics = [
        "macro_f1",
        "macro_roc_auc",
        "macro_recall",
        "macro_fpr",
        "worst_fpr",
    ]

    for metric in metrics:

        comparison_df[f"delta_{metric}"] = (
            comparison_df[metric]
            - baseline[metric]
        )

    return comparison_df


# ============================================================
# 9. SAVE AND DISPLAY REPORTS
# ============================================================

def generate_reports(results: list[dict]):
    """
    Save detailed metrics, cross-source summary
    and improvements against the original baseline.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # OVERALL RESULTS
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

    comparison_df = calculate_baseline_deltas(
        summary_df
    )

    # --------------------------------------------------------
    # URL_PHISH RESULTS
    # --------------------------------------------------------

    url_phish_df = results_df.loc[
        results_df["evaluation_source"] == "url_phish"
    ].copy()

    # --------------------------------------------------------
    # SAVE CSV REPORTS
    # --------------------------------------------------------

    reports = {
        "features_v2_results.csv":
            results_df,

        "features_v2_overall.csv":
            overall_df,

        "features_v2_summary.csv":
            summary_df,

        "features_v2_baseline_comparison.csv":
            comparison_df,
    }

    for filename, dataframe in reports.items():

        output_file = REPORT_DIR / filename

        dataframe.to_csv(
            output_file,
            index=False,
        )

    # --------------------------------------------------------
    # DISPLAY OVERALL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("OVERALL VALIDATION COMPARISON")

    print("=" * 100)

    columns = [
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

    print(
        overall_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # DISPLAY CROSS-SOURCE COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("FINAL CROSS-SOURCE FEATURE COMPARISON")

    print("=" * 100)

    summary_columns = [
        "experiment",
        "feature_count",
        "macro_f1",
        "macro_recall",
        "macro_roc_auc",
        "macro_fpr",
        "worst_fpr",
    ]

    print(
        summary_df[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # DISPLAY TARGET SOURCE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("URL_PHISH COMPARISON")

    print("=" * 100)

    print(
        url_phish_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # DISPLAY BASELINE IMPROVEMENTS
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print("IMPROVEMENTS AGAINST V1 BASELINE")

    print("=" * 100)

    delta_columns = [
        "experiment",
        "delta_macro_f1",
        "delta_macro_roc_auc",
        "delta_macro_recall",
        "delta_macro_fpr",
        "delta_worst_fpr",
    ]

    print(
        comparison_df[delta_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:+.4f}",
        )
    )

    # --------------------------------------------------------
    # DISPLAY GENERATED FILES
    # --------------------------------------------------------

    print("\nGENERATED REPORTS")

    for filename in reports:

        print(REPORT_DIR / filename)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("PHISHGUARD AI - FEATURE ENGINEERING V2 EXPERIMENT")

    print("=" * 70)

    print("\nLoading and validating datasets...")

    train_df, validation_df = load_datasets()

    print(
        f"\nTraining URLs: {len(train_df):,}"
    )

    print(
        f"Validation URLs: {len(validation_df):,}"
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

        print(
            f"{name}: "
            f"{len(config['features'])} features"
        )

    results = []

    # --------------------------------------------------------
    # RUN OVERALL EVALUATION
    # --------------------------------------------------------

    results.extend(
        evaluate_overall(
            train_df,
            validation_df,
        )
    )

    # --------------------------------------------------------
    # RUN CROSS-SOURCE EVALUATION
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

    print("\n" + "=" * 70)

    print("FEATURE ENGINEERING V2 EXPERIMENT COMPLETED")

    print("=" * 70)

    print(
        f"\nCompleted trainings: {len(results)}"
    )

    print("Original dataset splits were preserved.")

    print("No trained models were overwritten.")

    print("The test dataset was not accessed.")


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()