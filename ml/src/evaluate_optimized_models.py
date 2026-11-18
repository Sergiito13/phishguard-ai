"""
PhishGuard AI - Optimized Models Evaluation

Compare original and optimized HistGradientBoosting models.

Feature sets:
    - baseline_v1
    - hybrid_v1_extended
    - extended_v2

Evaluations:
    - Overall validation
    - Leave-one-source-out validation

IMPORTANT:
    Hyperparameters were selected using training data only.
    The existing validation split is used for diagnostic comparison.
    The test split is never accessed.
    No model artifacts are saved or overwritten.
"""

import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from compare_models import calculate_metrics

from experiment_features_v2 import (
    load_datasets,
    generate_cross_source_summary,
)

from experiment_hyperparameters import (
    FEATURE_SETS,
    BASELINE_PARAMS,
    create_model,
)

from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

BEST_PARAMS_FILE = (
    REPORT_DIR / "hyperparameters_best.csv"
)

THRESHOLD = 0.50


# ============================================================
# 2. LOAD OPTIMIZED PARAMETERS
# ============================================================

def load_best_parameters() -> dict:
    """
    Load the best hyperparameters for each feature set.
    """

    if not BEST_PARAMS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing hyperparameter report: {BEST_PARAMS_FILE}"
        )

    dataframe = pd.read_csv(BEST_PARAMS_FILE)

    required_columns = {
        "feature_set",
        "parameters",
        "feature_count",
    }

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "Invalid hyperparameter report columns"
        )

    if dataframe["feature_set"].duplicated().any():
        raise ValueError(
            "Duplicate feature sets in hyperparameter report"
        )

    if set(dataframe["feature_set"]) != set(FEATURE_SETS):
        raise ValueError(
            "Missing or unexpected optimized feature sets"
        )

    parameters = {}

    for _, row in dataframe.iterrows():

        feature_set = row["feature_set"]

        expected_count = len(
            FEATURE_SETS[feature_set]
        )

        if int(row["feature_count"]) != expected_count:
            raise ValueError(
                f"Feature count mismatch: {feature_set}"
            )

        try:
            params = json.loads(
                row["parameters"]
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid parameters: {feature_set}"
            ) from exc

        if not isinstance(params, dict):
            raise ValueError(
                f"Invalid parameters: {feature_set}"
            )

        if set(params) != set(BASELINE_PARAMS):
            raise ValueError(
                f"Unexpected parameters: {feature_set}"
            )

        # Check parameter compatibility.

        create_model(params)

        parameters[feature_set] = params

    return parameters


# ============================================================
# 3. CREATE EXPERIMENT CONFIGURATIONS
# ============================================================

def create_experiments(
    optimized_parameters: dict,
) -> dict:
    """
    Create original and optimized configurations
    for every feature set.
    """

    if set(optimized_parameters) != set(FEATURE_SETS):
        raise ValueError(
            "Missing or unexpected optimized feature sets"
        )

    experiments = {}

    for feature_set, features in FEATURE_SETS.items():

        experiments[f"{feature_set}_original"] = {
            "feature_set": feature_set,
            "variant": "original",
            "features": list(features),
            "parameters": BASELINE_PARAMS.copy(),
        }

        experiments[f"{feature_set}_optimized"] = {
            "feature_set": feature_set,
            "variant": "optimized",
            "features": list(features),
            "parameters": optimized_parameters[feature_set].copy(),
        }

    return experiments


# ============================================================
# 4. CREATE EVALUATION FOLDS
# ============================================================

def create_evaluation_folds(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict]:
    """
    Create overall and source-exclusion evaluations.

    For each source:
    - Exclude that source from training.
    - Exclude its domain groups from training.
    - Evaluate on its validation samples.
    """

    if train_df.empty or validation_df.empty:
        raise ValueError(
            "Training and validation datasets must not be empty"
        )

    if (
        train_df["label"].nunique() != 2
        or validation_df["label"].nunique() != 2
    ):
        raise ValueError(
            "Training and validation require both classes"
        )

    verify_group_isolation(
        train_df,
        validation_df,
    )

    folds = []

    # --------------------------------------------------------
    # OVERALL VALIDATION
    # --------------------------------------------------------

    folds.append({
        "source": "overall",
        "train_indices": np.arange(
            len(train_df)
        ),
        "validation_indices": np.arange(
            len(validation_df)
        ),
    })

    # --------------------------------------------------------
    # CROSS-SOURCE VALIDATION
    # --------------------------------------------------------

    for source in SOURCES:

        train_source_mask = belongs_to_source(
            train_df,
            source,
        )

        validation_source_mask = belongs_to_source(
            validation_df,
            source,
        )

        if not validation_source_mask.any():
            raise ValueError(
                f"No validation samples for {source}"
            )

        # Prevent shared-domain contamination.

        held_out_groups = set(
            train_df.loc[
                train_source_mask,
                "group_key",
            ]
        )

        held_out_groups.update(
            validation_df.loc[
                validation_source_mask,
                "group_key",
            ]
        )

        train_mask = (
            ~train_source_mask
            & ~train_df["group_key"].isin(
                held_out_groups
            )
        )

        filtered_train = train_df.loc[
            train_mask
        ]

        filtered_validation = validation_df.loc[
            validation_source_mask
        ]

        if filtered_train.empty:
            raise ValueError(
                f"Empty training fold: {source}"
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

        folds.append({
            "source": source,
            "train_indices": np.flatnonzero(
                train_mask.to_numpy()
            ),
            "validation_indices": np.flatnonzero(
                validation_source_mask.to_numpy()
            ),
        })

    return folds


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    features: list[str],
    parameters: dict,
) -> dict:
    """
    Train and evaluate one classifier.
    """

    X_train = train_df[features].to_numpy(
        dtype=np.float64
    )

    y_train = train_df["label"].to_numpy(
        dtype=int
    )

    X_validation = validation_df[features].to_numpy(
        dtype=np.float64
    )

    y_validation = validation_df["label"].to_numpy(
        dtype=int
    )

    model = create_model(parameters)

    model.fit(
        X_train,
        y_train,
    )

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

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics["fpr"] = (
        float(fp / (fp + tn))
        if (fp + tn) > 0
        else 0.0
    )

    return metrics


# ============================================================
# 6. RUN EXPERIMENTS
# ============================================================

def run_experiments(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    experiments: dict,
) -> pd.DataFrame:
    """
    Evaluate every configuration on every fold.
    """

    folds = create_evaluation_folds(
        train_df,
        validation_df,
    )

    results = []

    total = len(experiments) * len(folds)

    completed = 0

    print(f"\nTotal trainings: {total}")

    for experiment_name, config in experiments.items():

        print("\n" + "=" * 70)

        print(f"EXPERIMENT: {experiment_name}")

        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            training = train_df.iloc[
                fold["train_indices"]
            ]

            validation = validation_df.iloc[
                fold["validation_indices"]
            ]

            metrics = evaluate_model(
                train_df=training,
                validation_df=validation,
                features=config["features"],
                parameters=config["parameters"],
            )

            completed += 1

            results.append({
                "experiment": experiment_name,
                "feature_set": config["feature_set"],
                "variant": config["variant"],
                "evaluation_source": source,
                "feature_count": len(
                    config["features"]
                ),
                "training_samples": len(training),
                "validation_samples": len(validation),
                "parameters": json.dumps(
                    config["parameters"],
                    sort_keys=True,
                ),
                **metrics,
            })

            print(
                f"[{completed}/{total}] "
                f"{source:<12} "
                f"F1={metrics['f1']:.4f} "
                f"ROC-AUC={metrics['roc_auc']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"FPR={metrics['fpr']:.4f}"
            )

    return pd.DataFrame(results)


# ============================================================
# 7. COMPARE OPTIMIZED VS ORIGINAL
# ============================================================

def generate_comparison(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate macro-metric changes against the
    original configuration of each feature set.
    """

    comparison = summary_df.copy()

    comparison["variant"] = (
        comparison["experiment"]
        .str.rsplit("_", n=1)
        .str[-1]
    )

    comparison["feature_set"] = (
        comparison["experiment"]
        .str.rsplit("_", n=1)
        .str[0]
    )

    metrics = [
        "macro_f1",
        "macro_roc_auc",
        "macro_recall",
        "macro_fpr",
        "worst_fpr",
        "worst_recall",
    ]

    originals = comparison.loc[
        comparison["variant"] == "original",
        ["feature_set", *metrics],
    ].copy()

    originals = originals.rename(
        columns={
            metric: f"original_{metric}"
            for metric in metrics
        }
    )

    if originals["feature_set"].duplicated().any():
        raise ValueError(
            "Duplicate original reference metrics"
        )

    comparison = comparison.merge(
        originals,
        on="feature_set",
        how="left",
        validate="many_to_one",
    )

    if comparison[
        "original_macro_roc_auc"
    ].isna().any():
        raise ValueError(
            "Missing original reference metrics"
        )

    for metric in metrics:

        comparison[f"delta_{metric}"] = (
            comparison[metric]
            - comparison[f"original_{metric}"]
        )

    return comparison


# ============================================================
# 8. SAVE REPORTS
# ============================================================

def save_reports(
    results_df: pd.DataFrame,
):
    """
    Save overall, cross-source and paired comparisons.
    """

    if results_df.empty:
        raise ValueError(
            "Experiment results cannot be empty"
        )

    if results_df.duplicated(
        ["experiment", "evaluation_source"]
    ).any():
        raise ValueError(
            "Duplicate experiment evaluations"
        )

    expected_sources = {
        "overall",
        *SOURCES,
    }

    for _, subset in results_df.groupby("experiment"):

        if set(subset["evaluation_source"]) != expected_sources:
            raise ValueError(
                "Incomplete experiment evaluations"
            )

    overall_df = results_df.loc[
        results_df["evaluation_source"] == "overall"
    ].copy()

    overall_df = overall_df.sort_values(
        by="f1",
        ascending=False,
    )

    summary_df = generate_cross_source_summary(
        results_df
    )

    comparison_df = generate_comparison(
        summary_df
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "optimized_models_results.csv":
            results_df,

        "optimized_models_overall.csv":
            overall_df,

        "optimized_models_summary.csv":
            summary_df,

        "optimized_models_comparison.csv":
            comparison_df,
    }

    for filename, dataframe in reports.items():

        output_file = REPORT_DIR / filename

        dataframe.to_csv(
            output_file,
            index=False,
        )

        print(f"Saved: {output_file}")

    return overall_df, summary_df, comparison_df


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

def display_results(
    overall_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    results_df: pd.DataFrame,
):
    """
    Display diagnostic comparison tables.

    Results are passed explicitly.
    No global variables are required.
    """

    print("\n" + "=" * 100)

    print("OVERALL VALIDATION COMPARISON")

    print("=" * 100)

    overall_columns = [
        "experiment",
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
        overall_df[overall_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 100)

    print("CROSS-SOURCE COMPARISON")

    print("=" * 100)

    summary_columns = [
        "experiment",
        "macro_f1",
        "macro_roc_auc",
        "macro_recall",
        "macro_fpr",
        "worst_fpr",
        "worst_recall",
    ]

    print(
        summary_df[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 100)

    print("OPTIMIZED VS ORIGINAL DELTAS")

    print("=" * 100)

    optimized_comparison = comparison_df.loc[
        comparison_df["variant"] == "optimized"
    ]

    delta_columns = [
        "feature_set",
        "delta_macro_f1",
        "delta_macro_roc_auc",
        "delta_macro_recall",
        "delta_macro_fpr",
        "delta_worst_fpr",
        "delta_worst_recall",
    ]

    print(
        optimized_comparison[delta_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:+.4f}",
        )
    )

    print("\n" + "=" * 100)

    print("URL_PHISH COMPARISON")

    print("=" * 100)

    url_phish_df = results_df.loc[
        results_df["evaluation_source"] == "url_phish"
    ].copy()

    print(
        url_phish_df[overall_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("PHISHGUARD AI - OPTIMIZED MODELS EVALUATION")

    print("=" * 70)

    print("\nLoading optimized parameters...")

    best_parameters = load_best_parameters()

    experiments = create_experiments(
        best_parameters
    )

    print("\nLoading datasets...")

    train_df, validation_df = load_datasets()

    print(f"Training URLs: {len(train_df):,}")

    print(f"Validation URLs: {len(validation_df):,}")

    print(f"Configurations: {len(experiments)}")

    print("\nRunning evaluations...")

    results_df = run_experiments(
        train_df,
        validation_df,
        experiments,
    )

    expected_results = (
        len(experiments)
        * (len(SOURCES) + 1)
    )

    if len(results_df) != expected_results:
        raise ValueError(
            "Unexpected number of evaluations"
        )

    overall_df, summary_df, comparison_df = (
        save_reports(results_df)
    )

    display_results(
        overall_df,
        summary_df,
        comparison_df,
        results_df,
    )

    print("\n" + "=" * 70)

    print("OPTIMIZED MODELS EVALUATION COMPLETED")

    print("=" * 70)

    print("No model artifacts were overwritten.")

    print("The test dataset was not accessed.")


if __name__ == "__main__":
    main()