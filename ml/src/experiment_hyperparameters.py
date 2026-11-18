"""
PhishGuard AI - Hyperparameter Optimization Experiment

Optimize HistGradientBoostingClassifier hyperparameters
for three feature configurations:

    - baseline_v1
    - hybrid_v1_extended
    - extended_v2

Methodology:
    - Randomized parameter search.
    - Training-only leave-one-source-out validation.
    - Domain group isolation.
    - Macro-averaged source metrics.

IMPORTANT:
    Only the original training split is used.
    Validation and test splits are not accessed.
    No trained model artifacts are overwritten.
    No network requests are performed.
"""

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler

from compare_models import calculate_metrics

from experiment_features_v2 import (
    TRAIN_FILE,
    V2_TRAIN_FILE,
    load_dataset,
)

from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)

from extract_features import FEATURE_COLUMNS

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    ADDITIONAL_FEATURES,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42

THRESHOLD = 0.50

DEFAULT_TRIALS = 6


# ============================================================
# 2. FEATURE CONFIGURATIONS
# ============================================================

FEATURE_SETS = {
    "baseline_v1": list(FEATURE_COLUMNS),

    "hybrid_v1_extended": (
        list(FEATURE_COLUMNS)
        + list(ADDITIONAL_FEATURES)
    ),

    "extended_v2": list(FEATURE_COLUMNS_V2),
}


# ============================================================
# 3. ORIGINAL HYPERPARAMETERS
# ============================================================

BASELINE_PARAMS = {
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "learning_rate": 0.08,
    "l2_regularization": 5.0,
    "min_samples_leaf": 20,
    "max_depth": None,
}


# ============================================================
# 4. SEARCH SPACE
# ============================================================

PARAMETER_SPACE = {
    "max_iter": [
        100,
        150,
        250,
    ],

    "max_leaf_nodes": [
        7,
        15,
        31,
    ],

    "learning_rate": [
        0.04,
        0.08,
        0.12,
    ],

    "l2_regularization": [
        1.0,
        5.0,
        10.0,
    ],

    "min_samples_leaf": [
        20,
        60,
        120,
    ],

    "max_depth": [
        3,
        5,
        None,
    ],
}


# ============================================================
# 5. PARAMETER GENERATION
# ============================================================

def generate_parameters(
    trials: int,
) -> list[dict]:
    """
    Generate reproducible hyperparameter candidates.

    The original configuration is always included
    as trial zero.
    """

    if trials < 1:
        raise ValueError(
            "Trials must be greater than zero"
        )

    candidates = [
        BASELINE_PARAMS.copy()
    ]

    if trials == 1:
        return candidates

    random_candidates = list(
        ParameterSampler(
            param_distributions=PARAMETER_SPACE,
            n_iter=trials - 1,
            random_state=RANDOM_STATE,
        )
    )

    for params in random_candidates:

        if params not in candidates:
            candidates.append(params)

    return candidates


# ============================================================
# 6. CREATE MODEL
# ============================================================

def create_model(
    params: dict,
):
    """
    Create a fresh HistGradientBoosting model.

    No existing model file is loaded or modified.
    """

    return HistGradientBoostingClassifier(
        **params,
        random_state=RANDOM_STATE,
    )


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

def load_training_data() -> pd.DataFrame:
    """
    Load only the original training split.

    The V2 dataset is checked against the original V1
    data to ensure feature and metadata alignment.
    """

    dataframe = load_dataset(
        TRAIN_FILE,
        V2_TRAIN_FILE,
    )

    required_features = set()

    for features in FEATURE_SETS.values():
        required_features.update(features)

    missing_features = (
        required_features
        - set(dataframe.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    if dataframe["group_key"].eq("").any():
        raise ValueError(
            "Missing group keys"
        )

    if dataframe["label"].nunique() != 2:
        raise ValueError(
            "Training requires both classes"
        )

    return dataframe


# ============================================================
# 8. BUILD SOURCE-BASED FOLDS
# ============================================================

def create_source_folds(
    dataframe: pd.DataFrame,
) -> list[dict]:
    """
    Generate one validation fold for each dataset source.

    The target source is excluded from training.

    Domains present in the target source are also
    excluded from training, even if their URLs
    belong to other sources.
    """

    folds = []

    for source in SOURCES:

        source_mask = belongs_to_source(
            dataframe,
            source,
        )

        validation_df = dataframe.loc[
            source_mask
        ]

        if validation_df.empty:
            raise ValueError(
                f"Empty validation fold: {source}"
            )

        held_out_groups = set(
            validation_df["group_key"]
        )

        training_mask = (
            ~source_mask
            & ~dataframe["group_key"].isin(
                held_out_groups
            )
        )

        training_df = dataframe.loc[
            training_mask
        ]

        if training_df.empty:
            raise ValueError(
                f"Empty training fold: {source}"
            )

        if training_df["label"].nunique() != 2:
            raise ValueError(
                f"Training requires both classes: {source}"
            )

        if validation_df["label"].nunique() != 2:
            raise ValueError(
                f"Validation requires both classes: {source}"
            )

        verify_group_isolation(
            training_df,
            validation_df,
        )

        if belongs_to_source(
            training_df,
            source,
        ).any():

            raise ValueError(
                f"Source leakage detected: {source}"
            )

        folds.append({
            "source": source,

            "training_indices": np.flatnonzero(
                training_mask.to_numpy()
            ),

            "validation_indices": np.flatnonzero(
                source_mask.to_numpy()
            ),

            "training_samples": len(training_df),

            "validation_samples": len(validation_df),
        })

    return folds


# ============================================================
# 9. EVALUATE A SINGLE FOLD
# ============================================================

def evaluate_fold(
    X: np.ndarray,
    y: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    params: dict,
) -> dict:
    """
    Train a model and evaluate it on one held-out source.
    """

    X_train = X[training_indices]

    y_train = y[training_indices]

    X_validation = X[validation_indices]

    y_validation = y[validation_indices]

    model = create_model(params)

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

    negative_count = np.sum(
        y_validation == 0
    )

    false_positives = np.sum(
        (y_validation == 0)
        & (predictions == 1)
    )

    fpr = (
        false_positives / negative_count
        if negative_count > 0
        else 0.0
    )

    metrics["fpr"] = float(fpr)

    return metrics


# ============================================================
# 10. RUN HYPERPARAMETER SEARCH
# ============================================================

def run_search(
    dataframe: pd.DataFrame,
    trials: int = DEFAULT_TRIALS,
) -> pd.DataFrame:
    """
    Evaluate all parameter candidates against
    all held-out sources.

    Every feature set uses the same parameter
    candidates and evaluation folds.
    """

    folds = create_source_folds(
        dataframe
    )

    candidates = generate_parameters(
        trials
    )

    y = dataframe["label"].to_numpy(
        dtype=int
    )

    results = []

    total_trainings = (
        len(FEATURE_SETS)
        * len(candidates)
        * len(folds)
    )

    completed = 0

    print("\n" + "=" * 70)

    print("HYPERPARAMETER SEARCH")

    print("=" * 70)

    print(f"Feature sets: {len(FEATURE_SETS)}")

    print(f"Parameter candidates: {len(candidates)}")

    print(f"Source folds: {len(folds)}")

    print(f"Total trainings: {total_trainings}")

    # --------------------------------------------------------
    # FEATURE SETS
    # --------------------------------------------------------

    for feature_set_name, features in (
        FEATURE_SETS.items()
    ):

        print("\n" + "#" * 70)

        print(
            f"FEATURE SET: {feature_set_name}"
        )

        print(
            f"Features: {len(features)}"
        )

        print("#" * 70)

        X = dataframe[features].to_numpy(
            dtype=np.float64
        )

        # ----------------------------------------------------
        # PARAMETER CANDIDATES
        # ----------------------------------------------------

        for trial_id, params in enumerate(candidates):

            print("\n" + "-" * 70)

            print(
                f"TRIAL {trial_id + 1}/{len(candidates)}"
            )

            print(
                json.dumps(
                    params,
                    sort_keys=True,
                )
            )

            print("-" * 70)

            # ------------------------------------------------
            # SOURCE FOLDS
            # ------------------------------------------------

            for fold in folds:

                source = fold["source"]

                metrics = evaluate_fold(
                    X=X,
                    y=y,
                    training_indices=fold[
                        "training_indices"
                    ],
                    validation_indices=fold[
                        "validation_indices"
                    ],
                    params=params,
                )

                completed += 1

                result = {
                    "feature_set": feature_set_name,

                    "trial_id": trial_id,

                    "evaluation_source": source,

                    "feature_count": len(features),

                    "training_samples": fold[
                        "training_samples"
                    ],

                    "validation_samples": fold[
                        "validation_samples"
                    ],

                    "parameters": json.dumps(
                        params,
                        sort_keys=True,
                    ),

                    **metrics,
                }

                results.append(result)

                print(
                    f"[{completed}/{total_trainings}] "
                    f"{source:<12} "
                    f"ROC-AUC={metrics['roc_auc']:.4f} "
                    f"Recall={metrics['recall']:.4f} "
                    f"FPR={metrics['fpr']:.4f}"
                )

    return pd.DataFrame(results)


# ============================================================
# 11. GENERATE SEARCH SUMMARY
# ============================================================

def generate_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate macro-averaged metrics across source folds.

    Every source has equal weight.

    Parameter ranking uses macro ROC-AUC first.
    Other metrics remain available for candidate selection.
    """

    if results_df.empty:
        raise ValueError(
            "Experiment results cannot be empty"
        )

    summary = (
        results_df
        .groupby(
            [
                "feature_set",
                "trial_id",
            ],
            as_index=False,
        )
        .agg(
            evaluated_sources=(
                "evaluation_source",
                "nunique",
            ),

            feature_count=(
                "feature_count",
                "first",
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

            parameters=(
                "parameters",
                "first",
            ),
        )
    )

    if not summary[
        "evaluated_sources"
    ].eq(len(SOURCES)).all():

        raise ValueError(
            "Incomplete cross-source evaluation"
        )

    summary = summary.sort_values(
        by=[
            "feature_set",
            "macro_roc_auc",
            "macro_average_precision",
            "worst_recall",
            "macro_fpr",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary["rank"] = (
        summary
        .groupby("feature_set")
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # COMPARE AGAINST ORIGINAL PARAMETERS
    # --------------------------------------------------------

    original_results = summary.loc[
        summary["trial_id"] == 0,
        [
            "feature_set",
            "macro_roc_auc",
            "macro_fpr",
            "worst_recall",
        ],
    ].rename(
        columns={
            "macro_roc_auc": "original_macro_roc_auc",
            "macro_fpr": "original_macro_fpr",
            "worst_recall": "original_worst_recall",
        }
    )

    summary = summary.merge(
        original_results,
        on="feature_set",
        validate="many_to_one",
    )

    summary["delta_macro_roc_auc"] = (
        summary["macro_roc_auc"]
        - summary["original_macro_roc_auc"]
    )

    summary["delta_macro_fpr"] = (
        summary["macro_fpr"]
        - summary["original_macro_fpr"]
    )

    summary["delta_worst_recall"] = (
        summary["worst_recall"]
        - summary["original_worst_recall"]
    )

    return summary


# ============================================================
# 12. SAVE REPORTS
# ============================================================

def save_reports(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):
    """
    Save detailed source metrics, parameter rankings
    and the best configuration for each feature set.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_df = summary_df.loc[
        summary_df["rank"] == 1
    ].copy()

    reports = {
        "hyperparameters_source_results.csv":
            results_df,

        "hyperparameters_trials.csv":
            summary_df,

        "hyperparameters_best.csv":
            best_df,
    }

    for filename, dataframe in reports.items():

        output_file = REPORT_DIR / filename

        dataframe.to_csv(
            output_file,
            index=False,
        )

    return best_df


# ============================================================
# 13. DISPLAY BEST CONFIGURATIONS
# ============================================================

def display_results(
    best_df: pd.DataFrame,
):
    """
    Display the best trial per feature set.
    """

    print("\n" + "=" * 100)

    print("BEST HYPERPARAMETER CONFIGURATIONS")

    print("=" * 100)

    columns = [
        "feature_set",
        "trial_id",
        "feature_count",
        "macro_f1",
        "macro_recall",
        "macro_roc_auc",
        "macro_fpr",
        "worst_fpr",
        "worst_recall",
        "delta_macro_roc_auc",
    ]

    print(
        best_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSELECTED PARAMETERS")

    for _, row in best_df.iterrows():

        print(
            f"\n{row['feature_set']}:"
        )

        params = json.loads(
            row["parameters"]
        )

        for name, value in params.items():

            print(
                f"  {name}: {value}"
            )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PhishGuard AI hyperparameter optimization"
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Parameter candidates per feature set",
    )

    args = parser.parse_args()

    print("=" * 70)

    print("PHISHGUARD AI - HYPERPARAMETER OPTIMIZATION")

    print("=" * 70)

    print("\nLoading training data...")

    dataframe = load_training_data()

    print(
        f"Training URLs: {len(dataframe):,}"
    )

    print(
        f"Feature sets: {len(FEATURE_SETS)}"
    )

    print(
        f"Parameter candidates: {args.trials}"
    )

    # --------------------------------------------------------
    # RUN SEARCH
    # --------------------------------------------------------

    results_df = run_search(
        dataframe=dataframe,
        trials=args.trials,
    )

    # --------------------------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------------------------

    summary_df = generate_summary(
        results_df
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    best_df = save_reports(
        results_df,
        summary_df,
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    display_results(
        best_df
    )

    print("\n" + "=" * 70)

    print("HYPERPARAMETER OPTIMIZATION COMPLETED")

    print("=" * 70)

    print(
        f"Completed trainings: {len(results_df)}"
    )

    print(
        "Only the training split was used."
    )

    print(
        "No trained models were overwritten."
    )

    print(
        "Validation and test datasets were not accessed."
    )


if __name__ == "__main__":
    main()