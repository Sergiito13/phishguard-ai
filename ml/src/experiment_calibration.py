"""
PhishGuard AI - Probability Calibration Experiment

Compare probability calibration methods for:

    - hybrid_v1_extended
    - extended_v2

Calibration methods:

    - none
    - sigmoid
    - isotonic

Evaluation:
    - Leave-one-source-out validation.
    - Domain-group isolation.
    - Brier Score.
    - Log Loss.
    - Expected Calibration Error (ECE).
    - ROC-AUC.
    - Precision, Recall and F1.
    - False Positive Rate.

IMPORTANT:
    Only training data is used.
    Original validation and test datasets are not accessed.
    No trained models are overwritten.
    No network requests are performed.
"""

import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import GroupShuffleSplit

from evaluate_optimized_models import load_best_parameters

from experiment_hyperparameters import (
    FEATURE_SETS,
    create_model,
    create_source_folds,
    load_training_data,
)

from experiment_training_strategies import (
    SOURCES,
    verify_group_isolation,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42

CALIBRATION_FRACTION = 0.20

THRESHOLD = 0.50


# ============================================================
# 2. MODEL CANDIDATES
# ============================================================

CANDIDATES = (
    "hybrid_v1_extended",
    "extended_v2",
)

METHODS = (
    "none",
    "sigmoid",
    "isotonic",
)


# ============================================================
# 3. SPLIT TRAINING AND CALIBRATION
# ============================================================

def split_training_calibration(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into model-fitting and calibration subsets.

    Domain groups must never overlap.

    Both subsets must contain legitimate and phishing URLs.
    """

    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    X = np.zeros(
        (len(dataframe), 1)
    )

    y = dataframe["label"].to_numpy(
        dtype=int
    )

    groups = dataframe["group_key"].to_numpy()

    for fit_idx, calibration_idx in splitter.split(
        X,
        y,
        groups=groups,
    ):

        fit_df = dataframe.iloc[
            fit_idx
        ]

        calibration_df = dataframe.iloc[
            calibration_idx
        ]

        if fit_df["label"].nunique() != 2:
            continue

        if calibration_df["label"].nunique() != 2:
            continue

        verify_group_isolation(
            fit_df,
            calibration_df,
        )

        return fit_df, calibration_df

    raise ValueError(
        "Could not produce a two-class, "
        "group-isolated calibration split"
    )


# ============================================================
# 4. LOGIT TRANSFORMATION
# ============================================================

def logit(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Convert probabilities to log-odds.

    Used for sigmoid calibration.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    result = np.log(
        probabilities / (1.0 - probabilities)
    )

    return result.reshape(-1, 1)


# ============================================================
# 5. CALIBRATION METHODS
# ============================================================

def calibrate(
    method: str,
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Fit the calibration transformation on calibration data.

    Apply the transformation to held-out evaluation data.

    Evaluation labels are never used to fit calibration.
    """

    # --------------------------------------------------------
    # NO CALIBRATION
    # --------------------------------------------------------

    if method == "none":

        return np.asarray(
            evaluation_probabilities,
            dtype=float,
        )

    # --------------------------------------------------------
    # SIGMOID CALIBRATION
    # --------------------------------------------------------

    if method == "sigmoid":

        model = LogisticRegression(
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

        model.fit(
            logit(calibration_probabilities),
            calibration_labels,
        )

        calibrated_probabilities = (
            model.predict_proba(
                logit(evaluation_probabilities)
            )[:, 1]
        )

        return calibrated_probabilities

    # --------------------------------------------------------
    # ISOTONIC CALIBRATION
    # --------------------------------------------------------

    if method == "isotonic":

        model = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds="clip",
        )

        model.fit(
            calibration_probabilities,
            calibration_labels,
        )

        calibrated_probabilities = model.predict(
            evaluation_probabilities
        )

        return np.asarray(
            calibrated_probabilities,
            dtype=float,
        )

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


# ============================================================
# 6. EXPECTED CALIBRATION ERROR
# ============================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Lower values indicate better calibration.

    Equal-width probability bins are used.
    """

    if bins <= 0:
        raise ValueError(
            "bins must be positive"
        )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_true) == 0:
        raise ValueError(
            "Invalid calibration error inputs"
        )

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Invalid calibration error inputs"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Invalid probabilities"
        )

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):
        raise ValueError(
            "Invalid probabilities"
        )

    indices = np.minimum(
        (probabilities * bins).astype(int),
        bins - 1,
    )

    result = 0.0

    for index in range(bins):

        mask = indices == index

        if mask.any():

            observed_rate = (
                y_true[mask].mean()
            )

            predicted_probability = (
                probabilities[mask].mean()
            )

            bin_error = abs(
                observed_rate
                - predicted_probability
            )

            result += float(
                mask.mean() * bin_error
            )

    return result


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def metrics_for(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Calculate classification and calibration metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if set(np.unique(y_true)) != {0, 1}:

        raise ValueError(
            "Evaluation requires both classes"
        )

    if len(y_true) != len(probabilities):

        raise ValueError(
            "Invalid evaluation probabilities"
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Invalid evaluation probabilities"
        )

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):

        raise ValueError(
            "Probabilities must lie in [0, 1]"
        )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {

        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),

        "ece": expected_calibration_error(
            y_true,
            probabilities,
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "fpr": float(
            false_positive_rate
        ),

        "false_positives": int(fp),

        "false_negatives": int(fn),
    }


# ============================================================
# 8. RUN CALIBRATION EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    best_parameters: dict,
) -> pd.DataFrame:
    """
    Evaluate calibration methods on held-out sources.

    Models:
        - Hybrid V1 Extended
        - Extended V2

    Sources:
        - LegitPhish
        - PhiUSIIL
        - PhishBD
        - URL_PHISH

    Calibration:
        - none
        - sigmoid
        - isotonic

    Total:
        8 model fits.
        24 calibration evaluations.
    """

    folds = create_source_folds(
        dataframe
    )

    missing_candidates = (
        set(CANDIDATES)
        - set(best_parameters)
    )

    if missing_candidates:

        raise ValueError(
            "Missing optimized parameters"
        )

    results = []

    for feature_set in CANDIDATES:

        features = FEATURE_SETS[
            feature_set
        ]

        params = best_parameters[
            feature_set
        ]

        print("\n" + "=" * 70)

        print(
            f"MODEL: {feature_set}"
        )

        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            train_df = dataframe.iloc[
                fold["training_indices"]
            ]

            held_out_df = dataframe.iloc[
                fold["validation_indices"]
            ]

            # -----------------------------------------------
            # CREATE CALIBRATION SPLIT
            # -----------------------------------------------

            fit_df, calibration_df = (
                split_training_calibration(
                    train_df
                )
            )

            verify_group_isolation(
                fit_df,
                held_out_df,
            )

            verify_group_isolation(
                calibration_df,
                held_out_df,
            )

            # -----------------------------------------------
            # TRAIN CLASSIFIER
            # -----------------------------------------------

            model = create_model(
                params
            )

            model.fit(
                fit_df[features].to_numpy(
                    dtype=np.float64
                ),
                fit_df["label"].to_numpy(
                    dtype=int
                ),
            )

            # -----------------------------------------------
            # PREDICT CALIBRATION DATA
            # -----------------------------------------------

            calibration_probabilities = (
                model.predict_proba(
                    calibration_df[features].to_numpy(
                        dtype=np.float64
                    )
                )[:, 1]
            )

            # -----------------------------------------------
            # PREDICT HELD-OUT SOURCE
            # -----------------------------------------------

            evaluation_probabilities = (
                model.predict_proba(
                    held_out_df[features].to_numpy(
                        dtype=np.float64
                    )
                )[:, 1]
            )

            calibration_labels = (
                calibration_df["label"].to_numpy(
                    dtype=int
                )
            )

            evaluation_labels = (
                held_out_df["label"].to_numpy(
                    dtype=int
                )
            )

            # -----------------------------------------------
            # CALIBRATION METHODS
            # -----------------------------------------------

            for method in METHODS:

                probabilities = calibrate(
                    method,
                    calibration_probabilities,
                    calibration_labels,
                    evaluation_probabilities,
                )

                metrics = metrics_for(
                    evaluation_labels,
                    probabilities,
                )

                results.append({

                    "feature_set": feature_set,

                    "method": method,

                    "evaluation_source": source,

                    "feature_count": len(features),

                    "fit_samples": len(fit_df),

                    "calibration_samples": len(
                        calibration_df
                    ),

                    "evaluation_samples": len(
                        held_out_df
                    ),

                    "parameters": json.dumps(
                        params,
                        sort_keys=True,
                    ),

                    **metrics,
                })

                print(
                    f"{feature_set:<19} "
                    f"{source:<11} "
                    f"{method:<8} "
                    f"Brier={metrics['brier']:.4f} "
                    f"ECE={metrics['ece']:.4f} "
                    f"Recall={metrics['recall']:.4f} "
                    f"FPR={metrics['fpr']:.4f}"
                )

    results_df = pd.DataFrame(
        results
    )

    expected_rows = (
        len(CANDIDATES)
        * len(folds)
        * len(METHODS)
    )

    if len(results_df) != expected_rows:

        raise ValueError(
            "Incomplete calibration evaluations"
        )

    duplicate_results = results_df.duplicated(
        [
            "feature_set",
            "evaluation_source",
            "method",
        ]
    ).any()

    if duplicate_results:

        raise ValueError(
            "Duplicate calibration evaluations"
        )

    return results_df


# ============================================================
# 9. GENERATE REPORTS
# ============================================================

def make_reports(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate cross-source calibration summaries
    and differences against raw probabilities.
    """

    if results.empty:

        raise ValueError(
            "Empty calibration results"
        )

    expected = {
        (feature, method, source)
        for feature in CANDIDATES
        for method in METHODS
        for source in SOURCES
    }

    actual = set(
        zip(
            results["feature_set"],
            results["method"],
            results["evaluation_source"],
        )
    )

    if (
        len(results) != len(expected)
        or actual != expected
    ):

        raise ValueError(
            "Incomplete or unexpected calibration evaluations"
        )

    summary = (
        results
        .groupby(
            [
                "feature_set",
                "method",
            ],
            as_index=False,
        )
        .agg(

            evaluated_sources=(
                "evaluation_source",
                "nunique",
            ),

            macro_brier=(
                "brier",
                "mean",
            ),

            macro_log_loss=(
                "log_loss",
                "mean",
            ),

            macro_ece=(
                "ece",
                "mean",
            ),

            macro_roc_auc=(
                "roc_auc",
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

    # --------------------------------------------------------
    # RAW REFERENCE
    # --------------------------------------------------------

    baseline = summary.loc[
        summary["method"] == "none",
        [
            "feature_set",
            "macro_brier",
            "macro_ece",
            "macro_fpr",
            "macro_recall",
        ],
    ].rename(
        columns={

            "macro_brier": "raw_brier",

            "macro_ece": "raw_ece",

            "macro_fpr": "raw_fpr",

            "macro_recall": "raw_recall",
        }
    )

    comparison = summary.merge(
        baseline,
        on="feature_set",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # CALCULATE DELTAS
    # --------------------------------------------------------

    comparisons = (

        ("macro_brier", "raw_brier"),

        ("macro_ece", "raw_ece"),

        ("macro_fpr", "raw_fpr"),

        ("macro_recall", "raw_recall"),
    )

    for metric, reference in comparisons:

        comparison[f"delta_{metric}"] = (
            comparison[metric]
            - comparison[reference]
        )

    summary = summary.sort_values(
        by=[
            "feature_set",
            "macro_brier",
        ]
    )

    return summary, comparison


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:

    print("=" * 70)

    print(
        "PHISHGUARD AI - PROBABILITY CALIBRATION"
    )

    print("=" * 70)

    print(
        "Training split only; "
        "calibration uses disjoint domain groups."
    )

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print("\nLoading training data...")

    dataframe = load_training_data()

    print(
        f"Training URLs: {len(dataframe):,}"
    )

    # --------------------------------------------------------
    # LOAD OPTIMIZED PARAMETERS
    # --------------------------------------------------------

    print("\nLoading optimized parameters...")

    best_parameters = load_best_parameters()

    # --------------------------------------------------------
    # RUN EXPERIMENT
    # --------------------------------------------------------

    print("\nRunning calibration experiments...")

    results = run_experiment(
        dataframe,
        best_parameters,
    )

    # --------------------------------------------------------
    # GENERATE REPORTS
    # --------------------------------------------------------

    summary, comparison = make_reports(
        results
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {

        "calibration_source_results.csv":
            results,

        "calibration_summary.csv":
            summary,

        "calibration_comparison.csv":
            comparison,
    }

    for filename, frame in reports.items():

        output_file = (
            REPORT_DIR / filename
        )

        frame.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Saved: {output_file}"
        )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "CALIBRATION SUMMARY"
    )

    print("=" * 100)

    columns = [
        "feature_set",
        "method",
        "macro_brier",
        "macro_ece",
        "macro_roc_auc",
        "macro_recall",
        "macro_fpr",
        "worst_recall",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 70)

    print(
        "PROBABILITY CALIBRATION COMPLETED"
    )

    print("=" * 70)

    print(
        "Completed model fits: 8"
    )

    print(
        "Completed evaluations: 24"
    )

    print(
        "Only training data was accessed."
    )

    print(
        "No trained models were overwritten."
    )


if __name__ == "__main__":
    main()