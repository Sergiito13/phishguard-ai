"""
PhishGuard AI - Exploratory cross-source threshold sweep.

Uses ONLY the existing training split. Evaluates the optimized Hybrid V1
Extended and Extended V2 models on leave-one-source-out folds with domain
isolation. Thresholds are explored, not approved for deployment.

The same source folds were previously used to select hyperparameters, so this
analysis is NOT an unbiased estimate of future performance. Validation and
final test files are never opened, and no model artifacts are overwritten.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_optimized_models import load_best_parameters

from experiment_hyperparameters import (
    FEATURE_SETS,
    create_model,
    create_source_folds,
    load_training_data,
)

from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

CANDIDATES = (
    "hybrid_v1_extended",
    "extended_v2",
)

THRESHOLDS = tuple(
    sorted({
        0.01,
        0.025,
        *np.round(np.arange(0.05, 1.0, 0.05), 3),
        0.975,
        0.99,
    })
)

REFERENCE_THRESHOLD = 0.50

# Exploratory constraints, not production requirements.

MIN_WORST_RECALL = 0.80

MAX_WORST_FPR = 0.35


# ============================================================
# 2. VALIDATE THRESHOLDS
# ============================================================

def validate_thresholds(thresholds) -> tuple[float, ...]:
    """
    Validate the threshold grid.
    """

    values = np.asarray(
        thresholds,
        dtype=float,
    )

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Thresholds must be a non-empty "
            "one-dimensional sequence"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Thresholds must be finite"
        )

    if np.any((values <= 0) | (values >= 1)):
        raise ValueError(
            "Thresholds must be strictly between 0 and 1"
        )

    if len(np.unique(values)) != len(values):
        raise ValueError(
            "Duplicate thresholds"
        )

    if not np.any(
        np.isclose(
            values,
            REFERENCE_THRESHOLD,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Threshold grid must include the 0.50 reference"
        )

    return tuple(
        float(value)
        for value in sorted(values)
    )


# ============================================================
# 3. CALCULATE METRICS
# ============================================================

def threshold_metrics(
    y_true,
    probabilities,
    threshold: float,
) -> dict:
    """
    Evaluate classification metrics at one threshold.

    Positive class:
        1 = phishing

    Negative class:
        0 = legitimate
    """

    labels = np.asarray(y_true)

    scores = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(scores)
    ):
        raise ValueError(
            "Labels and probabilities must be "
            "equal-length 1D arrays"
        )

    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "Evaluation must contain both classes"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Invalid prediction probabilities"
        )

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(
            "Invalid prediction probabilities"
        )

    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError(
            "Invalid classification threshold"
        )

    predicted = scores >= threshold

    positive = labels == 1

    tp = int(
        np.sum(predicted & positive)
    )

    tn = int(
        np.sum(~predicted & ~positive)
    )

    fp = int(
        np.sum(predicted & ~positive)
    )

    fn = int(
        np.sum(~predicted & positive)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": (
            fn / (fn + tp)
            if fn + tp
            else 0.0
        ),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "legitimate_samples": fp + tn,
        "phishing_samples": tp + fn,
    }


# ============================================================
# 4. RUN EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    optimized_parameters: dict,
    thresholds=THRESHOLDS,
) -> pd.DataFrame:
    """
    Train both optimized models across four source folds.

    2 models x 4 sources = 8 model fits.

    Every trained model is evaluated at every threshold.
    """

    grid = validate_thresholds(
        thresholds
    )

    if not set(CANDIDATES).issubset(
        optimized_parameters
    ):
        raise ValueError(
            "Missing optimized parameters"
        )

    folds = create_source_folds(
        dataframe
    )

    if (
        {fold["source"] for fold in folds} != set(SOURCES)
        or len(folds) != len(SOURCES)
    ):
        raise ValueError(
            "Incomplete source fold configuration"
        )

    results = []

    completed_fits = 0

    for candidate in CANDIDATES:

        features = FEATURE_SETS[
            candidate
        ]

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing features for {candidate}: "
                f"{sorted(missing)}"
            )

        params = optimized_parameters[
            candidate
        ]

        print("\n" + "=" * 70)

        print(
            f"MODEL: {candidate}"
        )

        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            fit_df = dataframe.iloc[
                fold["training_indices"]
            ]

            held_out_df = dataframe.iloc[
                fold["validation_indices"]
            ]

            # Verify source and domain-group isolation.

            verify_group_isolation(
                fit_df,
                held_out_df,
            )

            if belongs_to_source(
                fit_df,
                source,
            ).any():
                raise ValueError(
                    f"Source leakage detected: {source}"
                )

            # Train the optimized model.

            model = create_model(
                params
            )

            X_train = fit_df[features].to_numpy(
                dtype=np.float64
            )

            y_train = fit_df["label"].to_numpy(
                dtype=int
            )

            model.fit(
                X_train,
                y_train,
            )

            # Predict held-out source.

            X_evaluation = held_out_df[features].to_numpy(
                dtype=np.float64
            )

            probabilities = model.predict_proba(
                X_evaluation
            )[:, 1]

            labels = held_out_df["label"].to_numpy(
                dtype=int
            )

            completed_fits += 1

            print(
                f"[{completed_fits}/8] "
                f"{candidate:<19} "
                f"{source:<11}"
            )

            # Evaluate all thresholds without retraining.

            for threshold in grid:

                metrics = threshold_metrics(
                    labels,
                    probabilities,
                    threshold,
                )

                results.append({
                    "feature_set": candidate,
                    "evaluation_source": source,
                    "threshold": threshold,
                    "feature_count": len(features),
                    "training_samples": len(fit_df),
                    "evaluation_samples": len(held_out_df),
                    "parameters": json.dumps(
                        params,
                        sort_keys=True,
                    ),
                    **metrics,
                })

                if np.isclose(
                    threshold,
                    REFERENCE_THRESHOLD,
                ):
                    print(
                        f"  Threshold 0.50: "
                        f"Recall={metrics['recall']:.4f} "
                        f"FPR={metrics['fpr']:.4f}"
                    )

    result_df = pd.DataFrame(
        results
    )

    expected_rows = (
        len(CANDIDATES)
        * len(SOURCES)
        * len(grid)
    )

    if len(result_df) != expected_rows:
        raise ValueError(
            "Incomplete threshold evaluations"
        )

    if result_df.duplicated(
        [
            "feature_set",
            "evaluation_source",
            "threshold",
        ]
    ).any():
        raise ValueError(
            "Duplicate threshold evaluations"
        )

    return result_df


# ============================================================
# 5. SUMMARIZE RESULTS
# ============================================================

def summarize_results(
    results: pd.DataFrame,
    thresholds=THRESHOLDS,
) -> pd.DataFrame:
    """
    Aggregate results across dataset sources.

    Every source has equal weight in macro metrics.
    """

    grid = validate_thresholds(
        thresholds
    )

    expected = {
        (candidate, source, threshold)
        for candidate in CANDIDATES
        for source in SOURCES
        for threshold in grid
    }

    if results.empty or len(results) != len(expected):
        raise ValueError(
            "Incomplete threshold results"
        )

    actual = set(
        zip(
            results["feature_set"],
            results["evaluation_source"],
            results["threshold"],
        )
    )

    if actual != expected:
        raise ValueError(
            "Missing, duplicate or unexpected "
            "source/threshold combinations"
        )

    summary = (
        results
        .groupby(
            [
                "feature_set",
                "threshold",
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
            macro_fpr=(
                "fpr",
                "mean",
            ),
            worst_recall=(
                "recall",
                "min",
            ),
            worst_fpr=(
                "fpr",
                "max",
            ),
            total_false_positives=(
                "false_positives",
                "sum",
            ),
            total_false_negatives=(
                "false_negatives",
                "sum",
            ),
        )
        .sort_values(
            [
                "feature_set",
                "threshold",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Exploratory feasibility rules.

    summary["policy_feasible"] = (
        (
            summary["worst_recall"]
            >= MIN_WORST_RECALL
        )
        &
        (
            summary["worst_fpr"]
            <= MAX_WORST_FPR
        )
    )

    return summary


# ============================================================
# 6. BUILD DIAGNOSTIC SHORTLIST
# ============================================================

def build_shortlist(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify interesting threshold configurations.

    The output is exploratory.
    No threshold is approved for deployment.
    """

    if (
        summary.empty
        or "policy_feasible" not in summary.columns
    ):
        raise ValueError(
            "Invalid threshold summary"
        )

    choices = []

    for candidate in CANDIDATES:

        frame = summary.loc[
            summary["feature_set"] == candidate
        ]

        if frame.empty:
            raise ValueError(
                f"Missing threshold summary for {candidate}"
            )

        # Original threshold.

        baseline = frame.loc[
            np.isclose(
                frame["threshold"],
                REFERENCE_THRESHOLD,
            )
        ]

        if len(baseline) != 1:
            raise ValueError(
                f"Missing 0.50 reference for {candidate}"
            )

        choices.append(
            (
                "reference_0_50",
                baseline.iloc[0],
            )
        )

        # Highest macro F1.

        best_f1 = frame.sort_values(
            [
                "macro_f1",
                "worst_fpr",
                "threshold",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).iloc[0]

        choices.append(
            (
                "highest_macro_f1",
                best_f1,
            )
        )

        # Lowest FPR while satisfying recall floor.

        recall_ok = frame.loc[
            frame["worst_recall"]
            >= MIN_WORST_RECALL
        ]

        if not recall_ok.empty:

            safest = recall_ok.sort_values(
                [
                    "worst_fpr",
                    "macro_f1",
                    "threshold",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            ).iloc[0]

            choices.append(
                (
                    "lowest_fpr_with_recall_floor",
                    safest,
                )
            )

        # Candidate satisfying both constraints.

        feasible = frame.loc[
            frame["policy_feasible"]
        ]

        if not feasible.empty:

            top = feasible.sort_values(
                [
                    "macro_f1",
                    "worst_fpr",
                    "threshold",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            ).iloc[0]

            choices.append(
                (
                    "policy_feasible_candidate",
                    top,
                )
            )

    output = pd.DataFrame([
        {
            "selection_reason": reason,
            **row.to_dict(),
        }
        for reason, row in choices
    ])

    return output.reset_index(
        drop=True
    )


# ============================================================
# 7. SAVE REPORTS
# ============================================================

def save_reports(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> None:
    """
    Export experimental results to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "threshold_optimization_source_results.csv":
            results,

        "threshold_optimization_summary.csv":
            summary,

        "threshold_optimization_shortlist.csv":
            shortlist,
    }

    for filename, dataframe in reports.items():

        target = REPORT_DIR / filename

        dataframe.to_csv(
            target,
            index=False,
        )

        print(
            f"Saved: {target}"
        )


# ============================================================
# 8. MAIN
# ============================================================

def main() -> None:

    print("=" * 70)

    print(
        "PHISHGUARD AI - THRESHOLD OPTIMIZATION"
    )

    print("=" * 70)

    print(
        "Training split only; "
        "leave-one-source-out and domain-group isolation."
    )

    print(
        "No calibration: raw optimized model probabilities."
    )

    print("\nLoading training data...")

    dataframe = load_training_data()

    print("\nLoading optimized parameters...")

    parameters = load_best_parameters()

    print(
        f"Training URLs: {len(dataframe):,}"
    )

    print(
        f"Thresholds: {len(THRESHOLDS)}"
    )

    print("\nRunning experiments...")

    results = run_experiment(
        dataframe,
        parameters,
    )

    summary = summarize_results(
        results
    )

    shortlist = build_shortlist(
        summary
    )

    save_reports(
        results,
        summary,
        shortlist,
    )

    print("\n" + "=" * 95)

    print(
        "DIAGNOSTIC SHORTLIST"
    )

    print("=" * 95)

    columns = [
        "feature_set",
        "selection_reason",
        "threshold",
        "macro_f1",
        "macro_recall",
        "macro_fpr",
        "worst_recall",
        "worst_fpr",
        "policy_feasible",
    ]

    print(
        shortlist[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    for candidate in CANDIDATES:

        candidate_rows = summary.loc[
            summary["feature_set"] == candidate
        ]

        feasible_count = int(
            candidate_rows["policy_feasible"].sum()
        )

        print(
            f"{candidate}: {feasible_count} "
            f"of {len(candidate_rows)} thresholds "
            f"satisfy both provisional constraints"
        )

    print("\n" + "=" * 70)

    print(
        "THRESHOLD OPTIMIZATION COMPLETED"
    )

    print("=" * 70)

    print(
        "This sweep does not select "
        "or deploy a final threshold."
    )

    print(
        "Validation/test datasets were not accessed."
    )

    print(
        "Existing model artifacts were not overwritten."
    )


if __name__ == "__main__":
    main()