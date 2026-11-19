"""
PhishGuard AI: diagnose cross-source prediction errors.

Training split only. For each source, train a fresh optimized model without
that source (and without overlapping registered-domain groups), then examine
its errors on the held-out source.

This is an exploratory diagnostic: prior hyperparameter selection has
already used these sources.

Exports only aggregate statistics, never raw URLs or domain identifiers.
Does not touch validation/test splits or save trained models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

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

THRESHOLD = 0.50

SCORE_BINS = (
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
)


# ============================================================
# 2. VALIDATE PREDICTION SCORES
# ============================================================

def validate_scores(
    labels,
    scores,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate binary ground truth and phishing scores.
    """

    labels = np.asarray(labels)

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) == 0
        or len(labels) != len(scores)
    ):
        raise ValueError(
            "Labels and scores must be non-empty "
            "1D arrays of equal size"
        )

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "Evaluation requires both classes"
        )

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "Scores must be finite probabilities in [0, 1]"
        )

    return labels.astype(int), scores


# ============================================================
# 3. CONFUSION MATRIX AND CLASSIFICATION METRICS
# ============================================================

def confusion_summary(
    labels,
    scores,
) -> dict:
    """
    Calculate prediction metrics and score distributions.
    """

    labels, scores = validate_scores(
        labels,
        scores,
    )

    predicted = scores >= THRESHOLD

    positive = labels == 1

    tp = int(
        np.sum(predicted & positive)
    )

    fp = int(
        np.sum(predicted & ~positive)
    )

    fn = int(
        np.sum(~predicted & positive)
    )

    tn = int(
        np.sum(~predicted & ~positive)
    )

    return {
        "samples": len(labels),

        "legitimate_samples": fp + tn,

        "phishing_samples": tp + fn,

        "true_positives": tp,

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "recall": tp / (tp + fn),

        "fpr": fp / (fp + tn),

        "precision": (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),

        "average_precision": float(
            average_precision_score(
                labels,
                scores,
            )
        ),

        "legitimate_score_median": float(
            np.median(
                scores[~positive]
            )
        ),

        "phishing_score_median": float(
            np.median(
                scores[positive]
            )
        ),

        "legitimate_score_p90": float(
            np.quantile(
                scores[~positive],
                0.90,
            )
        ),

        "phishing_score_p10": float(
            np.quantile(
                scores[positive],
                0.10,
            )
        ),
    }


# ============================================================
# 4. PREDICTION SCORE DISTRIBUTION
# ============================================================

def score_distribution(
    labels,
    scores,
) -> pd.DataFrame:
    """
    Generate probability histograms for both classes.

    The last bin includes scores exactly equal to 1.0.
    """

    labels, scores = validate_scores(
        labels,
        scores,
    )

    rows = []

    for label in (0, 1):

        selected = scores[
            labels == label
        ]

        counts, edges = np.histogram(
            selected,
            bins=SCORE_BINS,
        )

        for index, count in enumerate(counts):

            rows.append({
                "label": label,

                "score_low": float(
                    edges[index]
                ),

                "score_high": float(
                    edges[index + 1]
                ),

                "count": int(count),

                "class_samples": len(selected),

                "class_fraction": float(
                    count / len(selected)
                ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 5. COMPARE FEATURE DISTRIBUTIONS
# ============================================================

def compare_values(
    first,
    second,
) -> dict:
    """
    Compare two numerical feature distributions.

    Uses standardized mean difference.

    This is descriptive statistics, not causal
    inference or feature importance.

    First group:
        Target/error group.

    Second group:
        Comparator/reference group.
    """

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError(
            "Non-finite feature values"
        )

    result = {
        "target_samples": len(first),

        "reference_samples": len(second),

        "target_mean": np.nan,

        "reference_mean": np.nan,

        "target_median": np.nan,

        "reference_median": np.nan,

        "standardized_difference": np.nan,

        "absolute_difference": np.nan,

        "constant_nonoverlap": False,
    }

    if len(first) == 0 or len(second) == 0:
        return result

    first_mean = float(
        np.mean(first)
    )

    second_mean = float(
        np.mean(second)
    )

    pooled_variance = (
        float(np.var(first))
        + float(np.var(second))
    ) / 2.0

    result.update({
        "target_mean": first_mean,

        "reference_mean": second_mean,

        "target_median": float(
            np.median(first)
        ),

        "reference_median": float(
            np.median(second)
        ),
    })

    if pooled_variance > 0:

        difference = (
            first_mean - second_mean
        ) / np.sqrt(pooled_variance)

        result["standardized_difference"] = float(
            difference
        )

        result["absolute_difference"] = float(
            abs(difference)
        )

    elif np.isclose(
        first_mean,
        second_mean,
    ):

        result["standardized_difference"] = 0.0

        result["absolute_difference"] = 0.0

    else:

        result["constant_nonoverlap"] = True

    return result


# ============================================================
# 6. FEATURE DIAGNOSTICS
# ============================================================

def feature_diagnostics(
    fit_df: pd.DataFrame,
    held_out_df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare two types of feature differences.

    1. Source distribution shifts:
       Held-out source versus training sources.

    2. Classification error profiles:
       False positives versus true negatives.
       False negatives versus true positives.

    Comparisons use the same actual class.
    """

    labels, scores = validate_scores(
        labels,
        scores,
    )

    if len(held_out_df) != len(labels):
        raise ValueError(
            "Held-out frame and predictions must match"
        )

    if any(
        feature not in fit_df
        or feature not in held_out_df
        for feature in features
    ):
        raise ValueError(
            "Missing required features"
        )

    predicted = scores >= THRESHOLD

    shifts = []

    errors = []

    comparisons = (
        (
            0,
            predicted & (labels == 0),
            ~predicted & (labels == 0),
            "FP",
            "TN",
        ),

        (
            1,
            ~predicted & (labels == 1),
            predicted & (labels == 1),
            "FN",
            "TP",
        ),
    )

    for (
        label,
        mistake,
        correct,
        error_kind,
        comparison,
    ) in comparisons:

        train_class = fit_df.loc[
            fit_df["label"] == label
        ]

        held_class = held_out_df.loc[
            held_out_df["label"] == label
        ]

        if train_class.empty or held_class.empty:
            raise ValueError(
                "Each class must exist in training "
                "and evaluation"
            )

        for feature in features:

            train_values = train_class[
                feature
            ].to_numpy(
                dtype=float
            )

            held_values = held_class[
                feature
            ].to_numpy(
                dtype=float
            )

            all_held_values = held_out_df[
                feature
            ].to_numpy(
                dtype=float
            )

            shifts.append({
                "label": label,

                "feature": feature,

                **compare_values(
                    held_values,
                    train_values,
                ),
            })

            errors.append({
                "label": label,

                "error_type": error_kind,

                "reference_type": comparison,

                "feature": feature,

                **compare_values(
                    all_held_values[mistake],
                    all_held_values[correct],
                ),
            })

    return (
        pd.DataFrame(shifts),
        pd.DataFrame(errors),
    )


# ============================================================
# 7. CROSS-SOURCE ERROR EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    parameters: dict,
) -> dict[str, pd.DataFrame]:
    """
    Analyze both optimized models across four sources.

    Training:
        All sources except the held-out source.

    Evaluation:
        Held-out source.

    Group isolation:
        Registered domains cannot overlap.

    Total:
        8 model fits.
    """

    if not set(CANDIDATES).issubset(
        parameters
    ):
        raise ValueError(
            "Missing optimized model parameters"
        )

    folds = create_source_folds(
        dataframe
    )

    if (
        len(folds) != len(SOURCES)
        or {
            fold["source"]
            for fold in folds
        } != set(SOURCES)
    ):
        raise ValueError(
            "Expected exactly four source holdout folds"
        )

    summaries = []

    distributions = []

    shifts = []

    profiles = []

    for model_name in CANDIDATES:

        features = FEATURE_SETS[
            model_name
        ]

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing model features: {sorted(missing)}"
            )

        print("\n" + "=" * 70)

        print(
            f"MODEL: {model_name}"
        )

        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            fit_df = dataframe.iloc[
                fold["training_indices"]
            ]

            held_df = dataframe.iloc[
                fold["validation_indices"]
            ]

            # Verify domain isolation.

            verify_group_isolation(
                fit_df,
                held_df,
            )

            # Verify source isolation.

            if belongs_to_source(
                fit_df,
                source,
            ).any():
                raise ValueError(
                    f"Source leakage detected: {source}"
                )

            labels = held_df[
                "label"
            ].to_numpy(
                dtype=int
            )

            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(
                    "Held-out source requires both labels"
                )

            # Train model.

            model = create_model(
                parameters[model_name]
            )

            model.fit(
                fit_df[features].to_numpy(
                    dtype=np.float64
                ),
                fit_df["label"].to_numpy(
                    dtype=int
                ),
            )

            # Predict held-out source.

            scores = model.predict_proba(
                held_df[features].to_numpy(
                    dtype=np.float64
                )
            )[:, 1]

            scores = validate_scores(
                labels,
                scores,
            )[1]

            common = {
                "feature_set": model_name,
                "evaluation_source": source,
            }

            # General classification metrics.

            metrics = confusion_summary(
                labels,
                scores,
            )

            summaries.append({
                **common,
                **metrics,
            })

            # Probability distribution.

            distributions.append(
                score_distribution(
                    labels,
                    scores,
                ).assign(**common)
            )

            # Feature shifts and errors.

            shift, profile = feature_diagnostics(
                fit_df,
                held_df,
                labels,
                scores,
                features,
            )

            shifts.append(
                shift.assign(**common)
            )

            profiles.append(
                profile.assign(**common)
            )

            print(
                f"{model_name:<19} "
                f"{source:<11} "
                f"FP={metrics['false_positives']:>6,} "
                f"FN={metrics['false_negatives']:>6,} "
                f"FPR={metrics['fpr']:.4f} "
                f"Recall={metrics['recall']:.4f}"
            )

    result = {
        "summary": pd.DataFrame(
            summaries
        ),

        "score_distribution": pd.concat(
            distributions,
            ignore_index=True,
        ),

        "feature_shift": pd.concat(
            shifts,
            ignore_index=True,
        ),

        "error_profiles": pd.concat(
            profiles,
            ignore_index=True,
        ),
    }

    expected_rows = (
        len(CANDIDATES)
        * len(SOURCES)
    )

    if len(result["summary"]) != expected_rows:
        raise ValueError(
            "Incomplete diagnostic experiment"
        )

    return result


# ============================================================
# 8. SAVE REPORTS
# ============================================================

def save_reports(
    result: dict[str, pd.DataFrame],
) -> None:
    """
    Save aggregate statistics only.

    No raw URLs, group keys or individual
    domain identifiers are exported.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = {
        "summary":
            "cross_source_errors_summary.csv",

        "score_distribution":
            "cross_source_errors_score_distribution.csv",

        "feature_shift":
            "cross_source_errors_feature_shift.csv",

        "error_profiles":
            "cross_source_errors_feature_profiles.csv",
    }

    for key, filename in expected.items():

        if (
            key not in result
            or result[key].empty
        ):
            raise ValueError(
                f"Missing report data: {key}"
            )

        private_columns = {
            "url",
            "group_key",
            "sources",
        }

        if private_columns & set(
            result[key].columns
        ):
            raise ValueError(
                "A report would expose raw "
                "URL/domain/source-membership data"
            )

        destination = (
            REPORT_DIR / filename
        )

        result[key].to_csv(
            destination,
            index=False,
        )

        print(
            f"Saved: {destination}"
        )


# ============================================================
# 9. MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        "PHISHGUARD AI - CROSS-SOURCE ERROR DIAGNOSTICS"
    )

    print("=" * 72)

    print(
        "Only training split; no raw URL export; "
        "no model artifacts modified."
    )

    # Load training data only.

    dataframe = load_training_data()

    # Load optimized hyperparameters.

    params = load_best_parameters()

    print(
        f"Training URLs: {len(dataframe):,}"
    )

    # Run experiments.

    result = run_experiment(
        dataframe,
        params,
    )

    # Save aggregate results.

    save_reports(
        result
    )

    # Display source summary.

    print("\nSOURCE SUMMARY")

    columns = [
        "feature_set",
        "evaluation_source",
        "false_positives",
        "false_negatives",
        "fpr",
        "recall",
        "legitimate_score_median",
        "phishing_score_median",
    ]

    print(
        result["summary"][columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Display the most relevant feature contrasts.

    targets = (
        (
            "hybrid_v1_extended",
            "url_phish",
            "FP",
        ),

        (
            "extended_v2",
            "phishbd",
            "FN",
        ),
    )

    for (
        model_name,
        source,
        error_type,
    ) in targets:

        profiles = result[
            "error_profiles"
        ]

        subset = profiles.loc[
            (
                profiles["feature_set"]
                == model_name
            )
            &
            (
                profiles["evaluation_source"]
                == source
            )
            &
            (
                profiles["error_type"]
                == error_type
            )
        ]

        subset = (
            subset.sort_values(
                "absolute_difference",
                ascending=False,
            )
            .head(8)
        )

        print(
            f"\nTOP FEATURE CONTRASTS: "
            f"{model_name} / {source} / {error_type}"
        )

        print(
            subset[
                [
                    "feature",
                    "target_samples",
                    "reference_samples",
                    "standardized_difference",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    print(
        "\nDiagnostic only. Differences do not "
        "prove causation or incorrect labels."
    )

    print(
        "Reserved validation/test data were not accessed."
    )


if __name__ == "__main__":
    main()