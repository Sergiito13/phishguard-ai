"""
PhishGuard AI: targeted cross-source feature ablation (diagnostic only).

Compares prespecified feature removals after cross-source error analysis.
Uses ONLY training split with domain-disjoint leave-one-source-out folds.
The same sources informed model selection, so results are exploratory, not
independent estimates. Never reads reserved validation/test files or writes
model artifacts. Outputs aggregated statistics without URLs or domains.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_optimized_models import load_best_parameters
from experiment_hyperparameters import (
    FEATURE_SETS,
    create_model,
    create_source_folds,
    load_training_data,
)
from experiment_threshold_optimization import threshold_metrics
from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

THRESHOLD = 0.50

# Feature removal hypotheses derived from previous diagnostics.

ABLATIONS = {
    "hybrid_v1_extended": {
        "full": (),
        "without_https": (
            "is_https",
        ),
        "without_slash_count": (
            "slash_count",
        ),
        "without_https_and_slash": (
            "is_https",
            "slash_count",
        ),
    },

    "extended_v2": {
        "full": (),

        "without_ip_and_port": (
            "v2_has_ip",
            "v2_has_explicit_port",
        ),

        "without_entropy": (
            "v2_url_entropy",
            "v2_hostname_entropy",
            "v2_domain_label_entropy",
            "v2_path_entropy",
        ),

        "without_length_hyphen": (
            "v2_url_length",
            "v2_hostname_length",
            "v2_registered_domain_length",
            "v2_hyphen_count",
            "v2_hostname_hyphen_count",
        ),
    },
}


# ============================================================
# 2. BUILD FEATURE CONFIGURATIONS
# ============================================================

def build_configurations(
    feature_sets=FEATURE_SETS,
    ablations=ABLATIONS,
):
    """
    Generate feature configurations for each ablation.

    Validate:
    - Original feature configuration.
    - Removed features.
    - Duplicate feature configurations.
    - Empty feature sets.
    """

    configurations = []

    for model_name, variants in ablations.items():

        if (
            model_name not in feature_sets
            or "full" not in variants
        ):
            raise ValueError(
                f"Missing baseline features/configuration: {model_name}"
            )

        original_features = list(
            feature_sets[model_name]
        )

        if (
            not original_features
            or len(original_features) != len(set(original_features))
        ):
            raise ValueError(
                f"Empty or duplicate features: {model_name}"
            )

        seen = set()

        for variant_name, excluded in variants.items():

            if (
                not isinstance(excluded, (tuple, list))
                or len(excluded) != len(set(excluded))
            ):
                raise ValueError(
                    f"Duplicate or invalid excluded features: {variant_name}"
                )

            if variant_name == "full" and excluded:
                raise ValueError(
                    "Full baseline cannot exclude features"
                )

            unknown = (
                set(excluded)
                - set(original_features)
            )

            if unknown:
                raise ValueError(
                    f"Unknown excluded features: {sorted(unknown)}"
                )

            retained = tuple(
                feature
                for feature in original_features
                if feature not in excluded
            )

            if (
                not retained
                or retained in seen
            ):
                raise ValueError(
                    f"Empty or duplicate ablation: "
                    f"{model_name}/{variant_name}"
                )

            seen.add(retained)

            configurations.append({
                "feature_set": model_name,
                "ablation": variant_name,
                "features": list(retained),
                "excluded_features": ",".join(excluded),
            })

    return configurations


# ============================================================
# 3. RUN CROSS-SOURCE EXPERIMENT
# ============================================================

def run_experiment(
    dataframe,
    best_parameters,
    configurations=None,
):
    """
    Train one fresh model for each configuration
    and held-out dataset source.

    8 configurations x 4 sources = 32 model fits.

    Only the original training split is used.
    """

    if configurations is None:
        configurations = build_configurations()

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
            "Incomplete source holdout folds"
        )

    results = []

    for configuration in configurations:

        model_name = configuration["feature_set"]

        ablation_name = configuration["ablation"]

        features = configuration["features"]

        if model_name not in best_parameters:
            raise ValueError(
                f"Missing optimized parameters: {model_name}"
            )

        if (
            not features
            or set(features) - set(dataframe.columns)
        ):
            raise ValueError(
                f"Missing features for "
                f"{model_name}/{ablation_name}"
            )

        print("\n" + "=" * 70)

        print(
            f"MODEL: {model_name} / {ablation_name}"
        )

        print(
            f"Features: {len(features)}"
        )

        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            # Split training and held-out source.

            fit_df = dataframe.iloc[
                fold["training_indices"]
            ]

            held_df = dataframe.iloc[
                fold["validation_indices"]
            ]

            # Verify domain-group isolation.

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
                    f"Source leakage: {source}"
                )

            y_train = fit_df[
                "label"
            ].to_numpy(
                dtype=int
            )

            y_evaluation = held_df[
                "label"
            ].to_numpy(
                dtype=int
            )

            if (
                set(np.unique(y_train)) != {0, 1}
                or set(np.unique(y_evaluation)) != {0, 1}
            ):
                raise ValueError(
                    f"Both classes required: {source}"
                )

            # Create fresh optimized model.

            model = create_model(
                best_parameters[model_name]
            )

            # Training features.

            X_train = fit_df[
                features
            ].to_numpy(
                dtype=np.float64
            )

            # Held-out source features.

            X_evaluation = held_df[
                features
            ].to_numpy(
                dtype=np.float64
            )

            # Train model.

            model.fit(
                X_train,
                y_train,
            )

            # Predict phishing probabilities.

            probabilities = np.asarray(
                model.predict_proba(
                    X_evaluation
                )[:, 1],
                dtype=float,
            )

            # Classification metrics.

            metrics = threshold_metrics(
                y_evaluation,
                probabilities,
                THRESHOLD,
            )

            results.append({
                "feature_set": model_name,
                "ablation": ablation_name,
                "evaluation_source": source,
                "feature_count": len(features),
                "excluded_features": configuration[
                    "excluded_features"
                ],
                "training_samples": len(fit_df),
                "evaluation_samples": len(held_df),

                "roc_auc": float(
                    roc_auc_score(
                        y_evaluation,
                        probabilities,
                    )
                ),

                "average_precision": float(
                    average_precision_score(
                        y_evaluation,
                        probabilities,
                    )
                ),

                **metrics,
            })

            print(
                f"{source:<11} "
                f"FPR={metrics['fpr']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f}"
            )

    results_df = pd.DataFrame(
        results
    )

    expected_rows = (
        len(configurations)
        * len(SOURCES)
    )

    if (
        len(results_df) != expected_rows
        or results_df.duplicated(
            [
                "feature_set",
                "ablation",
                "evaluation_source",
            ]
        ).any()
    ):
        raise ValueError(
            "Missing or duplicate ablation evaluations"
        )

    return results_df


# ============================================================
# 4. SUMMARIZE RESULTS
# ============================================================

def summarize_results(
    results,
    configurations=None,
):
    """
    Calculate source-macro metrics.

    Every dataset source contributes equally.

    Include worst-source recall and FPR to identify
    configurations with poor generalization.
    """

    if configurations is None:
        configurations = build_configurations()

    expected = {
        (
            configuration["feature_set"],
            configuration["ablation"],
            source,
        )
        for configuration in configurations
        for source in SOURCES
    }

    if (
        results.empty
        or len(results) != len(expected)
        or set(
            zip(
                results["feature_set"],
                results["ablation"],
                results["evaluation_source"],
            )
        ) != expected
    ):
        raise ValueError(
            "Incomplete or duplicate ablation results"
        )

    summary = (
        results.groupby(
            [
                "feature_set",
                "ablation",
            ],
            as_index=False,
        )
        .agg(
            feature_count=(
                "feature_count",
                "first",
            ),

            evaluated_sources=(
                "evaluation_source",
                "nunique",
            ),

            macro_f1=(
                "f1",
                "mean",
            ),

            macro_recall=(
                "recall",
                "mean",
            ),

            worst_recall=(
                "recall",
                "min",
            ),

            macro_fpr=(
                "fpr",
                "mean",
            ),

            worst_fpr=(
                "fpr",
                "max",
            ),

            macro_roc_auc=(
                "roc_auc",
                "mean",
            ),

            macro_average_precision=(
                "average_precision",
                "mean",
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
    )

    return (
        summary.sort_values(
            [
                "feature_set",
                "ablation",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 5. COMPARE AGAINST ORIGINAL MODEL
# ============================================================

def compare_to_baseline(
    summary,
):
    """
    Compare each ablation with its corresponding
    full-feature model.

    Delta = ablated model - original model.

    Positive recall delta = improvement.
    Negative FPR delta = improvement.
    """

    full = (
        summary.loc[
            summary["ablation"] == "full"
        ]
        .set_index(
            "feature_set"
        )
    )

    if (
        len(full) != len(ABLATIONS)
        or set(full.index) != set(ABLATIONS)
    ):
        raise ValueError(
            "Missing exactly one baseline per feature family"
        )

    comparison = summary.copy()

    metrics = (
        "macro_f1",
        "macro_recall",
        "worst_recall",
        "macro_fpr",
        "worst_fpr",
        "macro_roc_auc",
    )

    for metric in metrics:

        comparison[f"delta_{metric}"] = (
            comparison[metric].to_numpy()
            - comparison["feature_set"]
            .map(full[metric])
            .to_numpy()
        )

    return comparison


# ============================================================
# 6. SAVE REPORTS
# ============================================================

def save_reports(
    results,
    summary,
    comparison,
):
    """
    Save aggregate CSV reports.

    No raw URLs or domain identifiers are exported.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "cross_source_ablation_source_results.csv":
            results,

        "cross_source_ablation_summary.csv":
            summary,

        "cross_source_ablation_comparison.csv":
            comparison,
    }

    sensitive_columns = {
        "url",
        "group_key",
        "sources",
    }

    for filename, dataframe in reports.items():

        if (
            dataframe.empty
            or sensitive_columns.intersection(
                dataframe.columns
            )
        ):
            raise ValueError(
                f"Invalid or sensitive report: {filename}"
            )

        destination = (
            REPORT_DIR / filename
        )

        dataframe.to_csv(
            destination,
            index=False,
        )

        print(
            f"Saved: {destination}"
        )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHISHGUARD AI - TARGETED CROSS-SOURCE FEATURE ABLATION"
    )

    print("=" * 72)

    print(
        "Training data only; group-isolated source holdouts."
    )

    print(
        "Exploratory analysis, not independent validation."
    )

    configurations = build_configurations()

    # Load training split only.

    dataframe = load_training_data()

    # Load optimized hyperparameters.

    parameters = load_best_parameters()

    print(
        f"\nTraining URLs: {len(dataframe):,}"
    )

    print(
        f"Configurations: {len(configurations)}"
    )

    print(
        f"Model fits: {len(configurations) * len(SOURCES)}"
    )

    # Run experiment.

    results = run_experiment(
        dataframe,
        parameters,
        configurations,
    )

    # Generate summaries.

    summary = summarize_results(
        results,
        configurations,
    )

    comparison = compare_to_baseline(
        summary
    )

    # Save reports.

    save_reports(
        results,
        summary,
        comparison,
    )

    # Display ablation comparison.

    print("\n" + "=" * 100)

    print(
        "ABLATION COMPARISON"
    )

    print("=" * 100)

    columns = [
        "feature_set",
        "ablation",
        "feature_count",
        "macro_f1",
        "macro_recall",
        "macro_fpr",
        "worst_recall",
        "worst_fpr",
        "delta_macro_f1",
        "delta_macro_fpr",
        "delta_worst_fpr",
    ]

    print(
        comparison[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Show focal source comparisons.

    print("\n" + "=" * 100)

    print(
        "FOCAL SOURCE COMPARISON"
    )

    print("=" * 100)

    focal = results.loc[
        (
            (
                results["feature_set"]
                == "hybrid_v1_extended"
            )
            &
            (
                results["evaluation_source"]
                == "url_phish"
            )
        )
        |
        (
            (
                results["feature_set"]
                == "extended_v2"
            )
            &
            (
                results["evaluation_source"]
                == "phishbd"
            )
        )
    ]

    focal_columns = [
        "feature_set",
        "ablation",
        "evaluation_source",
        "fpr",
        "recall",
        "false_positives",
        "false_negatives",
    ]

    print(
        focal[focal_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 72)

    print(
        "FEATURE ABLATION COMPLETED"
    )

    print("=" * 72)

    print(
        "No model was saved or overwritten."
    )

    print(
        "Reserved validation/test data were not accessed."
    )

    print(
        "No ablation was selected for deployment."
    )


if __name__ == "__main__":
    main()