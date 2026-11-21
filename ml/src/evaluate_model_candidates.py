"""PhishGuard AI: compare two Extended V2 candidates (diagnostic only).

Train on the existing training split, evaluate on the existing validation split,
and repeat with four leave-one-source-out training folds. No test data, model
artifacts, URL-level reports, threshold tuning or deployment changes.

WARNING: validation has already informed earlier development decisions; metrics
are not independent estimates of future performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_optimized_models import (
    create_evaluation_folds,
    load_best_parameters,
)
from experiment_features_v2 import load_datasets
from experiment_hyperparameters import FEATURE_SETS, create_model
from experiment_source_bias_mitigation import calculate_weights
from experiment_threshold_optimization import threshold_metrics
from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
FEATURE_SET = "extended_v2"
FEATURES = list(FEATURE_SETS[FEATURE_SET])
STRATEGIES = ("uniform", "source_class")
THRESHOLD = 0.50

# Exploratory screening criteria, NOT universal security guarantees.
MAX_WORST_SOURCE_FPR = 0.10
MIN_WORST_SOURCE_RECALL = 0.80

REPORT_FILES = {
    "folds": "model_candidates_folds.csv",
    "validation_sources": "model_candidates_validation_sources.csv",
    "comparison": "model_candidates_comparison.csv",
    "weights": "model_candidates_weights.csv",
}


def validate_inputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> None:
    """Verify the candidate feature schema and leakage boundaries."""

    if FEATURE_SET not in parameters:
        raise ValueError("Missing optimized Extended V2 parameters")

    if not FEATURES or len(FEATURES) != len(set(FEATURES)):
        raise ValueError("Invalid feature configuration")

    needed = {"label", "sources", "group_key", "is_https", *FEATURES}

    for name, frame in (("training", train), ("validation", validation)):

        missing = needed - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing {name} columns: {sorted(missing)}"
            )

        if frame.empty or set(frame["label"].unique()) != {0, 1}:
            raise ValueError(f"{name} requires both binary classes")

        if (
            frame["group_key"].isna().any()
            or frame["group_key"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Missing {name} group keys")

        if (
            frame["sources"].isna().any()
            or frame["sources"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Missing {name} source memberships")

        if not frame["is_https"].isin((0, 1)).all():
            raise ValueError(f"Invalid {name} HTTPS flags")

        features = frame[FEATURES].to_numpy(dtype=np.float64)

        if not np.isfinite(features).all():
            raise ValueError(f"Invalid {name} feature values")

    verify_group_isolation(train, validation)

    if "url" in train and "url" in validation:
        if not set(train["url"]).isdisjoint(set(validation["url"])):
            raise ValueError("URL leakage detected")


def score_model(model, X: np.ndarray) -> np.ndarray:
    """Extract class-1 probabilities with explicit class-order checking."""

    classes = list(model.classes_)

    if classes != [0, 1]:
        raise ValueError("Expected model classes [0, 1]")

    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=np.float64,
    )

    if probabilities.shape != (len(X), 2):
        raise ValueError("Invalid prediction shape")

    scores = probabilities[:, 1]

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Invalid phishing probabilities")

    return scores


def evaluate_scores(labels, scores) -> dict:
    """Evaluate a labeled population at a fixed threshold of 0.50."""

    y = np.asarray(labels, dtype=int)
    probabilities = np.asarray(scores, dtype=np.float64)

    values = threshold_metrics(
        y,
        probabilities,
        THRESHOLD,
    )

    values["roc_auc"] = float(
        roc_auc_score(y, probabilities)
    )

    values["average_precision"] = float(
        average_precision_score(y, probabilities)
    )

    return values


def run_experiment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict,
) -> dict[str, pd.DataFrame]:
    """Fit 2 strategies x (overall + 4 held-out sources) = 10 models."""

    validate_inputs(train, validation, parameters)

    folds = create_evaluation_folds(train, validation)

    expected = {"overall", *SOURCES}

    if (
        len(folds) != len(expected)
        or {fold["source"] for fold in folds} != expected
    ):
        raise ValueError("Incomplete or duplicated evaluation folds")

    fold_rows = []
    source_rows = []
    weight_rows = []

    for fold in folds:

        source = fold["source"]

        fit = train.iloc[fold["train_indices"]]
        held = validation.iloc[fold["validation_indices"]]

        verify_group_isolation(fit, held)

        if (
            source != "overall"
            and belongs_to_source(fit, source).any()
        ):
            raise ValueError(f"Source leakage detected: {source}")

        if (
            set(fit["label"].unique()) != {0, 1}
            or set(held["label"].unique()) != {0, 1}
        ):
            raise ValueError(f"Both classes required: {source}")

        X_fit = fit[FEATURES].to_numpy(dtype=np.float64)
        y_fit = fit["label"].to_numpy(dtype=int)

        X_held = held[FEATURES].to_numpy(dtype=np.float64)
        y_held = held["label"].to_numpy(dtype=int)

        for strategy in STRATEGIES:

            # Calculate weights using training data only.
            weights = calculate_weights(fit, strategy)

            model = create_model(parameters[FEATURE_SET])

            model.fit(
                X_fit,
                y_fit,
                sample_weight=weights,
            )

            scores = score_model(model, X_held)

            metrics = evaluate_scores(y_held, scores)

            fold_rows.append({
                "feature_set": FEATURE_SET,
                "strategy": strategy,
                "evaluation": source,
                "training_samples": len(fit),
                "evaluation_samples": len(held),
                "threshold": THRESHOLD,
                **metrics,
            })

            weight_rows.append({
                "strategy": strategy,
                "evaluation": source,
                "training_samples": len(fit),
                "weighted": weights is not None,
                "mean_weight": (
                    float(np.mean(weights))
                    if weights is not None
                    else 1.0
                ),
                "max_weight": (
                    float(np.max(weights))
                    if weights is not None
                    else 1.0
                ),
            })

            print(
                f"{source:<11} "
                f"{strategy:<12} "
                f"Recall={metrics['recall']:.4f} "
                f"FPR={metrics['fpr']:.4f}"
            )

            # Evaluate the full-training model by source.
            # Multi-source URLs are included in each relevant source.
            if source == "overall":

                for evaluation_source in SOURCES:

                    mask = belongs_to_source(
                        held,
                        evaluation_source,
                    ).to_numpy(dtype=bool)

                    if (
                        not mask.any()
                        or set(np.unique(y_held[mask])) != {0, 1}
                    ):
                        raise ValueError(
                            f"Source lacks both classes: {evaluation_source}"
                        )

                    source_metrics = evaluate_scores(
                        y_held[mask],
                        scores[mask],
                    )

                    source_rows.append({
                        "feature_set": FEATURE_SET,
                        "strategy": strategy,
                        "evaluation_source": evaluation_source,
                        "evaluation_samples": int(mask.sum()),
                        "threshold": THRESHOLD,
                        **source_metrics,
                    })

    results = {
        "folds": pd.DataFrame(fold_rows),
        "validation_sources": pd.DataFrame(source_rows),
        "weights": pd.DataFrame(weight_rows),
    }

    results["comparison"] = summarize_results(
        results["folds"]
    )

    return results


def summarize_results(folds: pd.DataFrame) -> pd.DataFrame:
    """Compare robustness without automatically promoting a model."""

    required = {
        "strategy",
        "evaluation",
        "f1",
        "recall",
        "fpr",
        "precision",
        "roc_auc",
        "average_precision",
    }

    if not required.issubset(folds.columns):
        raise ValueError("Missing fold metrics")

    expected = {
        (strategy, fold)
        for strategy in STRATEGIES
        for fold in ("overall", *SOURCES)
    }

    observed = list(
        zip(folds["strategy"], folds["evaluation"])
    )

    if (
        len(observed) != len(expected)
        or set(observed) != expected
    ):
        raise ValueError("Incomplete or duplicate candidate results")

    rows = []

    for strategy in STRATEGIES:

        group = folds.loc[
            folds["strategy"] == strategy
        ].set_index("evaluation")

        overall = group.loc["overall"]
        unseen = group.loc[list(SOURCES)]

        worst_fpr = float(unseen["fpr"].max())
        worst_recall = float(unseen["recall"].min())

        rows.append({
            "feature_set": FEATURE_SET,
            "strategy": strategy,

            "overall_f1": float(overall["f1"]),
            "overall_recall": float(overall["recall"]),
            "overall_fpr": float(overall["fpr"]),
            "overall_roc_auc": float(overall["roc_auc"]),
            "overall_average_precision": float(
                overall["average_precision"]
            ),

            "cross_source_macro_f1": float(
                unseen["f1"].mean()
            ),
            "cross_source_macro_recall": float(
                unseen["recall"].mean()
            ),
            "cross_source_macro_fpr": float(
                unseen["fpr"].mean()
            ),

            "worst_source_recall": worst_recall,
            "worst_source_fpr": worst_fpr,

            "meets_exploratory_guardrails": bool(
                worst_fpr <= MAX_WORST_SOURCE_FPR
                and worst_recall >= MIN_WORST_SOURCE_RECALL
            ),
        })

    summary = pd.DataFrame(rows)

    baseline = summary.set_index("strategy").loc["uniform"]

    summary["delta_macro_f1_vs_uniform"] = (
        summary["cross_source_macro_f1"]
        - baseline["cross_source_macro_f1"]
    )

    summary["delta_macro_fpr_vs_uniform"] = (
        summary["cross_source_macro_fpr"]
        - baseline["cross_source_macro_fpr"]
    )

    return summary


def save_reports(
    reports: dict[str, pd.DataFrame],
) -> None:
    """Write aggregate-only CSV reports."""

    if set(reports) != set(REPORT_FILES):
        raise ValueError("Incomplete candidate reports")

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    for name, frame in reports.items():

        if (
            frame.empty
            or not forbidden.isdisjoint(frame.columns)
        ):
            raise ValueError(
                f"Empty or sensitive report: {name}"
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, filename in REPORT_FILES.items():

        target = REPORT_DIR / filename

        reports[name].to_csv(
            target,
            index=False,
        )

        print(f"Saved: {target}")


def main() -> None:

    print("PHISHGUARD AI - EXTENDED V2 MODEL CANDIDATES")

    print(
        "Existing validation is diagnostic; "
        "test.csv is never accessed."
    )

    train, validation = load_datasets()
    parameters = load_best_parameters()

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(validation):,}"
    )

    reports = run_experiment(
        train,
        validation,
        parameters,
    )

    save_reports(reports)

    print("\nCANDIDATE COMPARISON")

    print(
        reports["comparison"].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nFULL-MODEL VALIDATION BY SOURCE")

    print(
        reports["validation_sources"][
            [
                "strategy",
                "evaluation_source",
                "evaluation_samples",
                "recall",
                "fpr",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nGuardrails are exploratory: "
        "worst-source FPR <= 10% and recall >= 80%."
    )

    print(
        "No candidate is automatically promoted. "
        "No model was saved or deployed."
    )

    print(
        "Source subsets can overlap; "
        "source-specific counts must not be summed."
    )


if __name__ == "__main__":
    main()