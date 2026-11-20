"""PhishGuard AI: train-only, cross-source sample-weighting comparison.

Four policies x two optimized models x four source-held-out folds = 32 fits.
Source memberships are fractional: a shared URL contributes 1/k to each
of its k source memberships when building weighting cells.

This is exploratory. It never reads reserved validation/test splits, publishes
individual URLs, or overwrites trained models. Prior tuning used these sources.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_optimized_models import load_best_parameters
from experiment_hyperparameters import (
    FEATURE_SETS, create_model, create_source_folds, load_training_data,
)
from experiment_threshold_optimization import threshold_metrics
from experiment_training_strategies import (
    SOURCES, belongs_to_source, verify_group_isolation,
)

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
MODELS = ("hybrid_v1_extended", "extended_v2")
STRATEGIES = ("uniform", "source", "source_class", "source_class_https")
THRESHOLD = 0.50
MIN_CELL_MASS = 100.0  # Avoid extreme weights for tiny cells.
MAX_RELATIVE_WEIGHT = 10.0


def source_memberships(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    """Validate and expand pipe-separated memberships; never pick one arbitrarily."""
    if "sources" not in frame:
        raise ValueError("Missing sources column")
    memberships = []
    allowed = set(SOURCES)
    for value in frame["sources"]:
        if not isinstance(value, str):
            raise ValueError("Invalid source membership")
        parts = tuple(value.split("|"))
        if not parts or len(parts) != len(set(parts)) or any(s not in allowed for s in parts):
            raise ValueError(f"Invalid source membership: {value!r}")
        memberships.append(parts)
    return memberships


def calculate_weights(frame: pd.DataFrame, strategy: str) -> np.ndarray | None:
    """Weight source, source/label, or source/label/HTTPS cells.

    Each URL contributes equal fractions across its source memberships.
    Weights are capped and renormalized; exact cell balance isn't guaranteed.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if frame.empty:
        raise ValueError("Cannot weight empty training data")
    if strategy == "uniform":
        return None
    for col in ("label", "is_https"):
        if col not in frame or not frame[col].isin([0, 1]).all():
            raise ValueError(f"Missing or invalid {col}")

    memberships = source_memberships(frame)
    labels = frame["label"].to_numpy(dtype=int)
    protocols = frame["is_https"].to_numpy(dtype=int)
    masses: defaultdict[tuple, float] = defaultdict(float)
    keys_per_row = []

    for members, label, https in zip(memberships, labels, protocols):
        if strategy == "source":
            keys = [(s,) for s in members]
        elif strategy == "source_class":
            keys = [(s, int(label)) for s in members]
        else:
            keys = [(s, int(label), int(https)) for s in members]
        fraction = 1.0 / len(keys)
        for key in keys:
            masses[key] += fraction
        keys_per_row.append(keys)

    weights = np.array([
        sum((1.0 / len(keys)) / max(masses[key], MIN_CELL_MASS) for key in keys)
        for keys in keys_per_row
    ], dtype=float)

    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid sample weights")
    weights /= weights.mean()
    weights = np.minimum(weights, MAX_RELATIVE_WEIGHT)
    weights /= weights.mean()
    return weights


def weight_diagnostics(weights, model_name, source, strategy, n):
    """Only aggregated diagnostics; ESS = (sum(w)^2) / sum(w^2)."""
    values = np.ones(n, dtype=float) if weights is None else weights
    return {
        "feature_set": model_name,
        "evaluation_source": source,
        "strategy": strategy,
        "training_samples": n,
        "min_weight": float(values.min()),
        "median_weight": float(np.median(values)),
        "max_weight": float(values.max()),
        "effective_sample_size": float(values.sum() ** 2 / np.square(values).sum()),
        "fraction_weight_gt_5": float(np.mean(values > 5)),
    }


def run_experiment(dataframe: pd.DataFrame, parameters: dict):
    """Train-only leave-one-source-out evaluation with domain isolation."""
    if not set(MODELS).issubset(parameters):
        raise ValueError("Missing optimized parameters")
    for field in ("label", "sources", "group_key", "is_https"):
        if field not in dataframe:
            raise ValueError(f"Missing training field: {field}")
    if not dataframe["label"].isin([0, 1]).all():
        raise ValueError("Invalid labels")

    folds = create_source_folds(dataframe)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Incomplete source folds")

    rows, diagnostics = [], []
    for model_name in MODELS:
        features = list(FEATURE_SETS[model_name])
        missing = set(features) - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        for fold in folds:
            source = fold["source"]
            train = dataframe.iloc[fold["training_indices"]]
            held = dataframe.iloc[fold["validation_indices"]]
            verify_group_isolation(train, held)
            if belongs_to_source(train, source).any():
                raise ValueError(f"Source leakage detected: {source}")
            if train["label"].nunique() != 2 or held["label"].nunique() != 2:
                raise ValueError(f"Both classes required: {source}")

            X_train = train[features].to_numpy(dtype=np.float64)
            y_train = train["label"].to_numpy(dtype=int)
            X_held = held[features].to_numpy(dtype=np.float64)
            y_held = held["label"].to_numpy(dtype=int)
            for strategy in STRATEGIES:
                weights = calculate_weights(train, strategy)
                model = create_model(parameters[model_name])
                model.fit(X_train, y_train, sample_weight=weights)
                probabilities = model.predict_proba(X_held)[:, 1]
                metrics = threshold_metrics(y_held, probabilities, THRESHOLD)
                rows.append({
                    "feature_set": model_name,
                    "evaluation_source": source,
                    "strategy": strategy,
                    "feature_count": len(features),
                    "training_samples": len(train),
                    "evaluation_samples": len(held),
                    "roc_auc": float(roc_auc_score(y_held, probabilities)),
                    "average_precision": float(average_precision_score(y_held, probabilities)),
                    **metrics,
                })
                diagnostics.append(weight_diagnostics(
                    weights, model_name, source, strategy, len(train)
                ))
                print(f"{model_name:19s} {source:11s} {strategy:19s} "
                      f"Recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}")
    results = pd.DataFrame(rows)
    if len(results) != len(MODELS) * len(SOURCES) * len(STRATEGIES):
        raise ValueError("Incomplete experiment")
    return results, pd.DataFrame(diagnostics)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    expected = {(m, s, p) for m in MODELS for s in SOURCES for p in STRATEGIES}
    observed = list(zip(results["feature_set"], results["evaluation_source"], results["strategy"]))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicate results")
    return results.groupby(["feature_set", "strategy"], as_index=False).agg(
        macro_f1=("f1", "mean"),
        macro_recall=("recall", "mean"),
        worst_recall=("recall", "min"),
        macro_fpr=("fpr", "mean"),
        worst_fpr=("fpr", "max"),
        macro_roc_auc=("roc_auc", "mean"),
        macro_average_precision=("average_precision", "mean"),
    ).sort_values(["feature_set", "strategy"]).reset_index(drop=True)


def compare_to_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["strategy"] == "uniform"].set_index("feature_set")
    if len(baseline) != len(MODELS) or set(baseline.index) != set(MODELS):
        raise ValueError("Missing baseline")
    comparison = summary.copy()
    for metric in ("macro_f1", "macro_recall", "worst_recall", "macro_fpr", "worst_fpr"):
        comparison[f"delta_{metric}"] = comparison[metric] - comparison["feature_set"].map(baseline[metric])
    return comparison


def save_reports(results, summary, comparison, diagnostics):
    outputs = {
        "source_bias_mitigation_results.csv": results,
        "source_bias_mitigation_summary.csv": summary,
        "source_bias_mitigation_comparison.csv": comparison,
        "source_bias_mitigation_weights.csv": diagnostics,
    }
    forbidden = {"url", "group_key", "sources", "hostname", "domain"}
    for filename, frame in outputs.items():
        if frame.empty or forbidden.intersection(frame.columns):
            raise ValueError(f"Invalid or sensitive report: {filename}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        path = REPORT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")


def main():
    print("PHISHGUARD AI - TRAIN-ONLY SOURCE BIAS MITIGATION")
    print("Fractional source memberships; rare-cell floor; weight cap.")
    data = load_training_data()
    params = load_best_parameters()
    results, diagnostics = run_experiment(data, params)
    summary = summarize_results(results)
    comparison = compare_to_baseline(summary)
    save_reports(results, summary, comparison, diagnostics)
    print("\nWEIGHTING COMPARISON")
    print(comparison[["feature_set", "strategy", "macro_f1", "macro_recall", "macro_fpr",
                      "worst_recall", "worst_fpr", "delta_macro_f1", "delta_macro_fpr"]].to_string(index=False))
    print("\nFOCAL SOURCE RESULTS")
    focal = results[((results["feature_set"] == "hybrid_v1_extended") &
                     (results["evaluation_source"] == "url_phish")) |
                    ((results["feature_set"] == "extended_v2") &
                     (results["evaluation_source"] == "phishbd"))]
    print(focal[["feature_set", "strategy", "evaluation_source", "recall", "fpr"]].to_string(index=False))
    print("\nExploratory only. Reserved validation/test not accessed; no model saved.")


if __name__ == "__main__":
    main()
