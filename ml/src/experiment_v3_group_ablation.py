"""PhishGuard AI: targeted group ablation of the six experimental V3 features.

Uses only the existing V3 training split; held-out sources remain domain-disjoint.
Compares source_class training and a fixed 0.50 decision threshold.
Exploratory diagnostic: source folds were previously used for model development.
No validation/test access, URLs in reports, or saved trained models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model_candidates import evaluate_scores, score_model
from evaluate_optimized_models import load_best_parameters
from experiment_features_v3 import load_aligned_training
from experiment_hyperparameters import create_model, create_source_folds
from experiment_source_bias_mitigation import calculate_weights
from experiment_training_strategies import SOURCES, belongs_to_source, verify_group_isolation
from extract_features_v2 import FEATURE_COLUMNS_V2
from extract_features_v3 import FEATURE_COLUMNS_EXTENDED_V3

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
THRESHOLD = 0.50

V3_GROUPS = {
    "auth": {"v3_hostname_auth_tokens", "v3_path_auth_tokens"},
    "path_shape": {
        "v3_avg_path_segment_length",
        "v3_max_path_segment_length",
        "v3_path_digit_ratio",
    },
    "query_entropy": {"v3_query_value_entropy"},
}


def build_configurations() -> dict[str, list[str]]:
    """Prespecify V2, V3 and three group-removal experiments."""
    v2 = list(FEATURE_COLUMNS_V2)
    v3 = list(FEATURE_COLUMNS_EXTENDED_V3)
    excluded = set().union(*V3_GROUPS.values())
    if (
        len(v2) != 31
        or len(v3) != 37
        or len(set(v3)) != len(v3)
        or v3[:len(v2)] != v2
        or excluded != set(v3) - set(v2)
        or sum(map(len, V3_GROUPS.values())) != len(excluded)
    ):
        raise ValueError("Unexpected V2/V3 feature definitions")
    configurations = {
        "v2_reference": v2,
        "v3_full": v3,
    }
    for group, names in V3_GROUPS.items():
        configurations[f"v3_without_{group}"] = [
            feature for feature in v3 if feature not in names
        ]
    if len({tuple(features) for features in configurations.values()}) != len(configurations):
        raise ValueError("Duplicate feature configurations")
    return configurations


def summarize(source_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every configuration to V2 on the exact same source folds."""
    configurations = build_configurations()
    expected = {(c, s) for c in configurations for s in SOURCES}
    actual = list(zip(source_rows["configuration"], source_rows["evaluation_source"]))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated source results")

    indexed = source_rows.set_index(["configuration", "evaluation_source"])
    compared = []
    for configuration in configurations:
        for source in SOURCES:
            current = indexed.loc[(configuration, source)]
            original = indexed.loc[("v2_reference", source)]
            compared.append({
                "configuration": configuration,
                "evaluation_source": source,
                "feature_count": int(current["feature_count"]),
                "f1": float(current["f1"]),
                "recall": float(current["recall"]),
                "fpr": float(current["fpr"]),
                "delta_f1_vs_v2": float(current["f1"] - original["f1"]),
                "delta_recall_vs_v2": float(current["recall"] - original["recall"]),
                "delta_fpr_vs_v2": float(current["fpr"] - original["fpr"]),
            })
    per_source = pd.DataFrame(compared)

    summary = (
        per_source.groupby("configuration", as_index=False, sort=False)
        .agg(
            feature_count=("feature_count", "first"),
            macro_f1=("f1", "mean"),
            macro_recall=("recall", "mean"),
            macro_fpr=("fpr", "mean"),
            worst_recall=("recall", "min"),
            worst_fpr=("fpr", "max"),
        )
    )
    baseline = summary.loc[summary["configuration"] == "v2_reference"].iloc[0]
    for metric in ("macro_f1", "macro_recall", "macro_fpr", "worst_recall", "worst_fpr"):
        summary[f"delta_{metric}_vs_v2"] = summary[metric] - baseline[metric]
    return per_source, summary


def main() -> None:
    print("PHISHGUARD AI - TARGETED V3 FEATURE GROUP ABLATION")
    print("Source-class weighting; training split only; test untouched.\n")
    frame = load_aligned_training()
    parameters = load_best_parameters()
    if "extended_v2" not in parameters:
        raise ValueError("Missing optimized Extended V2 parameters")

    configs = build_configurations()
    folds = create_source_folds(frame)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Incomplete or duplicate source folds")

    results = []
    for fold in folds:
        source = fold["source"]
        train = frame.iloc[fold["training_indices"]]
        held = frame.iloc[fold["validation_indices"]]
        verify_group_isolation(train, held)
        if belongs_to_source(train, source).any():
            raise ValueError(f"Source leakage: {source}")
        if set(train["label"].unique()) != {0, 1} or set(held["label"].unique()) != {0, 1}:
            raise ValueError(f"Both labels required: {source}")
        y_train = train["label"].to_numpy(dtype=int)
        y_held = held["label"].to_numpy(dtype=int)
        weights = calculate_weights(train, "source_class")

        for configuration, features in configs.items():
            model = create_model(parameters["extended_v2"])
            model.fit(
                train[features].to_numpy(dtype=np.float64),
                y_train,
                sample_weight=weights,
            )
            scores = score_model(model, held[features].to_numpy(dtype=np.float64))
            metrics = evaluate_scores(y_held, scores)
            results.append({
                "configuration": configuration,
                "evaluation_source": source,
                "strategy": "source_class",
                "feature_count": len(features),
                "evaluation_samples": len(held),
                "threshold": THRESHOLD,
                **metrics,
            })
            print(
                f"{source:<11} {configuration:<26} "
                f"F1={metrics['f1']:.4f} "
                f"Recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}"
            )

    source_results = pd.DataFrame(results)
    by_source, summary = summarize(source_results)

    # Aggregate reports only; no raw URL/domain/source membership columns.
    forbidden = {"url", "urls", "domain", "hostname", "group_key", "sources"}
    reports = {
        "v3_group_ablation_raw.csv": source_results,
        "v3_group_ablation_by_source.csv": by_source,
        "v3_group_ablation_summary.csv": summary,
    }
    for filename, report in reports.items():
        if report.empty or not forbidden.isdisjoint(report.columns):
            raise ValueError(f"Invalid or sensitive report: {filename}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, report in reports.items():
        report.to_csv(REPORT_DIR / filename, index=False)
        print(f"Saved: {REPORT_DIR / filename}")

    print("\nGROUP ABLATION MACRO SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGROUP ABLATION BY SOURCE")
    print(by_source.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nExploratory only. No model was saved or deployed.")


if __name__ == "__main__":
    main()
