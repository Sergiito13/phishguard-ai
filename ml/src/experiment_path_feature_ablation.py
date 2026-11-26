"""PhishGuard AI: controlled V2 path-feature ablation across held-out sources.

Train-only, domain-disjoint leave-one-source-out evaluation. Reuses the
optimized Extended V2 hyperparameters and the source_class weighting policy.
The same sources previously informed model development; results are exploratory.
Does not read the validation/test splits, save models, visit URLs, or export
row-level identifiers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_model_candidates import score_model
from evaluate_optimized_models import load_best_parameters
from experiment_hyperparameters import (
    FEATURE_SETS,
    create_model,
    create_source_folds,
    load_training_data,
)
from experiment_source_bias_mitigation import calculate_weights
from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)

FEATURE_SET = "extended_v2"
FEATURES = list(FEATURE_SETS[FEATURE_SET])
THRESHOLD = 0.50
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_FILES = {
    "by_source": "path_feature_ablation_by_source.csv",
    "by_path": "path_feature_ablation_by_path.csv",
    "macro": "path_feature_ablation_macro.csv",
}

# Prespecified configurations. Direct path information and URL-wide proxies
# are tested separately. Removing these features does NOT remove all indirect
# evidence about the path from other V2 counts and ratios.
DIRECT_PATH = (
    "v2_path_length",
    "v2_path_depth",
    "v2_path_entropy",
    "v2_slash_count",
)
URL_WIDE_PROXIES = (
    "v2_url_length",
    "v2_url_entropy",
)
AUTHORITY_ONLY = (
    "v2_hostname_length",
    "v2_registered_domain_length",
    "v2_hostname_entropy",
    "v2_domain_label_entropy",
    "v2_subdomain_count",
    "v2_hostname_digit_ratio",
    "v2_hostname_hyphen_count",
    "v2_domain_token_count",
    "v2_has_ip",
    "v2_has_punycode",
    "v2_has_explicit_port",
    "v2_has_userinfo",
)


def build_configurations() -> dict[str, list[str]]:
    """Return four unique, predefined V2 feature subsets."""
    base = list(FEATURES)
    if not base or len(base) != len(set(base)):
        raise ValueError("Empty or duplicated V2 features")
    needed = set(DIRECT_PATH) | set(URL_WIDE_PROXIES) | set(AUTHORITY_ONLY)
    if not needed.issubset(base):
        raise ValueError(f"Missing expected V2 features: {sorted(needed - set(base))}")
    no_path = [f for f in base if f not in DIRECT_PATH]
    no_path_proxies = [
        f for f in base if f not in set(DIRECT_PATH) | set(URL_WIDE_PROXIES)
    ]
    configs = {
        "v2_full": base,
        "without_direct_path": no_path,
        "without_direct_path_and_url_proxies": no_path_proxies,
        "authority_only": [f for f in base if f in AUTHORITY_ONLY],
    }
    if len({tuple(v) for v in configs.values()}) != 4:
        raise ValueError("Duplicate ablation configurations")
    if len(configs["authority_only"]) != len(AUTHORITY_ONLY):
        raise ValueError("Authority-only feature mismatch")
    return configs


def validate_data(frame: pd.DataFrame, parameters: dict) -> None:
    configs = build_configurations()
    if FEATURE_SET not in parameters:
        raise ValueError("Missing optimized Extended V2 parameters")
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Empty dataset or duplicate columns")
    required = {"label", "sources", "group_key", "v2_path_length", *FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Both binary labels required")
    if frame["group_key"].isna().any() or frame["group_key"].astype(str).eq("").any():
        raise ValueError("Invalid domain groups")
    allowed = set(SOURCES)
    for membership in frame["sources"].unique():
        if not isinstance(membership, str):
            raise ValueError("Invalid source membership")
        parts = membership.split("|")
        if not parts or len(parts) != len(set(parts)) or not set(parts).issubset(allowed):
            raise ValueError("Invalid source membership")
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (frame["v2_path_length"] < 0).any():
        raise ValueError("Non-finite or negative path features")
    if len(configs) != 4:
        raise ValueError("Invalid configurations")


def binary_metrics(labels, scores, threshold: float = THRESHOLD) -> dict:
    """Summarize a whole source including both labels."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if (
        labels.ndim != 1 or scores.shape != labels.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Invalid labels or prediction scores")
    pred = scores >= threshold
    tp = int(((labels == 1) & pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def path_group_rows(source: str, variant: str, held: pd.DataFrame, scores) -> list[dict]:
    """Calculate conditional FPR/recall for root versus nonempty paths."""
    labels = held["label"].to_numpy(dtype=int)
    path_length = held["v2_path_length"].to_numpy(dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (len(scores) != len(labels) or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid held-out scores")
    pred = scores >= THRESHOLD
    rows = []
    for label, metric in ((0, "fpr"), (1, "recall")):
        for group, group_mask in (("root", path_length == 0), ("nonempty", path_length > 0)):
            mask = (labels == label) & group_mask
            n = int(mask.sum())
            positives = int((pred & mask).sum())
            errors = positives if label == 0 else n - positives
            rows.append({
                "configuration": variant,
                "evaluation_source": source,
                "label": label,
                "path_group": group,
                "samples": n,
                "metric_name": metric,
                "metric_value": positives / n if n else np.nan,
                "errors": errors,
                "high_confidence_errors": int(((scores >= 0.80) & mask & pred).sum())
                if label == 0 else int(((scores <= 0.20) & mask & ~pred).sum()),
            })
    return rows


def run_experiment(frame: pd.DataFrame, parameters: dict) -> dict[str, pd.DataFrame]:
    validate_data(frame, parameters)
    configs = build_configurations()
    folds = create_source_folds(frame)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Invalid source-held-out folds")
    source_rows, path_rows = [], []
    for source_fold in folds:
        source = source_fold["source"]
        fit = frame.iloc[source_fold["training_indices"]]
        held = frame.iloc[source_fold["validation_indices"]]
        verify_group_isolation(fit, held)
        if belongs_to_source(fit, source).any():
            raise ValueError(f"Source leakage: {source}")
        if set(fit["label"].unique()) != {0, 1} or set(held["label"].unique()) != {0, 1}:
            raise ValueError(f"Both labels required for {source}")
        weights = calculate_weights(fit, "source_class")
        labels_train = fit["label"].to_numpy(dtype=int)
        labels_held = held["label"].to_numpy(dtype=int)
        for variant, features in configs.items():
            model = create_model(parameters[FEATURE_SET])
            model.fit(
                fit[features].to_numpy(dtype=np.float64),
                labels_train,
                sample_weight=weights,
            )
            scores = score_model(model, held[features].to_numpy(dtype=np.float64))
            metrics = binary_metrics(labels_held, scores)
            source_rows.append({
                "configuration": variant,
                "evaluation_source": source,
                "feature_count": len(features),
                "training_samples": len(fit),
                "evaluation_samples": len(held),
                **metrics,
            })
            path_rows.extend(path_group_rows(source, variant, held, scores))
            print(f"{source:<11} {variant:<37} F1={metrics['f1']:.4f} "
                  f"Recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}")
    source_results = pd.DataFrame(source_rows)
    path_results = pd.DataFrame(path_rows)
    expected = {(source, variant) for source in SOURCES for variant in configs}
    if len(source_results) != len(expected) or set(zip(
        source_results["evaluation_source"], source_results["configuration"]
    )) != expected or len(path_results) != len(expected) * 4:
        raise ValueError("Missing or duplicate source/path evaluations")
    summary = source_results.groupby("configuration", as_index=False, sort=False).agg(
        feature_count=("feature_count", "first"),
        macro_f1=("f1", "mean"),
        macro_recall=("recall", "mean"),
        macro_fpr=("fpr", "mean"),
        worst_recall=("recall", "min"),
        worst_fpr=("fpr", "max"),
        macro_average_precision=("average_precision", "mean"),
    )
    base = summary.loc[summary["configuration"] == "v2_full"].iloc[0]
    for metric in ("macro_f1", "macro_recall", "macro_fpr", "worst_recall", "worst_fpr"):
        summary[f"delta_{metric}"] = summary[metric] - base[metric]
    return {"by_source": source_results, "by_path": path_results, "macro": summary}


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Incomplete reports")
    private_columns = {"url", "urls", "group_key", "sources", "hostname", "domain"}
    for name, frame in reports.items():
        if frame.empty or not private_columns.isdisjoint(frame.columns):
            raise ValueError(f"Empty or sensitive report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in REPORT_FILES.items():
        path = REPORT_DIR / filename
        reports[key].to_csv(path, index=False)
        print(f"Saved: {path}")


def main() -> None:
    print("PHISHGUARD AI - V2 PATH FEATURE ABLATION")
    print("Source-class weighted, train-only, grouped LOSO; reserved test untouched.")
    print("Existing source folds are exploratory and NOT independent validation.\n")
    data = load_training_data()
    parameters = load_best_parameters()
    reports = run_experiment(data, parameters)
    save_reports(reports)
    print("\nMACRO COMPARISON (FPR lower; recall/F1 higher)")
    print(reports["macro"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nFOCAL PATH-SEGMENT COMPARISON")
    focal = reports["by_path"].loc[
        ((reports["by_path"]["evaluation_source"] == "url_phish") &
         (reports["by_path"]["label"] == 0)) |
        ((reports["by_path"]["evaluation_source"] == "phishbd") &
         (reports["by_path"]["label"] == 1))
    ]
    print(focal[["configuration", "evaluation_source", "path_group", "samples",
                 "metric_name", "metric_value", "errors"]].to_string(
                     index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDiagnostic only: no models saved or deployed; no URLs exported.")


if __name__ == "__main__":
    main()
