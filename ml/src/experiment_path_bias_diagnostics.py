"""PhishGuard AI: diagnose prediction errors by URL path complexity.

Exploratory, descriptive analysis. Reads ONLY the original V2 training split.
For each dataset source, fit a fresh Extended V2 source_class model on the
other sources with registered-domain isolation, and score the held-out source.

Never opens the validation/test split, exports individual URLs/domains,
visits URLs, or saves/overwrites model artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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
HIGH_FP_SCORE = 0.80
LOW_FN_SCORE = 0.20
NEAR_THRESHOLD = 0.10
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_FILES = {
    "by_source": "path_bias_by_source.csv",
    "contrasts": "path_bias_contrasts.csv",
    "macro": "path_bias_macro.csv",
}

# "all", "root" and "nonempty" are directly comparable. The two depth
# groups partition "nonempty". Empty groups are omitted from by_source.
GROUPS = ("all", "root", "nonempty", "shallow_1_2", "deep_3plus")


def validate_input(frame: pd.DataFrame, parameters: dict) -> None:
    """Validate features, labels and membership metadata before model fitting."""
    if FEATURE_SET not in parameters:
        raise ValueError("Missing optimized Extended V2 parameters")
    needed = {*FEATURES, "label", "sources", "group_key", "is_https"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.empty or frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Training requires valid binary labels")
    if frame["group_key"].isna().any() or frame["group_key"].astype(str).eq("").any():
        raise ValueError("Missing domain group keys")
    if frame["sources"].isna().any():
        raise ValueError("Missing source membership")
    allowed = set(SOURCES)
    for value in frame["sources"].unique():
        if not isinstance(value, str):
            raise ValueError("Invalid source membership")
        parts = value.split("|")
        if not parts or len(set(parts)) != len(parts) or not set(parts).issubset(allowed):
            raise ValueError("Invalid source membership")
    if not frame["is_https"].isin([0, 1]).all():
        raise ValueError("Invalid HTTPS indicator")
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values")
    if (
        (frame["v2_path_length"] < 0).any()
        or (frame["v2_path_depth"] < 0).any()
    ):
        raise ValueError("Negative path feature values")


def path_masks(lengths, depths) -> dict[str, np.ndarray]:
    """Partition URLs by path presence and depth, without examining raw URLs."""
    lengths = np.asarray(lengths, dtype=float)
    depths = np.asarray(depths, dtype=float)
    if lengths.ndim != 1 or depths.shape != lengths.shape:
        raise ValueError("Path features must be equal-length 1D arrays")
    if not np.isfinite(lengths).all() or not np.isfinite(depths).all():
        raise ValueError("Non-finite path features")
    if np.any(lengths < 0) or np.any(depths < 0):
        raise ValueError("Negative path features")

    present = lengths > 0
    return {
        "all": np.ones(len(lengths), dtype=bool),
        "root": ~present,
        "nonempty": present,
        "shallow_1_2": present & (depths < 3),
        "deep_3plus": present & (depths >= 3),
    }


def summarize_predictions(
    source: str,
    labels,
    scores,
    lengths,
    depths,
) -> pd.DataFrame:
    """Return counts and conditional FPR/recall for each path group/class.

    For label=0, metric_value is FPR (lower is better).
    For label=1, metric_value is recall (higher is better).
    """
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=float)
    if (
        y.ndim != 1 or p.ndim != 1 or y.shape != p.shape
        or len(y) == 0 or not set(np.unique(y)).issubset({0, 1})
    ):
        raise ValueError("Labels and scores must be aligned binary arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Scores must be probabilities in [0, 1]")
    groups = path_masks(lengths, depths)
    if len(groups["all"]) != len(y):
        raise ValueError("Path features and labels are not aligned")

    rows = []
    for label in (0, 1):
        for group_name in GROUPS:
            mask = (y == label) & groups[group_name]
            n = int(mask.sum())
            if n == 0:
                continue
            selected = p[mask]
            predicted_phishing = selected >= THRESHOLD
            errors = (~predicted_phishing) if label == 1 else predicted_phishing
            error_count = int(errors.sum())
            # For phish, high-confidence errors are scores <=0.20.
            wrong_scores = selected[errors]
            high_wrong = int(np.sum(
                wrong_scores <= LOW_FN_SCORE if label == 1
                else wrong_scores >= HIGH_FP_SCORE
            ))
            near_wrong = int(np.sum(np.abs(wrong_scores - THRESHOLD) <= NEAR_THRESHOLD))
            metric = float(np.mean(predicted_phishing))
            rows.append({
                "evaluation_source": source,
                "label": label,
                "path_group": group_name,
                "samples": n,
                "metric_name": "recall" if label == 1 else "fpr",
                "metric_value": metric,
                "errors": error_count,
                "error_rate": error_count / n,
                "median_phishing_score": float(np.median(selected)),
                "median_error_score": float(np.median(wrong_scores)) if error_count else np.nan,
                "high_confidence_errors": high_wrong,
                "high_confidence_error_fraction": high_wrong / error_count if error_count else 0.0,
                "near_threshold_errors": near_wrong,
                "near_threshold_error_fraction": near_wrong / error_count if error_count else 0.0,
            })
    return pd.DataFrame(rows)


def build_reports(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build path-presence contrasts and macro averages from source details."""
    if rows.empty or rows[["evaluation_source", "label", "path_group"]].duplicated().any():
        raise ValueError("Empty or duplicated path-group rows")

    contrast_rows = []
    for source in SOURCES:
        for label in (0, 1):
            section = rows.loc[
                (rows["evaluation_source"] == source) & (rows["label"] == label)
            ]
            parts = {r["path_group"]: r for _, r in section.iterrows()}
            if "all" not in parts:
                raise ValueError(f"Missing whole-class result: {source}, {label}")
            root = parts.get("root")
            nonempty = parts.get("nonempty")
            # The split is exhaustive. If a group is empty, no within-fold
            # contrast is possible; NaN is intentional (not a zero error rate).
            if root is not None and nonempty is not None:
                if int(root["samples"]) + int(nonempty["samples"]) != int(parts["all"]["samples"]):
                    raise ValueError("Path partition accounting mismatch")
                delta = float(nonempty["metric_value"] - root["metric_value"])
            else:
                delta = np.nan
            contrast_rows.append({
                "evaluation_source": source,
                "label": label,
                "metric_name": "recall" if label == 1 else "fpr",
                "all_samples": int(parts["all"]["samples"]),
                "all_rate": float(parts["all"]["metric_value"]),
                "root_samples": int(root["samples"]) if root is not None else 0,
                "nonempty_samples": int(nonempty["samples"]) if nonempty is not None else 0,
                "root_rate": float(root["metric_value"]) if root is not None else np.nan,
                "nonempty_rate": float(nonempty["metric_value"]) if nonempty is not None else np.nan,
                "nonempty_minus_root": delta,
            })

    contrasts = pd.DataFrame(contrast_rows)
    macro = contrasts.groupby(["label", "metric_name"], as_index=False).agg(
        sources_with_both_groups=("nonempty_minus_root", "count"),
        macro_all_rate=("all_rate", "mean"),
        macro_root_rate=("root_rate", "mean"),
        macro_nonempty_rate=("nonempty_rate", "mean"),
        macro_nonempty_minus_root=("nonempty_minus_root", "mean"),
    )
    return {"by_source": rows, "contrasts": contrasts, "macro": macro}


def run_experiment(frame: pd.DataFrame, parameters: dict) -> dict[str, pd.DataFrame]:
    """Exactly four fresh, source-excluded, domain-disjoint model fits."""
    validate_input(frame, parameters)
    folds = create_source_folds(frame)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Incomplete or duplicated source folds")

    rows = []
    for fold in folds:
        source = fold["source"]
        fit = frame.iloc[fold["training_indices"]]
        held = frame.iloc[fold["validation_indices"]]
        verify_group_isolation(fit, held)
        if belongs_to_source(fit, source).any():
            raise ValueError(f"Source leakage: {source}")
        if set(fit["label"].unique()) != {0, 1} or set(held["label"].unique()) != {0, 1}:
            raise ValueError(f"Missing class in source fold: {source}")

        model = create_model(parameters[FEATURE_SET])
        sample_weight = calculate_weights(fit, "source_class")
        model.fit(
            fit[FEATURES].to_numpy(dtype=np.float64),
            fit["label"].to_numpy(dtype=int),
            sample_weight=sample_weight,
        )
        scores = score_model(model, held[FEATURES].to_numpy(dtype=np.float64))
        fold_results = summarize_predictions(
            source,
            held["label"].to_numpy(dtype=int),
            scores,
            held["v2_path_length"].to_numpy(dtype=float),
            held["v2_path_depth"].to_numpy(dtype=float),
        )
        rows.append(fold_results)
        print(f"  {source:<11} {len(fit):>7,} fit / {len(held):>7,} held out")

    return build_reports(pd.concat(rows, ignore_index=True))


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    """Write only aggregate data, never URLs or domain/group identifiers."""
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Incomplete reports")
    private = {"url", "urls", "domain", "hostname", "sources", "group_key"}
    for name, frame in reports.items():
        if frame.empty or not private.isdisjoint(frame.columns):
            raise ValueError(f"Empty or sensitive report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        target = REPORT_DIR / filename
        reports[name].to_csv(target, index=False)
        print(f"Saved: {target}")


def main() -> None:
    print("PHISHGUARD AI - PATH BIAS DIAGNOSTICS")
    print("Extended V2 / source_class / threshold=0.50")
    print("Training-only source-held-out folds; reserved validation/test untouched.")
    print("Observational diagnostic: these sources informed earlier development.\n")

    frame = load_training_data()
    parameters = load_best_parameters()
    print(f"Loaded {len(frame):,} training URLs")
    reports = run_experiment(frame, parameters)
    save_reports(reports)

    print("\nFOCAL PATH CONTRASTS (FPR lower is better, recall higher is better)")
    focal = reports["contrasts"].loc[
        ((reports["contrasts"]["evaluation_source"] == "url_phish")
         & (reports["contrasts"]["label"] == 0))
        | ((reports["contrasts"]["evaluation_source"] == "phishbd")
           & (reports["contrasts"]["label"] == 1))
    ]
    print(focal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFOCAL PATH-GROUP DETAIL")
    detail = reports["by_source"]
    detail = detail.loc[
        ((detail["evaluation_source"] == "url_phish") & (detail["label"] == 0))
        | ((detail["evaluation_source"] == "phishbd") & (detail["label"] == 1))
    ]
    print(detail[[
        "evaluation_source", "label", "path_group", "samples",
        "metric_name", "metric_value", "errors", "median_phishing_score",
        "high_confidence_error_fraction",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nExploratory only: correlation does not prove causal path bias.")
    print("No URL was exported or visited. No model was saved or deployed.")


if __name__ == "__main__":
    main()
