"""PhishGuard AI: path-aware reweighting experiment on the V2 training split.

Compares the original 31-feature reference, the 27-feature path ablation,
and two bounded, prespecified reweightings of *legitimate* URLs with paths.
Uses domain-isolated leave-one-source-out folds drawn ONLY from training.
These sources have already informed development: this is exploratory work,
not an independent performance estimate or deployment-ready selection.
No URL/network access, reserved validation/test reads, or model exports.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model_candidates import score_model
from evaluate_optimized_models import load_best_parameters
from experiment_hyperparameters import (
    create_model,
    create_source_folds,
    load_training_data,
)
from experiment_path_feature_ablation import (
    FEATURES,
    binary_metrics,
    build_configurations,
    path_group_rows,
    validate_data,
)
from experiment_source_bias_mitigation import calculate_weights
from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)

FEATURE_SET = "extended_v2"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_FILES = {
    "by_source": "path_aware_training_by_source.csv",
    "by_path": "path_aware_training_by_path.csv",
    "macro": "path_aware_training_macro.csv",
    "weights": "path_aware_training_weight_diagnostics.csv",
}

# These alternatives are fixed before looking at their results.
# The reference configuration must reproduce the previous 31/27-feature runs.
VARIANTS = {
    "v2_full": ("v2_full", 1.0),
    "v2_27_reference": ("without_direct_path", 1.0),
    "v2_27_path_x2": ("without_direct_path", 2.0),
    "v2_27_path_x4": ("without_direct_path", 4.0),
}


def feature_variants() -> dict[str, tuple[list[str], float]]:
    """Resolve immutable experiment variants using existing V2 definitions."""
    original = build_configurations()
    result = {}
    for name, (feature_name, multiplier) in VARIANTS.items():
        result[name] = (list(original[feature_name]), float(multiplier))
    if len(result) != 4 or [len(v[0]) for v in result.values()] != [31, 27, 27, 27]:
        raise ValueError("Unexpected path-aware feature configurations")
    return result


def validate_inputs(frame: pd.DataFrame, parameters: dict) -> None:
    """Ensure the reweighting metadata exists and cannot be guessed silently."""
    validate_data(frame, parameters)
    if "is_https" not in frame or not frame["is_https"].isin((0, 1)).all():
        raise ValueError("Missing or invalid is_https metadata")
    if not np.isfinite(frame["v2_path_length"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite path lengths")
    if "v2_path_length" not in FEATURES:
        raise ValueError("Unexpected V2 feature definitions")
    feature_variants()


def path_aware_weights(frame: pd.DataFrame, multiplier: float) -> np.ndarray:
    """Reallocate *legitimate-class* weight to legitimate URLs with paths.

    Existing source_class weights are the starting point. Positive-label
    weights are unchanged. The *total* negative-class mass is preserved, so
    the experiment changes the mix of benign paths, not the global class mass.
    This is done on fitting rows only; held-out rows never influence weights.
    Multipliers are deliberately limited to 1/2/4 to avoid extreme boosts.
    """
    if multiplier not in (1.0, 2.0, 4.0):
        raise ValueError("Unsupported path-aware multiplier")
    required = {"label", "sources", "is_https", "v2_path_length"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing weighting fields: {sorted(required - set(frame.columns))}")
    if (not frame["label"].isin((0, 1)).all()
            or not frame["is_https"].isin((0, 1)).all()):
        raise ValueError("Invalid labels or is_https")
    path_lengths = frame["v2_path_length"].to_numpy(dtype=float)
    if not np.isfinite(path_lengths).all() or np.any(path_lengths < 0):
        raise ValueError("Invalid path lengths")

    base = np.asarray(calculate_weights(frame, "source_class"), dtype=float)
    if base.shape != (len(frame),) or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError("Invalid source_class weights")
    if multiplier == 1.0:
        return base.copy()  # Preserve exact baseline behavior.

    labels = frame["label"].to_numpy(dtype=int)
    negatives = labels == 0
    eligible = negatives & (path_lengths > 0)
    if not negatives.any():
        raise ValueError("No legitimate examples in training fold")
    if not eligible.any():
        return base.copy()  # No supported intervention in this fold.

    adjusted = base.copy()
    adjusted[eligible] *= multiplier
    old_mass = float(base[negatives].sum())
    new_mass = float(adjusted[negatives].sum())
    adjusted[negatives] *= old_mass / new_mass
    if not np.isfinite(adjusted).all() or (adjusted <= 0).any():
        raise ValueError("Invalid path-aware weights")
    if not np.isclose(adjusted[negatives].sum(), old_mass, rtol=1e-10):
        raise ValueError("Legitimate-class weight mass changed")
    return adjusted


def weight_diagnostics(
    frame: pd.DataFrame,
    weights: np.ndarray,
    source: str,
    variant: str,
    multiplier: float,
) -> dict:
    """Aggregate support and effective sample size; never export URLs."""
    y = frame["label"].to_numpy(dtype=int)
    p = frame["v2_path_length"].to_numpy(dtype=float)
    legit = y == 0
    has_path = legit & (p > 0)
    denom = float(weights[legit].sum())
    return {
        "evaluation_source": source,
        "configuration": variant,
        "path_multiplier": multiplier,
        "training_samples": len(frame),
        "legitimate_training_samples": int(legit.sum()),
        "legitimate_path_training_samples": int(has_path.sum()),
        "legitimate_path_sample_fraction": float(has_path.sum() / legit.sum()) if legit.any() else 0.0,
        "legitimate_path_weight_fraction": float(weights[has_path].sum() / denom) if denom else 0.0,
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def run_experiment(frame: pd.DataFrame, parameters: dict) -> dict[str, pd.DataFrame]:
    validate_inputs(frame, parameters)
    variants = feature_variants()
    folds = create_source_folds(frame)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Incomplete or duplicated source folds")

    source_rows, path_rows, weight_rows = [], [], []
    for fold in folds:
        source = fold["source"]
        fit = frame.iloc[fold["training_indices"]]
        held = frame.iloc[fold["validation_indices"]]
        verify_group_isolation(fit, held)
        if belongs_to_source(fit, source).any():
            raise ValueError(f"Source leakage in training: {source}")
        if set(fit["label"].unique()) != {0, 1} or set(held["label"].unique()) != {0, 1}:
            raise ValueError(f"Both labels required for {source}")

        y_fit = fit["label"].to_numpy(dtype=int)
        y_held = held["label"].to_numpy(dtype=int)
        for name, (features, multiplier) in variants.items():
            weights = path_aware_weights(fit, multiplier)
            weight_rows.append(weight_diagnostics(fit, weights, source, name, multiplier))
            model = create_model(parameters[FEATURE_SET])
            model.fit(fit[features].to_numpy(dtype=np.float64), y_fit, sample_weight=weights)
            scores = score_model(model, held[features].to_numpy(dtype=np.float64))
            metrics = binary_metrics(y_held, scores)
            source_rows.append({
                "configuration": name,
                "evaluation_source": source,
                "feature_count": len(features),
                "path_multiplier": multiplier,
                "training_samples": len(fit),
                "evaluation_samples": len(held),
                **metrics,
            })
            path_rows.extend(path_group_rows(source, name, held, scores))
            print(f"{source:<11} {name:<21} F1={metrics['f1']:.4f} "
                  f"Recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}")

    by_source = pd.DataFrame(source_rows)
    by_path = pd.DataFrame(path_rows)
    weights = pd.DataFrame(weight_rows)
    expected = {(source, name) for source in SOURCES for name in variants}
    actual = list(zip(by_source["evaluation_source"], by_source["configuration"]))
    if (len(actual) != len(expected) or set(actual) != expected
            or len(by_path) != 4 * len(expected)
            or len(weights) != len(expected)):
        raise ValueError("Missing or duplicated evaluation outputs")

    macro = by_source.groupby("configuration", as_index=False, sort=False).agg(
        feature_count=("feature_count", "first"),
        macro_f1=("f1", "mean"),
        macro_recall=("recall", "mean"),
        macro_fpr=("fpr", "mean"),
        worst_recall=("recall", "min"),
        worst_fpr=("fpr", "max"),
        macro_average_precision=("average_precision", "mean"),
    )
    reference = macro.loc[macro["configuration"] == "v2_27_reference"].iloc[0]
    for metric in ("macro_f1", "macro_recall", "macro_fpr", "worst_recall", "worst_fpr"):
        macro[f"delta_{metric}_vs_27"] = macro[metric] - reference[metric]
    return {"by_source": by_source, "by_path": by_path, "macro": macro, "weights": weights}


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Incomplete reports")
    forbidden = {"url", "urls", "group_key", "sources", "hostname", "domain", "full_url"}
    for name, report in reports.items():
        if report.empty or report.columns.duplicated().any() or not forbidden.isdisjoint(report.columns):
            raise ValueError(f"Empty or sensitive report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        path = REPORT_DIR / filename
        reports[name].to_csv(path, index=False)
        print(f"Saved: {path}")


def main() -> None:
    print("PHISHGUARD AI - PATH-AWARE SAMPLE WEIGHTING")
    print("Source_class + 27-feature candidate; grouped leave-one-source-out on TRAIN only.")
    print("Exploratory experiment; sources used in earlier model development.")
    print("Reserved validation/test untouched. No model saved/deployed.\n")
    frame = load_training_data()
    parameters = load_best_parameters()
    reports = run_experiment(frame, parameters)
    save_reports(reports)
    print("\nMACRO COMPARISON (FPR lower, F1/recall higher)")
    print(reports["macro"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nFOCAL PATH COMPARISON")
    p = reports["by_path"]
    focus = p.loc[
        ((p["evaluation_source"] == "url_phish") & (p["label"] == 0))
        | ((p["evaluation_source"] == "phishbd") & (p["label"] == 1))
    ]
    print(focus[["configuration", "evaluation_source", "path_group", "samples",
                 "metric_name", "metric_value", "errors"]].to_string(
                     index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nLEGITIMATE PATH COVERAGE AND WEIGHT SHARE")
    print(reports["weights"][["evaluation_source", "configuration",
                              "legitimate_path_training_samples",
                              "legitimate_path_sample_fraction",
                              "legitimate_path_weight_fraction",
                              "effective_sample_size"]].to_string(
                                  index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDiagnostic only. Do not select a production model from these repeated exploratory folds.")


if __name__ == "__main__":
    main()
