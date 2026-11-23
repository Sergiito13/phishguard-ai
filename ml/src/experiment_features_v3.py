"""PhishGuard AI: build V3 splits and compare Extended V2 vs V3.

- Reuses the already generated V2 train/validation datasets.
- Preserves original rows, metadata, splits and V2 values.
- Does NOT read the reserved test split, save models, or contact the network.
- Evaluates ONLY on train-source-exclusion/domain-disjoint folds.
- Results are exploratory: these sources informed previous model decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model_candidates import evaluate_scores, score_model
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
from extract_features_v2 import FEATURE_COLUMNS_V2
from extract_features_v3 import (
    FEATURE_COLUMNS_EXTENDED_V3,
    FEATURE_COLUMNS_V3,
    transform_dataframe_v3,
)

BASE_DIR = Path(__file__).resolve().parents[1]
V2_DIR = BASE_DIR / "data" / "splits" / "v2"
V3_DIR = BASE_DIR / "data" / "splits" / "v3"
REPORT_DIR = BASE_DIR / "reports"
V3_TRAIN_FILE = V3_DIR / "train.csv"
CHUNK_SIZE = 10_000
STRATEGIES = ("uniform", "source_class")
META = ["url", "label", "sources", "group_key"]
REPORT_FILES = {
    "source_results": "v3_cross_source_results.csv",
    "paired": "v3_vs_v2_paired.csv",
    "summary": "v3_vs_v2_summary.csv",
}


def read_metadata(path: Path) -> pd.DataFrame:
    """Read only metadata needed to protect the existing split boundaries."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 dataset: {path}")
    frame = pd.read_csv(
        path,
        usecols=META,
        keep_default_na=False,
        dtype={"url": "string", "sources": "string", "group_key": "string"},
    )
    if frame.empty or set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Invalid labels or empty split: {path}")
    if frame["url"].isna().any() or frame["url"].eq("").any():
        raise ValueError(f"Missing URLs in: {path}")
    if frame["url"].duplicated().any():
        raise ValueError(f"Duplicate URLs in: {path}")
    if frame["group_key"].isna().any() or frame["group_key"].eq("").any():
        raise ValueError(f"Missing group keys in: {path}")
    return frame


def process_split(source: Path, destination: Path, expected_rows: int) -> int:
    """Append six V3 columns in chunks, replacing the destination atomically."""
    if source.resolve() == destination.resolve():
        raise ValueError("V2 source and V3 destination cannot be the same file")
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    count = 0
    try:
        for chunk in pd.read_csv(
            source,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            dtype={"url": "string", "sources": "string", "group_key": "string"},
        ):
            augmented = transform_dataframe_v3(chunk)
            if augmented.columns.duplicated().any():
                raise ValueError("Duplicate columns after V3 transformation")
            augmented.to_csv(
                temporary,
                mode="w" if count == 0 else "a",
                header=(count == 0),
                index=False,
            )
            count += len(augmented)
            print(f"  {source.name}: {count:,}/{expected_rows:,}")
        if count != expected_rows:
            raise ValueError(f"Unexpected row count: {count} != {expected_rows}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def generate_v3_datasets() -> dict[str, int]:
    """Preserve the V2 train/validation partitions; never generate a test file."""
    train_meta = read_metadata(V2_DIR / "train.csv")
    val_meta = read_metadata(V2_DIR / "validation.csv")
    verify_group_isolation(train_meta, val_meta)
    if not set(train_meta["url"]).isdisjoint(set(val_meta["url"])):
        raise ValueError("Train/validation URL leakage detected")
    sizes = {"train": len(train_meta), "validation": len(val_meta)}
    del train_meta, val_meta
    for split, expected in sizes.items():
        print(f"\nGenerating V3 {split} dataset...")
        process_split(V2_DIR / f"{split}.csv", V3_DIR / f"{split}.csv", expected)
    return sizes


def load_aligned_training() -> pd.DataFrame:
    """Validate V3 against the *original* previously checked V2 training data."""
    original = load_training_data()
    if not V3_TRAIN_FILE.is_file():
        raise FileNotFoundError(f"Missing generated V3 dataset: {V3_TRAIN_FILE}")
    requested = [*META, "is_https", *FEATURE_COLUMNS_EXTENDED_V3]
    frame = pd.read_csv(
        V3_TRAIN_FILE,
        usecols=requested,
        keep_default_na=False,
        dtype={"url": "string", "sources": "string", "group_key": "string"},
    )
    if len(original) != len(frame):
        raise ValueError("V2 and V3 train row counts differ")
    for column in META:
        if not original[column].reset_index(drop=True).equals(
            frame[column].reset_index(drop=True)
        ):
            raise ValueError(f"V2/V3 metadata misalignment: {column}")
    v2_old = original[FEATURE_COLUMNS_V2].to_numpy(dtype=np.float64)
    v2_new = frame[FEATURE_COLUMNS_V2].to_numpy(dtype=np.float64)
    if not np.allclose(v2_old, v2_new, rtol=1e-12, atol=1e-12):
        raise ValueError("V2 features changed during V3 generation")
    if not np.isfinite(
        frame[FEATURE_COLUMNS_EXTENDED_V3].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite V3 training features")
    if not frame["is_https"].isin([0, 1]).all():
        raise ValueError("Invalid HTTPS indicators")
    return frame


def compare_candidates(frame: pd.DataFrame, parameters: dict) -> dict[str, pd.DataFrame]:
    """Paired 4-source x 2-weighting x 2-feature-set experiment (16 fits)."""
    if "extended_v2" not in parameters:
        raise ValueError("Missing optimized Extended V2 parameters")
    if set(FEATURE_SETS["extended_v2"]) != set(FEATURE_COLUMNS_V2):
        raise ValueError("Unexpected baseline V2 feature definition")
    required = {"label", "sources", "group_key", "is_https", *FEATURE_COLUMNS_EXTENDED_V3}
    if required - set(frame.columns):
        raise ValueError(f"Missing evaluation columns: {sorted(required - set(frame.columns))}")
    if not np.isfinite(
        frame[FEATURE_COLUMNS_EXTENDED_V3].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Invalid numerical features")
    folds = create_source_folds(frame)
    if len(folds) != len(SOURCES) or {f["source"] for f in folds} != set(SOURCES):
        raise ValueError("Missing or duplicate source folds")
    rows = []
    configurations = (
        ("extended_v2", FEATURE_COLUMNS_V2),
        ("extended_v3", FEATURE_COLUMNS_EXTENDED_V3),
    )
    for fold in folds:
        source = fold["source"]
        fit = frame.iloc[fold["training_indices"]]
        held = frame.iloc[fold["validation_indices"]]
        verify_group_isolation(fit, held)
        if belongs_to_source(fit, source).any():
            raise ValueError(f"Held-out source leaked into training: {source}")
        if set(fit["label"].unique()) != {0, 1} or set(held["label"].unique()) != {0, 1}:
            raise ValueError(f"Both classes required: {source}")
        y_fit = fit["label"].to_numpy(dtype=int)
        y_held = held["label"].to_numpy(dtype=int)
        for strategy in STRATEGIES:
            weights = calculate_weights(fit, strategy)
            for name, features in configurations:
                model = create_model(parameters["extended_v2"])
                model.fit(
                    fit[features].to_numpy(dtype=np.float64),
                    y_fit,
                    sample_weight=weights,
                )
                scores = score_model(
                    model,
                    held[features].to_numpy(dtype=np.float64),
                )
                metrics = evaluate_scores(y_held, scores)
                rows.append({
                    "feature_set": name,
                    "strategy": strategy,
                    "evaluation_source": source,
                    "feature_count": len(features),
                    "training_samples": len(fit),
                    "evaluation_samples": len(held),
                    "threshold": 0.5,
                    **metrics,
                })
                print(
                    f"{source:<11} {strategy:<12} {name:<11} "
                    f"F1={metrics['f1']:.4f} "
                    f"Recall={metrics['recall']:.4f} FPR={metrics['fpr']:.4f}"
                )

    source_results = pd.DataFrame(rows)
    expected = {
        (name, strategy, source)
        for name, _ in configurations
        for strategy in STRATEGIES
        for source in SOURCES
    }
    actual = list(zip(
        source_results["feature_set"],
        source_results["strategy"],
        source_results["evaluation_source"],
    ))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate result combinations")

    pairs = []
    for strategy in STRATEGIES:
        for source in SOURCES:
            indexed = source_results.loc[
                (source_results["strategy"] == strategy)
                & (source_results["evaluation_source"] == source)
            ].set_index("feature_set")
            v2, v3 = indexed.loc["extended_v2"], indexed.loc["extended_v3"]
            pairs.append({
                "strategy": strategy,
                "evaluation_source": source,
                "evaluation_samples": int(v2["evaluation_samples"]),
                "v2_f1": float(v2["f1"]), "v3_f1": float(v3["f1"]),
                "delta_f1": float(v3["f1"] - v2["f1"]),
                "v2_recall": float(v2["recall"]), "v3_recall": float(v3["recall"]),
                "delta_recall": float(v3["recall"] - v2["recall"]),
                "v2_fpr": float(v2["fpr"]), "v3_fpr": float(v3["fpr"]),
                "delta_fpr": float(v3["fpr"] - v2["fpr"]),
            })
    paired = pd.DataFrame(pairs)
    summary_rows = []
    for strategy in STRATEGIES:
        for name, _ in configurations:
            subset = source_results.loc[
                (source_results["strategy"] == strategy)
                & (source_results["feature_set"] == name)
            ]
            summary_rows.append({
                "strategy": strategy,
                "feature_set": name,
                "macro_f1": float(subset["f1"].mean()),
                "macro_recall": float(subset["recall"].mean()),
                "macro_fpr": float(subset["fpr"].mean()),
                "worst_recall": float(subset["recall"].min()),
                "worst_fpr": float(subset["fpr"].max()),
            })
    summary = pd.DataFrame(summary_rows)
    return {"source_results": source_results, "paired": paired, "summary": summary}


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Missing experiment reports")
    sensitive = {"url", "urls", "sources", "group_key", "hostname", "domain"}
    for name, frame in reports.items():
        if frame.empty or not sensitive.isdisjoint(frame.columns):
            raise ValueError(f"Empty or sensitive report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        path = REPORT_DIR / filename
        reports[name].to_csv(path, index=False)
        print(f"Saved report: {path}")


def main() -> None:
    print("PHISHGUARD AI - V2 VS V3, LEAVE-ONE-SOURCE-OUT")
    print("Only V2 train/validation files are read; the reserved test is untouched.")
    sizes = generate_v3_datasets()
    print(f"Generated V3 splits: {sizes}")
    training = load_aligned_training()
    parameters = load_best_parameters()
    reports = compare_candidates(training, parameters)
    save_reports(reports)
    print("\nCOMPARISON BY SOURCE")
    print(reports["paired"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMACRO SUMMARY")
    print(reports["summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDiagnostic only: no model was saved, selected, or deployed.")


if __name__ == "__main__":
    main()
