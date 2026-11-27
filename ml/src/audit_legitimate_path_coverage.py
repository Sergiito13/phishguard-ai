"""PhishGuard AI: legitimate URL path coverage across training sources.

Reads only V2 TRAIN data (minimal metadata and structural features).
Identifies source membership, domain-group diversity, and the precise
training coverage after each existing domain-isolated source holdout.

No raw URLs or group keys are exported; no models are fitted, no URLs are
visited, and validation/test files are NOT read. Descriptive only: raw-to-
processed attrition cannot be inferred from this training-only audit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hyperparameters import create_source_folds
from experiment_training_strategies import SOURCES, belongs_to_source, verify_group_isolation

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "v2" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"
REQUIRED_COLUMNS = (
    "sources", "label", "group_key", "v2_path_length", "v2_path_depth", "v2_query_length"
)
REPORT_FILES = {
    "by_source": "legitimate_path_source_coverage.csv",
    "by_fold": "legitimate_path_fold_coverage.csv",
    "provenance": "legitimate_path_fold_provenance.csv",
}
SENSITIVE_COLUMNS = {
    "url", "urls", "hostname", "domain", "registered_domain",
    "group_key", "raw_url", "normalized_url", "full_url", "sources",
}


def load_training(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Load only the six columns necessary for this training-only audit."""
    if not path.is_file():
        raise FileNotFoundError(f"V2 training split not found: {path}")
    try:
        frame = pd.read_csv(
            path,
            usecols=list(REQUIRED_COLUMNS),
            keep_default_na=False,
            dtype={"sources": "string", "group_key": "string"},
        )
    except ValueError as exc:
        raise ValueError("Missing expected V2 training columns") from exc
    validate_training(frame)
    return frame


def validate_training(frame: pd.DataFrame) -> None:
    """Guard against missing columns and malformed source/domain metadata."""
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Empty training data or duplicated columns")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Training data must contain both binary labels")
    groups = frame["group_key"]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Invalid domain group key")
    allowed = set(SOURCES)
    for value in frame["sources"].unique():
        if not isinstance(value, str):
            raise ValueError("Invalid source membership")
        members = value.split("|")
        if not members or len(members) != len(set(members)) or not set(members).issubset(allowed):
            raise ValueError("Invalid source membership")
    values = frame[["v2_path_length", "v2_path_depth", "v2_query_length"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid structural URL feature")


def concentration(groups: pd.Series) -> tuple[int, float]:
    """Number of domain groups and share of rows held by five largest groups."""
    if groups.empty:
        return 0, np.nan
    counts = groups.value_counts(dropna=False)
    return len(counts), float(counts.iloc[:5].sum() / len(groups))


def build_reports(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Describe source-level and LOSO-fold coverage of benign URL paths."""
    validate_training(frame)
    all_sources = tuple(SOURCES)
    memberships = frame["sources"].str.split("|", regex=False)
    path = frame["v2_path_length"].to_numpy(dtype=float) > 0
    label = frame["label"].to_numpy(dtype=int)
    multi = memberships.apply(lambda m: len(m) > 1).to_numpy(dtype=bool)

    source_rows = []
    for source in all_sources:
        source_mask = belongs_to_source(frame, source).to_numpy(dtype=bool)
        for target_label in (0, 1):
            mask = source_mask & (label == target_label)
            if not mask.any():
                raise ValueError(f"Missing label for source: {source}")
            nonempty = mask & path
            domain_count, top5 = concentration(frame.loc[nonempty, "group_key"])
            source_rows.append({
                "source": source,
                "label": target_label,
                "samples": int(mask.sum()),
                "domain_groups": int(frame.loc[mask, "group_key"].nunique()),
                "nonempty_path_rows": int(nonempty.sum()),
                "nonempty_path_fraction": float(nonempty.sum() / mask.sum()),
                "nonempty_path_domain_groups": domain_count,
                "nonempty_top5_domain_share": top5,
                "deep_path_rows": int((mask & (frame["v2_path_depth"].to_numpy() >= 3)).sum()),
                "query_rows": int((mask & (frame["v2_query_length"].to_numpy() > 0)).sum()),
                "multi_source_rows": int((mask & multi).sum()),
            })

    fold_rows, provenance_rows = [], []
    folds = create_source_folds(frame)
    if len(folds) != len(all_sources) or {fold["source"] for fold in folds} != set(all_sources):
        raise ValueError("Incomplete held-out source folds")
    for fold in folds:
        excluded = fold["source"]
        train = frame.iloc[fold["training_indices"]]
        held = frame.iloc[fold["validation_indices"]]
        verify_group_isolation(train, held)
        if belongs_to_source(train, excluded).any():
            raise ValueError("Source leakage")
        train_benign = train.loc[train["label"].eq(0)]
        held_benign = held.loc[held["label"].eq(0)]
        if train_benign.empty or held_benign.empty:
            raise ValueError("Missing benign examples in fold")
        train_path = train_benign.loc[train_benign["v2_path_length"].gt(0)]
        held_path = held_benign.loc[held_benign["v2_path_length"].gt(0)]
        path_domains, top5 = concentration(train_path["group_key"])
        fold_rows.append({
            "excluded_source": excluded,
            "training_samples": len(train),
            "held_out_samples": len(held),
            "training_benign_rows": len(train_benign),
            "training_benign_path_rows": len(train_path),
            "training_benign_path_fraction": len(train_path) / len(train_benign),
            "training_benign_path_domain_groups": path_domains,
            "training_benign_path_top5_domain_share": top5,
            "held_out_benign_rows": len(held_benign),
            "held_out_benign_path_rows": len(held_path),
            "held_out_benign_path_domain_groups": held_path["group_key"].nunique(),
        })

        # Attribute all remaining benign paths to every contributing source;
        # fractional mass ensures multi-source rows sum to exactly 1.
        for source in all_sources:
            if source == excluded:
                continue
            source_mask = belongs_to_source(train_path, source)
            rows = train_path.loc[source_mask]
            members_per_row = rows["sources"].str.count(r"\|") + 1
            provenance_rows.append({
                "excluded_source": excluded,
                "contributing_source": source,
                "membership_rows": len(rows),
                "fractional_rows": float((1.0 / members_per_row).sum()),
                "domain_groups": rows["group_key"].nunique(),
            })
        total_mass = sum(
            entry["fractional_rows"]
            for entry in provenance_rows if entry["excluded_source"] == excluded
        )
        if not np.isclose(total_mass, len(train_path), rtol=0, atol=1e-8):
            raise ValueError("Source provenance mass mismatch")

    return {
        "by_source": pd.DataFrame(source_rows),
        "by_fold": pd.DataFrame(fold_rows),
        "provenance": pd.DataFrame(provenance_rows),
    }


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    """Reject private data before writing any file; export aggregates only."""
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Unexpected audit report keys")
    for name, result in reports.items():
        if result.empty or not SENSITIVE_COLUMNS.isdisjoint(result.columns):
            raise ValueError(f"Empty or sensitive audit report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        destination = REPORT_DIR / filename
        reports[name].to_csv(destination, index=False)
        print(f"Saved: {destination}")


def main() -> None:
    print("PHISHGUARD AI - LEGITIMATE URL PATH COVERAGE AUDIT")
    print("Only V2 TRAIN; no model, URL fetching, validation or test access.")
    print("Rows may belong to multiple sources; source counts overlap.\n")
    frame = load_training()
    print(f"Loaded {len(frame):,} unique training records")
    reports = build_reports(frame)
    save_reports(reports)
    print("\nSOURCE-CLASS COVERAGE")
    print(reports["by_source"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDOMAIN-ISOLATED FOLD COVERAGE")
    print(reports["by_fold"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSOURCE CONTRIBUTIONS TO BENIGN PATH TRAINING")
    print(reports["provenance"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation: training split only; this cannot establish raw-to-processed attrition")
    print("or prove that any dataset labels are incorrect. No URLs or domains exported.")


if __name__ == "__main__":
    main()
