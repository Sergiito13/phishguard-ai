"""PhishGuard AI: audit class-conditional URL patterns across dataset sources.

Reads only the existing V2 TRAIN split and only aggregate feature values.
No raw URL, domain, or group-key columns are loaded or exported.
Never opens validation/test, trains a model, or changes FastAPI artifacts.

Descriptive diagnostics, not proof of label errors or causal explanations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "v2" / "train.csv"
REPORT_DIR = BASE_DIR / "reports"

SOURCES = ("legitphish", "phiusiil", "phishbd", "url_phish")
FEATURES = (
    "v2_url_length",
    "v2_hostname_length",
    "v2_path_length",
    "v2_path_depth",
    "v2_path_entropy",
    "v2_url_entropy",
    "v2_hostname_hyphen_count",
    "v2_slash_count",
    "v2_query_length",
)
REQUIRED_COLUMNS = ("sources", "label", *FEATURES)
REPORT_FILES = {
    "counts": "cross_source_audit_counts.csv",
    "patterns": "cross_source_audit_patterns.csv",
    "shifts": "cross_source_audit_shifts.csv",
}
FOCAL_PAIRS = (("url_phish", 0), ("phishbd", 1))


def read_training(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Load only feature values and non-identifying class/source metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing training dataset: {path}")
    try:
        frame = pd.read_csv(
            path,
            usecols=list(REQUIRED_COLUMNS),
            keep_default_na=False,
            dtype={"sources": "string"},
        )
    except ValueError as exc:
        raise ValueError(f"Missing required V2 training columns: {exc}") from exc
    validate_training(frame)
    return frame


def validate_training(frame: pd.DataFrame) -> None:
    """Validate source memberships, binary labels and finite features."""
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Training data must be nonempty with unique columns")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Training must have binary labels 0 and 1")
    for source_value in frame["sources"]:
        if not isinstance(source_value, str):
            raise ValueError("Missing or invalid source membership")
        members = source_value.split("|")
        if (
            not members
            or len(members) != len(set(members))
            or any(member not in SOURCES for member in members)
        ):
            raise ValueError(f"Invalid source membership: {source_value!r}")
    values = frame[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite V2 feature values")


def source_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    memberships = frame["sources"].str.split("|", regex=False)
    return {
        source: memberships.apply(lambda sources: source in sources)
        for source in SOURCES
    }


def describe_patterns(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Measure class-conditional source patterns and their shift vs other sources.

    URLs tagged to multiple sources contribute to each relevant source's
    counts. Comparison references EXCLUDE all URLs tagged to the focal source.
    """
    validate_training(frame)
    masks = source_masks(frame)
    labels = frame["label"]

    # Predeclare patterns; conditions deliberately use V2 features only.
    patterns = {
        "root_or_empty_path": frame["v2_path_length"] == 0,
        "has_nonempty_path": frame["v2_path_length"] > 0,
        "path_depth_3plus": frame["v2_path_depth"] >= 3,
        "path_entropy_2plus": frame["v2_path_entropy"] >= 2.0,
        "has_query": frame["v2_query_length"] > 0,
        "short_hostname_18minus": frame["v2_hostname_length"] <= 18,
        "url_length_40plus": frame["v2_url_length"] >= 40,
    }

    counts = []
    pattern_rows = []
    for source in SOURCES:
        group = masks[source]
        n_source = int(group.sum())
        for label in (0, 1):
            selection = group & (labels == label)
            n = int(selection.sum())
            if n == 0:
                raise ValueError(f"No label {label} examples for source {source}")
            counts.append({
                "source": source,
                "label": label,
                "samples": n,
                "source_total": n_source,
                "class_fraction_within_source": n / n_source,
            })
            for pattern, matched in patterns.items():
                count = int((selection & matched).sum())
                pattern_rows.append({
                    "source": source,
                    "label": label,
                    "pattern": pattern,
                    "samples": n,
                    "matching_samples": count,
                    "fraction": count / n,
                })

    shifts = []
    for source, label in FOCAL_PAIRS:
        target = masks[source] & (labels == label)
        reference = ~masks[source] & (labels == label)
        n_target, n_ref = int(target.sum()), int(reference.sum())
        if not n_target or not n_ref:
            raise ValueError(f"Missing comparison group: {source}, label {label}")
        for feature in FEATURES:
            target_values = frame.loc[target, feature].to_numpy(dtype=float)
            ref_values = frame.loc[reference, feature].to_numpy(dtype=float)
            mean_target, mean_ref = float(target_values.mean()), float(ref_values.mean())
            pooled_sd = float(np.sqrt(
                (np.var(target_values) + np.var(ref_values)) / 2.0
            ))
            standardized_difference = (
                (mean_target - mean_ref) / pooled_sd
                if pooled_sd > 0 else 0.0
            )
            shifts.append({
                "source": source,
                "label": label,
                "feature": feature,
                "source_samples": n_target,
                "other_sources_samples": n_ref,
                "source_mean": mean_target,
                "other_sources_mean": mean_ref,
                "source_median": float(np.median(target_values)),
                "other_sources_median": float(np.median(ref_values)),
                "standardized_difference": standardized_difference,
                "absolute_difference": abs(standardized_difference),
            })

    return {
        "counts": pd.DataFrame(counts),
        "patterns": pd.DataFrame(pattern_rows),
        "shifts": pd.DataFrame(shifts),
    }


def save_reports(reports: dict[str, pd.DataFrame]) -> None:
    """Write aggregate-only CSVs after checking *all* report schemas."""
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Missing audit reports")
    forbidden = {"url", "urls", "sources", "group_key", "domain", "hostname"}
    for name, report in reports.items():
        if report.empty or not forbidden.isdisjoint(report.columns):
            raise ValueError(f"Empty or sensitive report: {name}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        dest = REPORT_DIR / filename
        reports[name].to_csv(dest, index=False)
        print(f"Saved: {dest}")


def main() -> None:
    print("PHISHGUARD AI - CROSS-SOURCE PATTERN AUDIT")
    print("Only V2 TRAIN; no URLs exported, no test access, no model fitting.\n")
    frame = read_training(TRAIN_FILE)
    print(f"Loaded {len(frame):,} unique training rows")
    print("Membership can overlap across sources; source counts are not additive.\n")
    reports = describe_patterns(frame)
    save_reports(reports)

    print("\nSOURCE-CLASS COUNTS")
    print(reports["counts"].to_string(index=False))

    print("\nFOCAL PATTERN PREVALENCE")
    focal = reports["patterns"].loc[
        ((reports["patterns"]["source"] == "url_phish") &
         (reports["patterns"]["label"] == 0)) |
        ((reports["patterns"]["source"] == "phishbd") &
         (reports["patterns"]["label"] == 1))
    ]
    print(focal[["source", "label", "pattern", "fraction"]].to_string(
        index=False, float_format=lambda n: f"{n:.4f}"
    ))

    print("\nTOP SOURCE SHIFTS (descriptive, not causal)")
    for source, label in FOCAL_PAIRS:
        subset = reports["shifts"].loc[
            (reports["shifts"]["source"] == source) &
            (reports["shifts"]["label"] == label)
        ].sort_values("absolute_difference", ascending=False).head(7)
        print(f"\n{source} / label={label}")
        print(subset[["feature", "source_mean", "other_sources_mean",
                      "standardized_difference"]].to_string(
            index=False, float_format=lambda n: f"{n:.3f}"
        ))
    print("\nDescriptive audit only; suspicious patterns do not prove mislabeling.")


if __name__ == "__main__":
    main()
