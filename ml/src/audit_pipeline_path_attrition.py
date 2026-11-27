"""PhishGuard AI: audit path coverage through the existing preprocessing pipeline.

Read-only audit. Examines original RAW files, prepared intermediate files, and
ONLY the existing TRAIN and V2 TRAIN splits. It NEVER opens validation/test,
fetches URLs, refits a model, rewrites data, or exports individual URLs/domains.

Important: unique URL counts change during deduplication; records excluded from
TRAIN may be deliberately held out, not lost. Dataset memberships can overlap.
All path lengths follow V2's convention: root '/' and empty path have length 0.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from clean_data import DATASETS, DOMAIN_PATTERN

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
SOURCES = tuple(DATASETS)
STAGES = (
    "raw_valid_labels",
    "cleaned",
    "deduplicated",
    "domain_groups",
    "eligible_for_split",
    "train",
    "v2_train",
)
REPORT_FILES = {
    "coverage": "pipeline_path_stage_coverage.csv",
    "transitions": "pipeline_path_transitions.csv",
    "excluded": "pipeline_path_exclusion_reasons.csv",
    "integrity": "pipeline_path_integrity.csv",
}
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def url_path_length(value: object) -> int | None:
    """Interpret URL paths like extract_features_v2, without any network I/O.

    Return None for missing, unsupported or unparsable URLs. A recoverable
    leading apostrophe is removed only for raw/cleaned domain-like input, as
    clean_data.py does. This does not claim that every URL is a live website.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    url = value.strip()
    if url.startswith("'") and re.match(DOMAIN_PATTERN, url[1:], re.I):
        url = url[1:]
    if SCHEME_PATTERN.match(url):
        if not url.lower().startswith(("http://", "https://")):
            return None
        candidate = url
    elif url.startswith("//"):
        candidate = url
    elif re.match(DOMAIN_PATTERN, url, re.I):
        candidate = "//" + url
    else:
        return None
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in ("", "http", "https") or not parsed.hostname:
            return None
        _ = parsed.port  # Detect invalid ports.
    except ValueError:
        return None
    return 0 if parsed.path == "/" else len(parsed.path)


def read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Load named columns only; refuse absent files or changed schemas."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing pipeline input: {path}")
    try:
        return pd.read_csv(path, usecols=list(columns), keep_default_na=False,
                           low_memory=False)
    except ValueError as exc:
        raise ValueError(f"Invalid columns in {path.name}: {exc}") from exc


def load_raw(base_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    original_counts: dict[str, int] = {}
    for source, config in DATASETS.items():
        path = base_dir / "data" / "raw" / config["file"]
        original = read_csv(path, (config["url_column"], config["label_column"]))
        original_counts[source] = len(original)
        label = pd.to_numeric(original[config["label_column"]], errors="coerce")
        if config["invert_labels"]:
            label = 1 - label
        valid = label.isin((0, 1))
        frames.append(pd.DataFrame({
            "source": source,
            "label": label.loc[valid].astype(int).to_numpy(),
            "url": original.loc[valid, config["url_column"]].astype("string").fillna("").to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True), original_counts


def expand_stage(frame: pd.DataFrame, *, single_source: bool) -> pd.DataFrame:
    """Expand pipe-separated source memberships, preserving unique URL rows."""
    expected = {"url", "label", "source" if single_source else "sources"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(expected - set(frame.columns))}")
    if not frame["label"].isin((0, 1)).all():
        raise ValueError("Invalid binary labels in pipeline stage")
    if single_source:
        result = frame.copy()
        if not result["source"].isin(SOURCES).all():
            raise ValueError("Unknown dataset source")
    else:
        memberships = frame["sources"].astype(str).str.split("|", regex=False)
        if not memberships.apply(
            lambda members: (len(members) == len(set(members))
                             and all(member in SOURCES for member in members))
        ).all():
            raise ValueError("Invalid or duplicated source membership")
        result = frame.drop(columns="sources").copy()
        result["source"] = memberships
        result = result.explode("source", ignore_index=True)
    result["url"] = result["url"].astype("string")
    if result["url"].isna().any():
        raise ValueError("Pipeline stage contains null URLs")
    return result


def with_path_lengths(frame: pd.DataFrame, *, v2: bool = False) -> pd.DataFrame:
    result = frame.copy()
    if v2:
        if "v2_path_length" not in result.columns:
            raise ValueError("Missing V2 path length")
        numbers = pd.to_numeric(result["v2_path_length"], errors="coerce")
        values = numbers.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid V2 path lengths")
        result["path_length"] = values
    else:
        result["path_length"] = [url_path_length(u) for u in result["url"]]
    return result


def coverage(stage: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for source in SOURCES:
        for label in (0, 1):
            selection = frame.loc[(frame["source"] == source) & (frame["label"] == label)]
            with_path = selection.loc[selection["path_length"].fillna(0) > 0]
            rows.append({
                "stage": stage,
                "source": source,
                "label": label,
                "records": len(selection),
                "unique_urls": selection["url"].nunique(),
                "path_records": len(with_path),
                "unique_urls_with_path": with_path["url"].nunique(),
                "path_fraction": len(with_path) / len(selection) if len(selection) else np.nan,
                "unparseable_records": int(selection["path_length"].isna().sum()),
            })
    return rows


def url_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["source", "label", "url"]].drop_duplicates()


def assert_stage_subset(before: pd.DataFrame, after: pd.DataFrame, label: str) -> None:
    """A stage cannot introduce a new source/class/URL combination."""
    left = url_pairs(before)
    right = url_pairs(after)
    merged = right.merge(left, on=["source", "label", "url"], how="left",
                         indicator=True, validate="one_to_one")
    if (merged["_merge"] != "both").any():
        raise ValueError(f"Unexpected source/label/URL introduced by {label}")


def path_transition(name: str, before: pd.DataFrame, after: pd.DataFrame,
                    *, raw_records: bool = False) -> list[dict]:
    """Track paths by unique URL after cleaning; raw uses record counts.

    Deduplication may merge identical records without losing a distinct URL.
    URLs not in TRAIN are not 'deleted' -- they may be in reserved splits.
    """
    rows = []
    for source in SOURCES:
        for label in (0, 1):
            a = before.loc[(before["source"] == source) & (before["label"] == label)]
            b = after.loc[(after["source"] == source) & (after["label"] == label)]
            a = a.loc[a["path_length"].fillna(0) > 0]
            b = b.loc[b["path_length"].fillna(0) > 0]
            n_before = len(a) if raw_records else a["url"].nunique()
            n_after = len(b) if raw_records else b["url"].nunique()
            rows.append({
                "transition": name,
                "source": source,
                "label": label,
                "count_basis": "records" if raw_records else "unique_urls",
                "input_with_path": n_before,
                "output_with_path": n_after,
                "difference": n_before - n_after,
                "interpretation": (
                    "cleaning_and_quarantine" if name == "raw_to_cleaned"
                    else "duplicate_or_conflict_removal" if name == "cleaned_to_deduplicated"
                    else "no_records_should_be_dropped" if name == "deduplicated_to_domain_groups"
                    else "excluded_without_domain_group" if name == "domain_groups_to_eligible"
                    else "reserved_or_other_split_not_lost" if name == "eligible_to_train"
                    else "feature_extraction_not_filtering"
                ),
            })
    return rows


def exclusion_reasons(quarantine: pd.DataFrame, conflicts: pd.DataFrame,
                      ungrouped: pd.DataFrame, invalid_raw: dict[str, int]) -> pd.DataFrame:
    chunks = []
    for frame, reason_col, default_reason, single in (
        (quarantine, "reason", None, True),
        (conflicts, None, "conflicting_label", True),
        (ungrouped, None, "missing_group_key", False),
    ):
        if frame.empty:
            continue
        # Quarantined rows with invalid labels are separately classified.
        copy = frame.copy()
        labels = pd.to_numeric(copy["label"], errors="coerce")
        copy["audit_label"] = labels.apply(
            lambda number: str(int(number)) if number in (0, 1) else "invalid"
        )
        if single:
            expanded = copy
            if not expanded["source"].isin(SOURCES).all():
                raise ValueError("Unknown quarantine/conflict source")
        else:
            expanded = copy.assign(
                source=copy["sources"].str.split("|", regex=False)
            ).explode("source", ignore_index=True)
        expanded["audit_reason"] = expanded[reason_col] if reason_col else default_reason
        expanded["path_length"] = [url_path_length(v) for v in expanded["url"]]
        chunks.append(expanded[["source", "audit_label", "audit_reason", "path_length"]])
    if chunks:
        joined = pd.concat(chunks, ignore_index=True)
        rows = []
        for (src, lab, why), group in joined.groupby(
            ["source", "audit_label", "audit_reason"], dropna=False, sort=True
        ):
            rows.append({
                "source": src, "label": lab, "reason": why,
                "records": len(group),
                "records_with_path": int(group["path_length"].fillna(0).gt(0).sum()),
                "unparseable_records": int(group["path_length"].isna().sum()),
            })
    else:
        rows = []
    # These are included for explicit reconciliation with each original dataset.
    for source, count in invalid_raw.items():
        rows.append({
            "source": source, "label": "invalid", "reason": "raw_invalid_label_total",
            "records": count, "records_with_path": 0, "unparseable_records": 0,
        })
    return pd.DataFrame(rows, columns=["source", "label", "reason", "records",
                                       "records_with_path", "unparseable_records"])


def verify_unchanged(left: pd.DataFrame, right: pd.DataFrame, label: str,
                     *, compare_group_key: bool = False) -> None:
    """Compare only metadata necessary for checking an unchanged stage."""
    cols = ["url", "label", "sources"]
    if compare_group_key:
        cols.append("group_key")
    if left["url"].duplicated().any() or right["url"].duplicated().any():
        raise ValueError(f"Duplicate URL in {label}")
    l = left[cols].set_index("url").sort_index()
    r = right[cols].set_index("url").sort_index()
    if not l.equals(r):
        # Equality should ignore pandas dtype differences from CSV loading.
        if list(l.index) != list(r.index) or any(
            not l[col].astype(str).equals(r[col].astype(str)) for col in cols[1:]
        ):
            raise ValueError(f"Unexpected stage metadata change: {label}")


def audit(base_dir: Path = BASE_DIR) -> dict[str, pd.DataFrame]:
    """Run the read-only, train-only audit and return aggregate reports."""
    base_dir = Path(base_dir)
    processed_dir = base_dir / "data" / "processed"
    split_dir = base_dir / "data" / "splits"
    raw, raw_counts = load_raw(base_dir)
    clean = read_csv(processed_dir / "cleaned_urls.csv", ("url", "label", "source"))
    quarantine = read_csv(processed_dir / "quarantine_urls.csv", ("url", "label", "source", "reason"))
    dedup = read_csv(processed_dir / "deduplicated_urls.csv", ("url", "label", "sources"))
    conflicts = read_csv(processed_dir / "conflicting_labels.csv", ("url", "label", "source"))
    grouped = read_csv(processed_dir / "domain_groups.csv", ("url", "label", "sources", "group_key"))
    # Crucial: never open validation.csv, test.csv or their V2 counterparts.
    train = read_csv(split_dir / "train.csv", ("url", "label", "sources", "group_key"))
    v2_train = read_csv(split_dir / "v2" / "train.csv",
                        ("url", "label", "sources", "group_key", "v2_path_length"))

    if not raw_counts or any(count == 0 for count in raw_counts.values()):
        raise ValueError("Empty original dataset")
    invalid_raw = {src: raw_counts[src] - int((raw["source"] == src).sum())
                   for src in SOURCES}
    quarantine_labels = pd.to_numeric(quarantine["label"], errors="coerce")
    for source in SOURCES:
        actual = int((clean["source"] == source).sum()) + int((quarantine["source"] == source).sum())
        if actual != raw_counts[source]:
            raise ValueError(f"Raw->clean/quarantine row totals inconsistent: {source}")
        for label in (0, 1):
            original = int(((raw["source"] == source) & (raw["label"] == label)).sum())
            clean_n = int(((clean["source"] == source) & (clean["label"] == label)).sum())
            rejected = int(((quarantine["source"] == source) & (quarantine_labels == label)).sum())
            if original != clean_n + rejected:
                raise ValueError(f"Raw->clean class totals inconsistent: {source}, {label}")

    verify_unchanged(dedup, grouped, "dedup->domain_groups")
    eligible = grouped.loc[grouped["group_key"].astype(str).ne("")].copy()
    ungrouped = grouped.loc[grouped["group_key"].astype(str).eq("")].copy()
    if train["group_key"].astype(str).eq("").any():
        raise ValueError("TRAIN contains blank group keys")
    if eligible["url"].duplicated().any() or train["url"].duplicated().any():
        raise ValueError("Duplicate URLs before/during training split")
    # TRAIN must be a faithful subset of the eligible, pre-split metadata.
    checked = train[["url", "label", "sources", "group_key"]].merge(
        eligible[["url", "label", "sources", "group_key"]],
        on="url", how="left", suffixes=("_train", "_eligible"),
        indicator=True, validate="one_to_one",
    )
    if (checked["_merge"] != "both").any() or any(
        not checked[f"{column}_train"].astype(str).equals(
            checked[f"{column}_eligible"].astype(str))
        for column in ("label", "sources", "group_key")
    ):
        raise ValueError("TRAIN metadata does not match eligible input")
    verify_unchanged(train, v2_train, "train->V2 train", compare_group_key=True)

    raw_stage = with_path_lengths(expand_stage(raw, single_source=True))
    clean_stage = with_path_lengths(expand_stage(clean, single_source=True))
    dedup_stage = with_path_lengths(expand_stage(dedup, single_source=False))
    group_stage = with_path_lengths(expand_stage(grouped, single_source=False))
    eligible_stage = with_path_lengths(expand_stage(eligible, single_source=False))
    train_stage = with_path_lengths(expand_stage(train, single_source=False))
    v2_stage = with_path_lengths(expand_stage(v2_train, single_source=False), v2=True)

    assert_stage_subset(clean_stage, dedup_stage, "clean->dedup")
    assert_stage_subset(group_stage, eligible_stage, "grouped->eligible")
    assert_stage_subset(eligible_stage, train_stage, "eligible->train")
    assert_stage_subset(train_stage, v2_stage, "train->V2 train")

    stages = {
        "raw_valid_labels": raw_stage,
        "cleaned": clean_stage,
        "deduplicated": dedup_stage,
        "domain_groups": group_stage,
        "eligible_for_split": eligible_stage,
        "train": train_stage,
        "v2_train": v2_stage,
    }
    coverage_rows = [row for stage, frame in stages.items() for row in coverage(stage, frame)]
    transitions = []
    for name, a, b, raw_records in (
        ("raw_to_cleaned", raw_stage, clean_stage, True),
        ("cleaned_to_deduplicated", clean_stage, dedup_stage, False),
        ("deduplicated_to_domain_groups", dedup_stage, group_stage, False),
        ("domain_groups_to_eligible", group_stage, eligible_stage, False),
        ("eligible_to_train", eligible_stage, train_stage, False),
        ("train_to_v2_train", train_stage, v2_stage, False),
    ):
        transitions.extend(path_transition(name, a, b, raw_records=raw_records))

    # Verify path feature semantics for every TRAIN row (one URL, one feature).
    original_path = [url_path_length(url) for url in train["url"]]
    observed = pd.to_numeric(v2_train["v2_path_length"], errors="coerce")
    compare = train[["url"]].assign(expected=original_path).merge(
        v2_train[["url", "v2_path_length"]], on="url", validate="one_to_one"
    )
    expected = pd.to_numeric(compare["expected"], errors="coerce")
    actual = pd.to_numeric(compare["v2_path_length"], errors="coerce")
    inconsistent = expected.isna() | actual.isna() | ~np.isclose(
        expected.fillna(-1).to_numpy(dtype=float),
        actual.fillna(-2).to_numpy(dtype=float), atol=0, rtol=0
    )
    integrity = pd.DataFrame([
        {"check": "raw_records_reconciled_with_clean_and_quarantine",
         "checked": sum(raw_counts.values()), "anomalies": 0},
        {"check": "deduplicated_rows_preserved_in_domain_groups",
         "checked": len(dedup), "anomalies": 0},
        {"check": "train_rows_preserved_in_v2_train",
         "checked": len(train), "anomalies": 0},
        {"check": "v2_path_length_matches_parsed_train_url",
         "checked": len(compare), "anomalies": int(inconsistent.sum())},
    ])
    return {
        "coverage": pd.DataFrame(coverage_rows),
        "transitions": pd.DataFrame(transitions),
        "excluded": exclusion_reasons(quarantine, conflicts, ungrouped, invalid_raw),
        "integrity": integrity,
    }


def save_reports(reports: dict[str, pd.DataFrame], directory: Path = REPORT_DIR) -> None:
    """Save aggregate-only CSVs. Never include URLs, group keys, or hostnames."""
    if set(reports) != set(REPORT_FILES):
        raise ValueError("Missing audit reports")
    forbidden = {"url", "original_url", "group_key", "hostname", "domain", "source_row"}
    for name, frame in reports.items():
        if frame.empty or not forbidden.isdisjoint(frame.columns):
            raise ValueError(f"Empty or potentially identifying report: {name}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, filename in REPORT_FILES.items():
        target = directory / filename
        reports[name].to_csv(target, index=False)
        print(f"Saved: {target}")


def main() -> None:
    print("PHISHGUARD AI - PIPELINE PATH ATTRITION AUDIT")
    print("Read-only: RAW/processed/TRAIN/V2 TRAIN. Never validation/test.\n")
    reports = audit(BASE_DIR)
    save_reports(reports)
    paths = reports["coverage"]
    print("\nBENIGN URL PATH COVERAGE BY STAGE")
    print(paths.loc[paths["label"] == 0, [
        "stage", "source", "records", "path_records", "path_fraction"
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nRAW/CLEAN/DEDUP/SPLIT TRANSITIONS - BENIGN")
    print(reports["transitions"].loc[
        reports["transitions"]["label"] == 0,
        ["transition", "source", "input_with_path", "output_with_path", "difference"]
    ].to_string(index=False))
    print("\nINTEGRITY CHECKS")
    print(reports["integrity"].to_string(index=False))
    print("\nNote: counts overlap across sources; held-out rows are not deleted.")
    print("The audit does not validate the truth of source labels or model performance.")


if __name__ == "__main__":
    main()
