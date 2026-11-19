"""
PhishGuard AI - Training Source Quality Audit

Analyze source-specific biases in the training split.

Checks:
- Label distribution by source.
- HTTP/HTTPS distribution by source and label.
- URL structural characteristics.
- Shared URLs and domain groups.
- Feature extraction consistency.
- Domain group concentration.

IMPORTANT:
- Only train.csv is accessed.
- Validation and test datasets are not accessed.
- No models are trained or modified.
- No network requests are performed.
- Reports contain aggregate statistics only.

These diagnostics cannot establish dataset provenance
or prove that individual labels are incorrect.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_training_strategies import SOURCES


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data" / "splits" / "train.csv"

REPORT_DIR = BASE_DIR / "reports"

REQUIRED = (
    "url",
    "label",
    "sources",
    "group_key",
    "domain_status",
    "is_https",
    "has_ip",
    "url_length",
    "slash_count",
)

REPORT_NAMES = {
    "overview": "source_quality_overview.csv",
    "scheme": "source_quality_scheme_by_label.csv",
    "structure": "source_quality_structure_by_label.csv",
    "integrity": "source_quality_integrity.csv",
}


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data(
    path: Path | None = None,
) -> pd.DataFrame:
    """
    Load only the existing training split.

    Read the necessary columns without modifying
    the original dataset.
    """

    path = (
        TRAIN_FILE
        if path is None
        else Path(path)
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=list(REQUIRED),
        keep_default_na=False,
    )

    validate_training_data(df)

    return df


# ============================================================
# 3. VALIDATE DATASET
# ============================================================

def validate_training_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate labels, URLs, source provenance,
    domain groups and numerical features.
    """

    missing = set(REQUIRED) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Training data is empty"
        )

    if not df["label"].isin((0, 1)).all():
        raise ValueError(
            "Invalid labels: expected 0 or 1"
        )

    if (
        df["url"].isna().any()
        or df["url"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "Empty URLs"
        )

    if df["url"].duplicated().any():
        raise ValueError(
            "Duplicate URLs in training data"
        )

    if (
        df["group_key"].isna().any()
        or df["group_key"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Missing domain group keys"
        )

    # Validate source memberships.

    allowed = set(SOURCES)

    for value in df["sources"]:

        parts = str(value).split("|")

        if (
            not parts
            or any(
                not part or part not in allowed
                for part in parts
            )
        ):
            raise ValueError(
                f"Unknown or empty source membership: {value!r}"
            )

        if len(parts) != len(set(parts)):
            raise ValueError(
                f"Duplicate source membership: {value!r}"
            )

    # Validate binary features.

    for feature in (
        "is_https",
        "has_ip",
    ):

        if not df[feature].isin((0, 1)).all():
            raise ValueError(
                f"Invalid binary feature: {feature}"
            )

    # Validate numerical features.

    for feature in (
        "url_length",
        "slash_count",
    ):

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        if (
            not np.isfinite(
                values.to_numpy(dtype=float)
            ).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"Invalid numeric feature: {feature}"
            )


# ============================================================
# 4. ANALYZE DATASET QUALITY
# ============================================================

def build_reports(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze:
    - Label distribution.
    - URL schemes.
    - Structural characteristics.
    - Shared domain groups.
    - Feature consistency.

    URLs belonging to multiple sources contribute
    once to each corresponding source.
    """

    validate_training_data(df)

    working = df.copy()

    urls = working["url"].astype(str)

    lower = urls.str.lower()

    # --------------------------------------------------------
    # A. URL PROTOCOLS
    # --------------------------------------------------------

    https = lower.str.startswith(
        "https://"
    )

    http = lower.str.startswith(
        "http://"
    )

    explicit = lower.str.match(
        r"^[a-z][a-z0-9+.-]*://"
    )

    working["scheme"] = np.select(
        [
            https,
            http,
            explicit,
        ],
        [
            "https",
            "http",
            "other_scheme",
        ],
        default="missing_scheme",
    )

    working["raw_https"] = https.astype(int)

    # --------------------------------------------------------
    # B. URL STRUCTURE
    # --------------------------------------------------------

    working["raw_url_length"] = urls.str.len()

    working["raw_slash_count"] = urls.str.count(
        "/"
    )

    working["has_query"] = (
        urls.str.contains(
            "?",
            regex=False,
        ).astype(int)
    )

    working["has_at"] = (
        urls.str.contains(
            "@",
            regex=False,
        ).astype(int)
    )

    # --------------------------------------------------------
    # C. SOURCE MEMBERSHIP
    # --------------------------------------------------------

    working["source_count"] = (
        working["sources"].str.count(r"\|") + 1
    )

    # --------------------------------------------------------
    # D. FEATURE CONSISTENCY
    # --------------------------------------------------------

    working["https_mismatch"] = (
        working["raw_https"]
        != working["is_https"]
    ).astype(int)

    working["length_mismatch"] = (
        working["raw_url_length"]
        != working["url_length"]
    ).astype(int)

    working["slash_mismatch"] = (
        working["raw_slash_count"]
        != working["slash_count"]
    ).astype(int)

    # --------------------------------------------------------
    # E. EXPAND SOURCE MEMBERSHIPS
    # --------------------------------------------------------

    expanded = working.assign(
        source=working["sources"].str.split(
            "|",
            regex=False,
        )
    ).explode(
        "source",
        ignore_index=True,
    )

    # Identify groups appearing in multiple sources.

    source_count_by_group = (
        expanded.groupby("group_key")["source"]
        .nunique()
    )

    expanded["group_shared_by_sources"] = (
        expanded["group_key"].map(
            source_count_by_group
        ) > 1
    )

    # ========================================================
    # REPORT 1: SOURCE OVERVIEW
    # ========================================================

    overview_rows = []

    for source in SOURCES:

        frame = expanded.loc[
            expanded["source"] == source
        ]

        groups = frame.groupby(
            "group_key",
            sort=False,
        )

        group_sizes = groups.size()

        label_counts = frame[
            "label"
        ].value_counts()

        legitimate = int(
            label_counts.get(0, 0)
        )

        phishing = int(
            label_counts.get(1, 0)
        )

        total = len(frame)

        overview_rows.append({
            "source": source,

            "rows": total,

            "legitimate": legitimate,

            "phishing": phishing,

            "phishing_fraction": (
                phishing / total
                if total
                else np.nan
            ),

            "multi_source_rows": int(
                frame["source_count"].gt(1).sum()
            ),

            "unique_group_keys": int(
                len(group_sizes)
            ),

            "mixed_label_group_keys": int(
                groups["label"]
                .nunique()
                .gt(1)
                .sum()
            ),

            "rows_in_groups_shared_across_sources": int(
                frame["group_shared_by_sources"].sum()
            ),

            "largest_group_fraction": (
                float(
                    group_sizes.max() / total
                )
                if total
                else np.nan
            ),
        })

    overview = pd.DataFrame(
        overview_rows
    )

    # ========================================================
    # REPORT 2: SCHEME DISTRIBUTION
    # ========================================================

    scheme = (
        expanded.groupby(
            [
                "source",
                "label",
                "scheme",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    scheme["class_total"] = (
        scheme.groupby(
            [
                "source",
                "label",
            ]
        )["count"].transform("sum")
    )

    scheme["fraction_within_class"] = (
        scheme["count"]
        / scheme["class_total"]
    )

    # ========================================================
    # REPORT 3: STRUCTURAL CHARACTERISTICS
    # ========================================================

    structure = (
        expanded.groupby(
            [
                "source",
                "label",
            ],
            as_index=False,
        )
        .agg(
            rows=(
                "url",
                "size",
            ),

            https_fraction=(
                "raw_https",
                "mean",
            ),

            ip_fraction_v1=(
                "has_ip",
                "mean",
            ),

            has_query_fraction=(
                "has_query",
                "mean",
            ),

            has_at_fraction=(
                "has_at",
                "mean",
            ),

            url_length_mean=(
                "raw_url_length",
                "mean",
            ),

            url_length_median=(
                "raw_url_length",
                "median",
            ),

            slash_count_mean=(
                "raw_slash_count",
                "mean",
            ),

            https_feature_mismatches=(
                "https_mismatch",
                "sum",
            ),

            length_feature_mismatches=(
                "length_mismatch",
                "sum",
            ),

            slash_feature_mismatches=(
                "slash_mismatch",
                "sum",
            ),
        )
    )

    # ========================================================
    # REPORT 4: INTEGRITY
    # ========================================================

    integrity = pd.DataFrame([
        {
            "training_rows": len(working),

            "source_memberships": len(expanded),

            "urls_in_multiple_sources": int(
                working["source_count"].gt(1).sum()
            ),

            "unique_group_keys": int(
                working["group_key"].nunique()
            ),

            "group_keys_shared_across_sources": int(
                source_count_by_group.gt(1).sum()
            ),

            "mixed_label_group_keys": int(
                working.groupby(
                    "group_key"
                )["label"]
                .nunique()
                .gt(1)
                .sum()
            ),

            "blank_domain_status_rows": int(
                working["domain_status"]
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            ),

            "https_feature_mismatches": int(
                working["https_mismatch"].sum()
            ),

            "length_feature_mismatches": int(
                working["length_mismatch"].sum()
            ),

            "slash_feature_mismatches": int(
                working["slash_mismatch"].sum()
            ),
        }
    ])

    return {
        "overview": overview,
        "scheme": scheme,
        "structure": structure,
        "integrity": integrity,
    }


# ============================================================
# 5. SAVE REPORTS
# ============================================================

def save_reports(
    reports: dict[str, pd.DataFrame],
) -> None:
    """
    Export aggregate results without raw URLs
    or domain identifiers.
    """

    if set(reports) != set(REPORT_NAMES):
        raise ValueError(
            "Incomplete report set"
        )

    forbidden = {
        "url",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    for name, frame in reports.items():

        if (
            frame.empty
            or forbidden.intersection(
                frame.columns
            )
        ):
            raise ValueError(
                f"Invalid or sensitive report: {name}"
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, filename in REPORT_NAMES.items():

        target = REPORT_DIR / filename

        reports[name].to_csv(
            target,
            index=False,
        )

        print(
            f"Saved: {target}"
        )


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("PHISHGUARD AI - TRAINING SOURCE QUALITY AUDIT")
    print("=" * 72)

    print(
        "Only train.csv is read. "
        "No model training or network access."
    )

    data = load_training_data()

    reports = build_reports(data)

    save_reports(reports)

    # --------------------------------------------------------
    # SOURCE OVERVIEW
    # --------------------------------------------------------

    print("\nSOURCE OVERVIEW")

    print(
        reports["overview"].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # SOURCE / LABEL STRUCTURE
    # --------------------------------------------------------

    print("\nSOURCE / LABEL STRUCTURE")

    columns = [
        "source",
        "label",
        "rows",
        "https_fraction",
        "ip_fraction_v1",
        "url_length_median",
        "slash_count_mean",
        "https_feature_mismatches",
    ]

    print(
        reports["structure"][columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # INTEGRITY
    # --------------------------------------------------------

    print("\nINTEGRITY")

    print(
        reports["integrity"].to_string(
            index=False,
        )
    )

    print(
        "\nNOTES: Multi-source rows are "
        "counted once per source."
    )

    print(
        "Mixed-label domains do not necessarily "
        "indicate incorrect labels."
    )

    print(
        "Raw-source label conflicts are audited "
        "separately in data_audit.json."
    )

    print(
        "Dataset provenance or synthetic origin "
        "cannot be inferred from these statistics."
    )

    print(
        "Reserved validation/test splits "
        "were not accessed."
    )


if __name__ == "__main__":
    main()