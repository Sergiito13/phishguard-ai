"""
PhishGuard AI - Exploratory dataset source predictability.

Predict which dataset supplied an URL using protocol-independent V2 features.

Only the EXISTING training split is used. A new, group-disjoint diagnostic
holdout is taken from that split, so this is NOT independent validation.

Multi-source URLs are excluded because they have no single provenance label.

Separate models are evaluated for all URLs, legitimate only, and phishing
only, to reveal possible confounding by the phishing label.

No saved phishing model is altered; no URLs or domain identifiers are exported.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from sklearn.model_selection import GroupShuffleSplit

from experiment_hyperparameters import (
    FEATURE_SETS,
    load_training_data,
)

from experiment_training_strategies import (
    SOURCES,
    verify_group_isolation,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

FEATURES = list(
    FEATURE_SETS["extended_v2"]
)

RANDOM_STATE = 42

HOLDOUT_FRACTION = 0.20

TRAIN_LIMIT_PER_SOURCE_LABEL = 3000

HOLDOUT_LIMIT_PER_SOURCE_LABEL = 1000

IMPORTANCE_LIMIT_PER_SOURCE = 200

IMPORTANCE_REPEATS = 2


REPORT_FILES = {
    "coverage": "source_predictability_coverage.csv",

    "summary": "source_predictability_summary.csv",

    "per_source": "source_predictability_per_source.csv",

    "confusion": "source_predictability_confusion.csv",

    "importance": "source_predictability_feature_importance.csv",
}


# ============================================================
# 2. PREPARE SINGLE-SOURCE DATA
# ============================================================

def prepare_single_source(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep URLs associated with exactly one known source.

    URLs belonging to multiple datasets are excluded
    because their provenance is not a single class.
    """

    required = {
        "sources",
        "label",
        "group_key",
        *FEATURES,
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if data.empty or not data["label"].isin([0, 1]).all():
        raise ValueError(
            "Dataset empty or labels invalid"
        )

    if (
        data["group_key"].isna().any()
        or data["group_key"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Missing domain group keys"
        )

    if data["sources"].isna().any():
        raise ValueError(
            "Missing source memberships"
        )

    memberships = (
        data["sources"]
        .astype(str)
        .str.split("|", regex=False)
    )

    invalid_memberships = memberships.map(
        lambda parts: (
            not parts
            or any(
                part not in SOURCES
                for part in parts
            )
            or len(parts) != len(set(parts))
        )
    )

    if invalid_memberships.any():
        raise ValueError(
            "Invalid source memberships"
        )

    single_mask = memberships.map(len).eq(1)

    # Keep only metadata and the V2 features.
    # Raw URLs are not needed for this experiment.

    single = data.loc[
        single_mask,
        [
            "sources",
            "label",
            "group_key",
            *FEATURES,
        ],
    ].copy()

    single["source"] = (
        single["sources"].astype(str)
    )

    if single.empty:
        raise ValueError(
            "No single-source URLs available"
        )

    if not set(SOURCES).issubset(
        set(single["source"])
    ):
        raise ValueError(
            "Some sources have no single-source URLs"
        )

    numerical = single[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numerical).all():
        raise ValueError(
            "Invalid V2 features"
        )

    coverage = pd.DataFrame([
        {
            "input_rows": int(len(data)),

            "single_source_rows": int(
                single_mask.sum()
            ),

            "excluded_multi_source_rows": int(
                (~single_mask).sum()
            ),

            "excluded_fraction": float(
                (~single_mask).mean()
            ),

            "single_source_group_count": int(
                single["group_key"].nunique()
            ),
        }
    ])

    return single, coverage


# ============================================================
# 3. GROUP-AWARE DIAGNOSTIC SPLIT
# ============================================================

def group_holdout(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a diagnostic holdout from training data.

    The split is performed BEFORE balancing and sampling.

    No registered-domain group may appear in both
    the training and diagnostic holdout.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_idx, holdout_idx = next(
        splitter.split(
            data,
            groups=data["group_key"],
        )
    )

    train = data.iloc[
        train_idx
    ].copy()

    holdout = data.iloc[
        holdout_idx
    ].copy()

    verify_group_isolation(
        train,
        holdout,
    )

    return train, holdout


# ============================================================
# 4. BALANCED SAMPLING
# ============================================================

def balanced_sample(
    data: pd.DataFrame,
    max_per_cell: int,
) -> pd.DataFrame:
    """
    Sample each source/label combination separately.

    This limits label-distribution confounding.

    Sampling is performed without replacement.
    """

    if max_per_cell < 1:
        raise ValueError(
            "max_per_cell must be positive"
        )

    pieces = []

    for source in SOURCES:

        for label in (0, 1):

            cell = data.loc[
                (
                    (data["source"] == source)
                    &
                    (data["label"] == label)
                )
            ]

            if cell.empty:
                raise ValueError(
                    f"Missing source/label cell: "
                    f"{source}/{label}"
                )

            sampled = cell.sample(
                n=min(
                    len(cell),
                    max_per_cell,
                ),
                random_state=RANDOM_STATE,
            )

            pieces.append(
                sampled
            )

    return (
        pd.concat(
            pieces,
            axis=0,
        )
        .sort_index()
    )


# ============================================================
# 5. CREATE SOURCE CLASSIFIER
# ============================================================

def create_classifier():
    """
    Create a dedicated provenance classifier.

    This is not the deployed phishing classifier.
    """

    return HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
    )


# ============================================================
# 6. FEATURE IMPORTANCE SAMPLING
# ============================================================

def select_importance_rows(
    frame: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    """
    Build a source-balanced evaluation sample
    for permutation feature importance.
    """

    if limit < 1:
        raise ValueError(
            "Importance limit must be positive"
        )

    pieces = []

    for source in SOURCES:

        subset = frame.loc[
            frame["source"] == source
        ]

        if subset.empty:
            raise ValueError(
                f"No examples for source: {source}"
            )

        sampled = subset.sample(
            n=min(
                limit,
                len(subset),
            ),
            random_state=RANDOM_STATE,
        )

        pieces.append(
            sampled
        )

    return pd.concat(
        pieces,
        axis=0,
    )


# ============================================================
# 7. RUN SOURCE PREDICTABILITY EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    train_limit: int = TRAIN_LIMIT_PER_SOURCE_LABEL,
    holdout_limit: int = HOLDOUT_LIMIT_PER_SOURCE_LABEL,
    importance_limit: int = IMPORTANCE_LIMIT_PER_SOURCE,
    importance_repeats: int = IMPORTANCE_REPEATS,
) -> dict[str, pd.DataFrame]:
    """
    Train and evaluate three source classifiers.

    Scenarios:
        1. All URLs.
        2. Legitimate URLs only.
        3. Phishing URLs only.

    Uses a group-disjoint diagnostic holdout
    derived exclusively from the training split.
    """

    if importance_repeats < 1:
        raise ValueError(
            "importance_repeats must be positive"
        )

    # --------------------------------------------------------
    # A. FILTER MULTI-SOURCE URLS
    # --------------------------------------------------------

    single, coverage = prepare_single_source(
        dataframe
    )

    # --------------------------------------------------------
    # B. DOMAIN-DISJOINT SPLIT
    # --------------------------------------------------------

    train_all, holdout_all = group_holdout(
        single
    )

    # --------------------------------------------------------
    # C. BALANCED SUBSAMPLING
    # --------------------------------------------------------

    train = balanced_sample(
        train_all,
        train_limit,
    )

    held = balanced_sample(
        holdout_all,
        holdout_limit,
    )

    verify_group_isolation(
        train,
        held,
    )

    # --------------------------------------------------------
    # D. COVERAGE REPORT
    # --------------------------------------------------------

    coverage["diagnostic_fit_rows"] = len(
        train
    )

    coverage["diagnostic_holdout_rows"] = len(
        held
    )

    coverage["diagnostic_fit_groups"] = (
        train["group_key"].nunique()
    )

    coverage["diagnostic_holdout_groups"] = (
        held["group_key"].nunique()
    )

    # --------------------------------------------------------
    # E. EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    summaries = []
    per_sources = []
    confusions = []
    importances = []

    scenarios = {
        "all": None,
        "legitimate_only": 0,
        "phishing_only": 1,
    }

    for scenario, label in scenarios.items():

        # ----------------------------------------------------
        # F. SELECT SCENARIO
        # ----------------------------------------------------

        fit = (
            train
            if label is None
            else train.loc[
                train["label"] == label
            ]
        )

        evaluate = (
            held
            if label is None
            else held.loc[
                held["label"] == label
            ]
        )

        if (
            set(fit["source"]) != set(SOURCES)
            or set(evaluate["source"]) != set(SOURCES)
        ):
            raise ValueError(
                f"Incomplete source coverage in: {scenario}"
            )

        verify_group_isolation(
            fit,
            evaluate,
        )

        # ----------------------------------------------------
        # G. PREPARE FEATURES
        # ----------------------------------------------------

        X_fit = fit[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )

        y_fit = fit[
            "source"
        ].to_numpy()

        X_eval = evaluate[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )

        y_eval = evaluate[
            "source"
        ].to_numpy()

        # ----------------------------------------------------
        # H. TRAIN SOURCE CLASSIFIER
        # ----------------------------------------------------

        model = create_classifier()

        model.fit(
            X_fit,
            y_fit,
        )

        predictions = model.predict(
            X_eval
        )

        # ----------------------------------------------------
        # I. CALCULATE METRICS
        # ----------------------------------------------------

        balanced_accuracy = float(
            balanced_accuracy_score(
                y_eval,
                predictions,
            )
        )

        majority_source = (
            fit["source"]
            .value_counts()
            .idxmax()
        )

        majority_accuracy = float(
            (
                y_eval == majority_source
            ).mean()
        )

        summaries.append({
            "scenario": scenario,

            "fit_samples": len(fit),

            "holdout_samples": len(evaluate),

            "accuracy": float(
                accuracy_score(
                    y_eval,
                    predictions,
                )
            ),

            "balanced_accuracy": balanced_accuracy,

            "macro_f1": float(
                f1_score(
                    y_eval,
                    predictions,
                    labels=SOURCES,
                    average="macro",
                    zero_division=0,
                )
            ),

            "majority_baseline_accuracy":
                majority_accuracy,

            "chance_balanced_accuracy":
                1 / len(SOURCES),
        })

        # ----------------------------------------------------
        # J. PER-SOURCE PERFORMANCE
        # ----------------------------------------------------

        for source in SOURCES:

            mask = (
                y_eval == source
            )

            per_sources.append({
                "scenario": scenario,

                "source": source,

                "holdout_samples": int(
                    mask.sum()
                ),

                "source_recall": float(
                    (
                        predictions[mask] == source
                    ).mean()
                ),
            })

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            for predicted_source in SOURCES:

                count = int(
                    (
                        (
                            predictions
                            == predicted_source
                        )
                        &
                        mask
                    ).sum()
                )

                confusions.append({
                    "scenario": scenario,

                    "true_source": source,

                    "predicted_source": predicted_source,

                    "count": count,
                })

        # ----------------------------------------------------
        # K. PERMUTATION FEATURE IMPORTANCE
        # ----------------------------------------------------

        importance_set = select_importance_rows(
            evaluate,
            importance_limit,
        )

        importance_result = permutation_importance(
            model,

            importance_set[
                FEATURES
            ].to_numpy(
                dtype=np.float64
            ),

            importance_set[
                "source"
            ].to_numpy(),

            scoring="balanced_accuracy",

            n_repeats=importance_repeats,

            random_state=RANDOM_STATE,

            n_jobs=1,
        )

        for index, feature in enumerate(FEATURES):

            importances.append({
                "scenario": scenario,

                "feature": feature,

                "mean_balanced_accuracy_drop": float(
                    importance_result
                    .importances_mean[index]
                ),

                "std_balanced_accuracy_drop": float(
                    importance_result
                    .importances_std[index]
                ),
            })

        # ----------------------------------------------------
        # L. DISPLAY PROGRESS
        # ----------------------------------------------------

        print(
            f"{scenario:<17} "
            f"train={len(fit):>6,} "
            f"holdout={len(evaluate):>6,} "
            f"balanced_accuracy={balanced_accuracy:.4f} "
            f"(chance=0.2500)"
        )

    # ========================================================
    # 8. RETURN REPORTS
    # ========================================================

    return {
        "coverage": coverage,

        "summary": pd.DataFrame(
            summaries
        ),

        "per_source": pd.DataFrame(
            per_sources
        ),

        "confusion": pd.DataFrame(
            confusions
        ),

        "importance": pd.DataFrame(
            importances
        ),
    }


# ============================================================
# 9. SAVE REPORTS
# ============================================================

def save_reports(
    reports: dict[str, pd.DataFrame],
) -> None:
    """
    Export aggregated statistics only.

    Never export URLs, hostnames or domain identifiers.
    """

    if set(reports) != set(REPORT_FILES):
        raise ValueError(
            "Incomplete reports"
        )

    forbidden = {
        "url",
        "urls",
        "group_key",
        "sources",
        "hostname",
        "domain",
    }

    for name, frame in reports.items():

        if (
            frame.empty
            or not forbidden.isdisjoint(
                frame.columns
            )
        ):
            raise ValueError(
                f"Empty or sensitive report: {name}"
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, filename in REPORT_FILES.items():

        output = REPORT_DIR / filename

        reports[name].to_csv(
            output,
            index=False,
        )

        print(
            f"Saved: {output}"
        )


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        "PHISHGUARD AI - DATASET SOURCE PREDICTABILITY"
    )

    print("=" * 72)

    print(
        "Original training split only; "
        "domain-disjoint diagnostic holdout."
    )

    print(
        "Multi-source URLs are excluded; "
        "source/label cells sampled separately."
    )

    data = load_training_data()

    print(
        f"\nLoaded training rows: {len(data):,}"
    )

    reports = run_experiment(
        data
    )

    save_reports(
        reports
    )

    # --------------------------------------------------------
    # SOURCE PREDICTABILITY SUMMARY
    # --------------------------------------------------------

    print("\nSOURCE PREDICTABILITY SUMMARY")

    print(
        reports["summary"].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # PER-SOURCE PERFORMANCE
    # --------------------------------------------------------

    print("\nSOURCE RECALL")

    print(
        reports["per_source"].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # MOST IMPORTANT FEATURES
    # --------------------------------------------------------

    print(
        "\nTOP PROVENANCE FEATURES "
        "(PERMUTATION IMPORTANCE)"
    )

    for scenario in (
        "all",
        "legitimate_only",
        "phishing_only",
    ):

        subset = reports["importance"].loc[
            reports["importance"]["scenario"]
            == scenario
        ]

        print(
            f"\n{scenario}:"
        )

        print(
            subset.nlargest(
                8,
                "mean_balanced_accuracy_drop",
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print(
        "\nDiagnostic only: source predictability "
        "does not prove data leakage or incorrect labels."
    )

    print(
        "Reserved validation/test data and saved "
        "phishing models were not accessed."
    )


if __name__ == "__main__":
    main()