"""
PhishGuard AI: paired cross-source failure diagnostics for Extended V2.

Uses ONLY the existing training split. For each source, train two separate
models without that source or overlapping domain groups, then compare their
predictions on exactly the same held-out URLs.

Strategies: uniform and source_class. This analysis is exploratory: its source
folds already informed earlier model development, so it is NOT independent
validation. Does not read validation/test, export URLs or save model artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_cross_source_errors import compare_values
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

FEATURE_SET = "extended_v2"

FEATURES = list(
    FEATURE_SETS[FEATURE_SET]
)

STRATEGIES = (
    "uniform",
    "source_class",
)

THRESHOLD = 0.50

NEAR_THRESHOLD = 0.10

HIGH_CONFIDENCE = 0.80

LOW_CONFIDENCE = 0.20

SCORE_BINS = (
    0.0, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports"
)

REPORT_FILES = {
    "summary": "candidate_failures_summary.csv",
    "error_confidence": "candidate_failures_error_confidence.csv",
    "paired_changes": "candidate_failures_paired_changes.csv",
    "score_distribution": "candidate_failures_score_distribution.csv",
    "error_profiles": "candidate_failures_error_profiles.csv",
    "feature_shift": "candidate_failures_feature_shift.csv",
}


# ============================================================
# 2. VALIDATE INPUT
# ============================================================

def validate_input(
    data: pd.DataFrame,
    parameters: dict,
) -> None:

    if FEATURE_SET not in parameters:
        raise ValueError(
            "Missing optimized Extended V2 parameters"
        )

    if (
        not FEATURES
        or len(FEATURES) != len(set(FEATURES))
    ):
        raise ValueError(
            "Invalid feature configuration"
        )

    required = {
        "label",
        "sources",
        "group_key",
        "is_https",
        *FEATURES,
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing training fields: {sorted(missing)}"
        )

    if (
        data.empty
        or set(data["label"].unique()) != {0, 1}
    ):
        raise ValueError(
            "Training requires binary labels 0 and 1"
        )

    if not data["is_https"].isin((0, 1)).all():
        raise ValueError(
            "Invalid is_https values"
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
            "Missing group keys"
        )

    if not all(
        isinstance(value, str) and value
        for value in data["sources"]
    ):
        raise ValueError(
            "Missing source memberships"
        )

    for value in data["sources"].unique():

        parts = value.split("|")

        if (
            not parts
            or len(parts) != len(set(parts))
            or any(
                source not in SOURCES
                for source in parts
            )
        ):
            raise ValueError(
                f"Invalid source membership: {value!r}"
            )

    if not np.isfinite(
        data[FEATURES].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Non-finite V2 features"
        )


# ============================================================
# 3. CLASSIFICATION ERROR MASKS
# ============================================================

def wrong_masks(
    labels,
    scores,
) -> dict[str, np.ndarray]:

    y = np.asarray(labels)

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    if (
        y.ndim != 1
        or s.ndim != 1
        or len(y) == 0
        or len(y) != len(s)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(s).all()
        or ((s < 0) | (s > 1)).any()
    ):
        raise ValueError(
            "Invalid aligned labels and scores"
        )

    predicted = s >= THRESHOLD

    return {
        "FP": (y == 0) & predicted,
        "TN": (y == 0) & ~predicted,
        "FN": (y == 1) & ~predicted,
        "TP": (y == 1) & predicted,
    }


# ============================================================
# 4. ERROR CONFIDENCE ANALYSIS
# ============================================================

def confidence_rows(
    source: str,
    strategy: str,
    labels,
    scores,
) -> list[dict]:

    masks = wrong_masks(
        labels,
        scores,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    rows = []

    for kind in ("FP", "FN"):

        values = scores[
            masks[kind]
        ]

        near = int(
            np.sum(
                np.abs(
                    values - THRESHOLD
                ) <= NEAR_THRESHOLD
            )
        )

        if kind == "FP":

            high = int(
                np.sum(
                    values >= HIGH_CONFIDENCE
                )
            )

        else:

            high = int(
                np.sum(
                    values <= LOW_CONFIDENCE
                )
            )

        rows.append({
            "evaluation_source": source,

            "strategy": strategy,

            "error_type": kind,

            "errors": len(values),

            "near_threshold_errors": near,

            "near_threshold_fraction": (
                near / len(values)
                if len(values)
                else 0.0
            ),

            "high_confidence_errors": high,

            "high_confidence_fraction": (
                high / len(values)
                if len(values)
                else 0.0
            ),

            "median_error_score": (
                float(np.median(values))
                if len(values)
                else np.nan
            ),

            "mean_error_score": (
                float(np.mean(values))
                if len(values)
                else np.nan
            ),
        })

    return rows


# ============================================================
# 5. PAIRED ERROR COMPARISON
# ============================================================

def paired_change_rows(
    source: str,
    labels,
    uniform_scores,
    weighted_scores,
) -> list[dict]:

    y = np.asarray(labels)

    old = wrong_masks(
        y,
        uniform_scores,
    )

    new = wrong_masks(
        y,
        weighted_scores,
    )

    rows = []

    for kind, label in (
        ("FP", 0),
        ("FN", 1),
    ):

        population = y == label

        old_wrong = old[kind]

        new_wrong = new[kind]

        resolved = int(
            np.sum(
                old_wrong & ~new_wrong
            )
        )

        introduced = int(
            np.sum(
                ~old_wrong & new_wrong
            )
        )

        persistent = int(
            np.sum(
                old_wrong & new_wrong
            )
        )

        both_correct = int(
            np.sum(
                population
                & ~old_wrong
                & ~new_wrong
            )
        )

        total = int(
            population.sum()
        )

        if total != (
            resolved
            + introduced
            + persistent
            + both_correct
        ):
            raise ValueError(
                "Paired error accounting mismatch"
            )

        rows.append({
            "evaluation_source": source,

            "error_type": kind,

            "class_samples": total,

            "uniform_errors": (
                resolved + persistent
            ),

            "source_class_errors": (
                introduced + persistent
            ),

            "resolved_by_source_class": resolved,

            "introduced_by_source_class": introduced,

            "persistent_errors": persistent,

            "correct_in_both": both_correct,

            "net_error_change": (
                introduced - resolved
            ),

            "resolved_fraction": (
                resolved / total
                if total
                else 0.0
            ),

            "introduced_fraction": (
                introduced / total
                if total
                else 0.0
            ),
        })

    return rows


# ============================================================
# 6. SCORE DISTRIBUTION
# ============================================================

def distribution_rows(
    source: str,
    strategy: str,
    labels,
    scores,
) -> list[dict]:

    wrong_masks(
        labels,
        scores,
    )

    y = np.asarray(labels)

    s = np.asarray(
        scores,
        dtype=float,
    )

    rows = []

    for label in (0, 1):

        selected = s[
            y == label
        ]

        counts, edges = np.histogram(
            selected,
            bins=SCORE_BINS,
        )

        for index, count in enumerate(counts):

            rows.append({
                "evaluation_source": source,

                "strategy": strategy,

                "label": label,

                "score_low": float(
                    edges[index]
                ),

                "score_high": float(
                    edges[index + 1]
                ),

                "count": int(count),

                "class_samples": len(selected),
            })

    return rows


# ============================================================
# 7. FEATURE ERROR PROFILES
# ============================================================

def profile_rows(
    source: str,
    strategy: str,
    held: pd.DataFrame,
    labels,
    scores,
) -> list[dict]:

    if len(held) != len(labels):
        raise ValueError(
            "Held-out rows and predictions are not aligned"
        )

    masks = wrong_masks(
        labels,
        scores,
    )

    rows = []

    comparisons = (
        ("FP", "TN"),
        ("FN", "TP"),
    )

    for error, reference in comparisons:

        for feature in FEATURES:

            values = held[
                feature
            ].to_numpy(
                dtype=float
            )

            comparison = compare_values(
                values[masks[error]],
                values[masks[reference]],
            )

            rows.append({
                "evaluation_source": source,

                "strategy": strategy,

                "error_type": error,

                "reference_type": reference,

                "feature": feature,

                **comparison,
            })

    return rows


# ============================================================
# 8. FEATURE SHIFT
# ============================================================

def shift_rows(
    source: str,
    train: pd.DataFrame,
    held: pd.DataFrame,
) -> list[dict]:

    rows = []

    for label in (0, 1):

        train_class = train.loc[
            train["label"] == label
        ]

        held_class = held.loc[
            held["label"] == label
        ]

        for feature in FEATURES:

            comparison = compare_values(
                held_class[
                    feature
                ].to_numpy(
                    dtype=float
                ),

                train_class[
                    feature
                ].to_numpy(
                    dtype=float
                ),
            )

            rows.append({
                "evaluation_source": source,

                "label": label,

                "feature": feature,

                **comparison,
            })

    return rows


# ============================================================
# 9. RUN EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    parameters: dict,
) -> dict[str, pd.DataFrame]:

    validate_input(
        dataframe,
        parameters,
    )

    folds = create_source_folds(
        dataframe
    )

    if (
        len(folds) != len(SOURCES)
        or {
            fold["source"]
            for fold in folds
        } != set(SOURCES)
    ):
        raise ValueError(
            "Incomplete or duplicated source folds"
        )

    result = {
        name: []
        for name in REPORT_FILES
    }

    for fold in folds:

        source = fold["source"]

        fit = dataframe.iloc[
            fold["training_indices"]
        ]

        held = dataframe.iloc[
            fold["validation_indices"]
        ]

        verify_group_isolation(
            fit,
            held,
        )

        if belongs_to_source(
            fit,
            source,
        ).any():
            raise ValueError(
                f"Source leakage detected: {source}"
            )

        if (
            set(fit["label"].unique()) != {0, 1}
            or set(held["label"].unique()) != {0, 1}
        ):
            raise ValueError(
                f"Both classes required: {source}"
            )

        X_train = fit[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )

        y_train = fit[
            "label"
        ].to_numpy(
            dtype=int
        )

        X_held = held[
            FEATURES
        ].to_numpy(
            dtype=np.float64
        )

        y_held = held[
            "label"
        ].to_numpy(
            dtype=int
        )

        result["feature_shift"].extend(
            shift_rows(
                source,
                fit,
                held,
            )
        )

        scores_by_strategy = {}

        for strategy in STRATEGIES:

            model = create_model(
                parameters[FEATURE_SET]
            )

            weights = calculate_weights(
                fit,
                strategy,
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
            )

            scores = score_model(
                model,
                X_held,
            )

            scores_by_strategy[
                strategy
            ] = scores

            metrics = evaluate_scores(
                y_held,
                scores,
            )

            result["summary"].append({
                "feature_set": FEATURE_SET,

                "evaluation_source": source,

                "strategy": strategy,

                "training_samples": len(fit),

                "evaluation_samples": len(held),

                "threshold": THRESHOLD,

                **metrics,
            })

            result[
                "error_confidence"
            ].extend(
                confidence_rows(
                    source,
                    strategy,
                    y_held,
                    scores,
                )
            )

            result[
                "score_distribution"
            ].extend(
                distribution_rows(
                    source,
                    strategy,
                    y_held,
                    scores,
                )
            )

            result[
                "error_profiles"
            ].extend(
                profile_rows(
                    source,
                    strategy,
                    held,
                    y_held,
                    scores,
                )
            )

            print(
                f"{source:<11} "
                f"{strategy:<12} "
                f"FP={metrics['false_positives']:>6,} "
                f"FN={metrics['false_negatives']:>6,} "
                f"FPR={metrics['fpr']:.4f} "
                f"Recall={metrics['recall']:.4f}"
            )

        result[
            "paired_changes"
        ].extend(
            paired_change_rows(
                source,

                y_held,

                scores_by_strategy[
                    "uniform"
                ],

                scores_by_strategy[
                    "source_class"
                ],
            )
        )

    # ========================================================
    # 10. BUILD REPORTS
    # ========================================================

    reports = {
        name: pd.DataFrame(rows)
        for name, rows in result.items()
    }

    expected = {
        "summary": 2 * len(SOURCES),

        "error_confidence": 4 * len(SOURCES),

        "paired_changes": 2 * len(SOURCES),

        "score_distribution": (
            2 * len(SOURCES)
            * 2 * (len(SCORE_BINS) - 1)
        ),

        "error_profiles": (
            2 * len(SOURCES)
            * 2 * len(FEATURES)
        ),

        "feature_shift": (
            len(SOURCES)
            * 2 * len(FEATURES)
        ),
    }

    for name, count in expected.items():

        if len(reports[name]) != count:

            raise ValueError(
                f"Incomplete report {name}: "
                f"{len(reports[name])} != {count}"
            )

    return reports


# ============================================================
# 11. SAVE REPORTS
# ============================================================

def save_reports(
    reports: dict[str, pd.DataFrame],
) -> None:

    if set(reports) != set(REPORT_FILES):
        raise ValueError(
            "Incomplete reports"
        )

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
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

        target = REPORT_DIR / filename

        reports[name].to_csv(
            target,
            index=False,
        )

        print(
            f"Saved: {target}"
        )


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:

    print(
        "PHISHGUARD AI - PAIRED CANDIDATE FAILURE DIAGNOSTICS"
    )

    print(
        "Training split only; "
        "no validation/test access or model export."
    )

    print(
        "Source-held-out folds are exploratory, "
        "not independent validation.\n"
    )

    dataframe = load_training_data()

    parameters = load_best_parameters()

    print(
        f"Training URLs: {len(dataframe):,}\n"
    )

    reports = run_experiment(
        dataframe,
        parameters,
    )

    save_reports(
        reports
    )

    print(
        "\nERROR SUMMARY (SOURCE HELD OUT)"
    )

    print(
        reports["summary"][
            [
                "evaluation_source",
                "strategy",
                "evaluation_samples",
                "false_positives",
                "false_negatives",
                "fpr",
                "recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nPAIRED CHANGES: UNIFORM -> SOURCE_CLASS"
    )

    print(
        reports["paired_changes"].to_string(
            index=False
        )
    )

    print(
        "\nFOCAL ERROR CONFIDENCE"
    )

    focal = reports[
        "error_confidence"
    ]

    focal = focal.loc[
        (
            (focal["evaluation_source"] == "phishbd")
            &
            (focal["error_type"] == "FN")
        )
        |
        (
            (focal["evaluation_source"] == "url_phish")
            &
            (focal["error_type"] == "FP")
        )
    ]

    print(
        focal.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        "\nTOP ERROR FEATURE CONTRASTS "
        "(STANDARDIZED, DESCRIPTIVE)"
    )

    for source, kind in (
        ("phishbd", "FN"),
        ("url_phish", "FP"),
    ):

        for strategy in STRATEGIES:

            profile = reports[
                "error_profiles"
            ]

            subset = profile.loc[
                (
                    profile["evaluation_source"]
                    == source
                )
                &
                (
                    profile["error_type"]
                    == kind
                )
                &
                (
                    profile["strategy"]
                    == strategy
                )
            ]

            subset = (
                subset.sort_values(
                    "absolute_difference",
                    ascending=False,
                )
                .head(8)
            )

            print(
                f"\n{source} / {strategy} / {kind}"
            )

            print(
                subset[
                    [
                        "feature",
                        "target_samples",
                        "reference_samples",
                        "standardized_difference",
                        "constant_nonoverlap",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda value: f"{value:.3f}",
                )
            )

    print(
        "\nNo trained model was saved or deployed. "
        "Reports contain aggregates only."
    )


if __name__ == "__main__":
    main()