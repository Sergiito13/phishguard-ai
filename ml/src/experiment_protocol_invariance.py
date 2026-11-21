"""
PhishGuard AI: HTTP/HTTPS counterfactual sensitivity diagnostics.

Training split only. Eight optimized leave-one-source-out model fits with
registered-domain isolation. The holdout is probed with swapped schemes; it is
NEVER used as labeled augmented training data. Original features are checked
against freshly extracted ones before comparing scores.

Reports contain aggregated numerical information, no URLs or domain groups.
This exploratory analysis is NOT independent validation or evidence that
HTTP and HTTPS URLs necessarily share the same real-world phishing label.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from evaluate_optimized_models import load_best_parameters

from experiment_hyperparameters import (
    FEATURE_SETS,
    create_model,
    create_source_folds,
    load_training_data,
)

from experiment_training_strategies import (
    SOURCES,
    belongs_to_source,
    verify_group_isolation,
)

from extract_features import extract_url_features
from extract_features_v2 import extract_url_features_v2


# ============================================================
# 1. CONFIGURATION
# ============================================================

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

MODELS = (
    "hybrid_v1_extended",
    "extended_v2",
)

THRESHOLD = 0.50

MAX_PER_LABEL_AND_SOURCE = 800

RANDOM_STATE = 42

SCHEME_RE = re.compile(
    r"^(https?)://",
    flags=re.IGNORECASE,
)

REPORT_FILES = {
    "detail": "protocol_invariance_detail.csv",
    "summary": "protocol_invariance_summary.csv",
    "feature_changes": "protocol_invariance_feature_changes.csv",
}


# ============================================================
# 2. SWAP HTTP / HTTPS
# ============================================================

def swap_scheme(url: str) -> tuple[str, str]:
    """
    Flip only the leading HTTP(S) scheme,
    preserving the URL remainder.
    """

    if not isinstance(url, str):
        raise ValueError(
            "URL must begin with http:// or https://"
        )

    match = SCHEME_RE.match(url)

    if match is None:
        raise ValueError(
            "URL must begin with http:// or https://"
        )

    previous = match.group(1).lower()

    replacement = (
        "https"
        if previous == "http"
        else "http"
    )

    return (
        replacement + url[match.end(1):],
        previous,
    )


# ============================================================
# 3. EXTRACT FEATURE VECTOR
# ============================================================

def extract_vector(
    url: str,
    features: list[str],
) -> np.ndarray:
    """
    Use the existing V1/V2 feature extractors.

    Preserve the exact feature order expected
    by the trained classifier.
    """

    values = {}

    requires_v1 = any(
        not feature.startswith("v2_")
        for feature in features
    )

    requires_v2 = any(
        feature.startswith("v2_")
        for feature in features
    )

    if requires_v1:
        values.update(
            extract_url_features(url)
        )

    if requires_v2:
        values.update(
            extract_url_features_v2(url)
        )

    try:

        array = np.asarray(
            [
                values[feature]
                for feature in features
            ],
            dtype=np.float64,
        )

    except KeyError as exc:

        raise ValueError(
            "Feature extractor omitted a required feature"
        ) from exc

    if not np.isfinite(array).all():
        raise ValueError(
            "Non-finite extracted features"
        )

    return array


# ============================================================
# 4. EXTRACT FEATURE MATRIX
# ============================================================

def extract_matrix(
    urls,
    features: list[str],
) -> np.ndarray:
    """
    Extract ordered numerical features
    from multiple URLs.
    """

    return np.vstack([
        extract_vector(url, features)
        for url in urls
    ])


# ============================================================
# 5. SELECT PROBE SAMPLES
# ============================================================

def select_probe_rows(
    frame: pd.DataFrame,
    max_per_label: int = MAX_PER_LABEL_AND_SOURCE,
) -> pd.DataFrame:
    """
    Select reproducible samples per label.

    Include only URLs with explicit
    HTTP(S) schemes.
    """

    if max_per_label < 1:
        raise ValueError(
            "max_per_label must be >= 1"
        )

    eligible = frame.loc[
        frame["url"].str.match(
            SCHEME_RE,
            na=False,
        )
    ]

    subsets = []

    for label in (0, 1):

        subset = eligible.loc[
            eligible["label"] == label
        ]

        if subset.empty:
            raise ValueError(
                "Each class requires explicit HTTP(S) URLs"
            )

        selected = subset.sample(
            n=min(
                len(subset),
                max_per_label,
            ),
            random_state=RANDOM_STATE,
        )

        subsets.append(selected)

    return (
        pd.concat(subsets)
        .sort_index()
        .copy()
    )


# ============================================================
# 6. MODEL SCORES
# ============================================================

def checked_scores(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Obtain phishing probabilities.

    Class 1 represents phishing.
    """

    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=np.float64,
    )

    if probabilities.shape != (len(X), 2):
        raise ValueError(
            "Invalid probability matrix shape"
        )

    scores = probabilities[:, 1]

    if (
        not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError(
            "Invalid phishing probabilities"
        )

    return scores


# ============================================================
# 7. PROBABILITY CHANGE METRICS
# ============================================================

def change_metrics(
    before: np.ndarray,
    after: np.ndarray,
) -> dict:
    """
    Calculate prediction sensitivity.

    This does not evaluate counterfactual accuracy.
    """

    before = np.asarray(
        before,
        dtype=np.float64,
    )

    after = np.asarray(
        after,
        dtype=np.float64,
    )

    if (
        before.ndim != 1
        or after.shape != before.shape
        or not len(before)
    ):
        raise ValueError(
            "Expected non-empty aligned 1D score arrays"
        )

    if (
        not np.isfinite(before).all()
        or not np.isfinite(after).all()
    ):
        raise ValueError(
            "Non-finite scores"
        )

    delta = after - before

    original_predictions = (
        before >= THRESHOLD
    )

    changed_predictions = (
        after >= THRESHOLD
    )

    flips = (
        original_predictions
        != changed_predictions
    )

    return {
        "samples": len(before),

        "mean_original_score": float(
            before.mean()
        ),

        "mean_changed_score": float(
            after.mean()
        ),

        "mean_signed_delta": float(
            delta.mean()
        ),

        "mean_absolute_delta": float(
            np.abs(delta).mean()
        ),

        "p90_absolute_delta": float(
            np.quantile(
                np.abs(delta),
                0.90,
            )
        ),

        "flipped_predictions": int(
            flips.sum()
        ),

        "flip_fraction": float(
            flips.mean()
        ),

        "large_score_changes": int(
            (np.abs(delta) >= 0.10).sum()
        ),
    }


# ============================================================
# 8. RUN EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    params: dict,
    max_per_label: int = MAX_PER_LABEL_AND_SOURCE,
) -> dict[str, pd.DataFrame]:
    """
    Evaluate protocol sensitivity.

    Models:
        - Hybrid V1 Extended.
        - Extended V2.

    Evaluation:
        - Leave-one-source-out.
        - Domain group isolation.

    Interventions:
        - Full HTTP/HTTPS scheme swap.
        - HTTPS flag only (when available).

    Total:
        - 8 model fits.
    """

    if not set(MODELS).issubset(params):
        raise ValueError(
            "Missing optimized parameters"
        )

    if any(
        name not in FEATURE_SETS
        for name in MODELS
    ):
        raise ValueError(
            "Missing model feature definitions"
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
            "Missing or duplicate source holdout folds"
        )

    detail = []

    feature_changes = []

    for model_name in MODELS:

        features = list(
            FEATURE_SETS[model_name]
        )

        missing = (
            set(features)
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Training data has missing model features"
            )

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        for fold in folds:

            source = fold["source"]

            # ------------------------------------------------
            # A. TRAINING AND HOLDOUT DATA
            # ------------------------------------------------

            fit = dataframe.iloc[
                fold["training_indices"]
            ]

            held = dataframe.iloc[
                fold["validation_indices"]
            ]

            # Verify domain isolation.

            verify_group_isolation(
                fit,
                held,
            )

            # Verify source isolation.

            if belongs_to_source(
                fit,
                source,
            ).any():

                raise ValueError(
                    "Source leakage detected"
                )

            # ------------------------------------------------
            # B. SELECT SAMPLES
            # ------------------------------------------------

            selected = select_probe_rows(
                held,
                max_per_label,
            )

            y_fit = fit[
                "label"
            ].to_numpy(
                dtype=int
            )

            if set(np.unique(y_fit)) != {0, 1}:
                raise ValueError(
                    "Training requires both classes"
                )

            X_fit = fit[
                features
            ].to_numpy(
                dtype=np.float64
            )

            X_original = selected[
                features
            ].to_numpy(
                dtype=np.float64
            )

            original_urls = selected[
                "url"
            ].tolist()

            # ------------------------------------------------
            # C. VERIFY ORIGINAL FEATURES
            # ------------------------------------------------

            original_recomputed = extract_matrix(
                original_urls,
                features,
            )

            # Check that the stored values are consistent
            # with the current feature extractors.

            matches = np.isclose(
                X_original,
                original_recomputed,
                rtol=1e-8,
                atol=1e-8,
            )

            if not matches.all():

                mismatched = int(
                    np.count_nonzero(~matches)
                )

                raise ValueError(
                    "Feature reproducibility check failed "
                    f"({mismatched} values)"
                )

            # ------------------------------------------------
            # D. NUMERICAL CONSISTENCY CORRECTION
            # ------------------------------------------------

            # IMPORTANT:
            #
            # Use the same extractor for both sides
            # of the comparison.
            #
            # CSV floating-point serialization may
            # introduce tiny differences.
            #
            # Such differences can cross tree decision
            # boundaries and produce false indications
            # of protocol sensitivity.

            X_original = original_recomputed

            # ------------------------------------------------
            # E. CREATE COUNTERFACTUAL URLS
            # ------------------------------------------------

            swapped = [
                swap_scheme(url)
                for url in original_urls
            ]

            swapped_urls = [
                url
                for url, _ in swapped
            ]

            original_schemes = np.array([
                scheme
                for _, scheme in swapped
            ])

            X_swapped = extract_matrix(
                swapped_urls,
                features,
            )

            # ------------------------------------------------
            # F. VERIFY V2 PROTOCOL INDEPENDENCE
            # ------------------------------------------------

            # Extended V2 is designed to exclude
            # the HTTP/HTTPS scheme.
            #
            # Both extracted feature matrices must
            # therefore be exactly identical.

            if (
                model_name == "extended_v2"
                and not np.array_equal(
                    X_original,
                    X_swapped,
                )
            ):
                raise ValueError(
                    "V2 protocol independence violated"
                )

            # ------------------------------------------------
            # G. TRAIN MODEL
            # ------------------------------------------------

            model = create_model(
                params[model_name]
            )

            model.fit(
                X_fit,
                y_fit,
            )

            original_scores = checked_scores(
                model,
                X_original,
            )

            # ------------------------------------------------
            # H. BUILD INTERVENTIONS
            # ------------------------------------------------

            probes = {
                "scheme_swap": X_swapped,
            }

            if "is_https" in features:

                X_flag_only = X_original.copy()

                flag_index = features.index(
                    "is_https"
                )

                X_flag_only[:, flag_index] = (
                    1 - X_original[:, flag_index]
                )

                probes["https_flag_only"] = (
                    X_flag_only
                )

            # ------------------------------------------------
            # I. EVALUATE INTERVENTIONS
            # ------------------------------------------------

            for probe_name, X_changed in probes.items():

                changed_scores = checked_scores(
                    model,
                    X_changed,
                )

                for label in (0, 1):

                    for scheme in ("http", "https"):

                        mask = (
                            (
                                selected["label"]
                                .to_numpy()
                                == label
                            )
                            &
                            (
                                original_schemes
                                == scheme
                            )
                        )

                        if not mask.any():
                            continue

                        metrics = change_metrics(
                            original_scores[mask],
                            changed_scores[mask],
                        )

                        detail.append({
                            "feature_set": model_name,
                            "evaluation_source": source,
                            "intervention": probe_name,
                            "label": label,
                            "original_scheme": scheme,
                            **metrics,
                        })

            # ------------------------------------------------
            # J. MEASURE CHANGED FEATURES
            # ------------------------------------------------

            changed_features = ~np.isclose(
                X_original,
                X_swapped,
                rtol=1e-8,
                atol=1e-8,
            )

            for index, feature in enumerate(features):

                feature_changes.append({
                    "feature_set": model_name,

                    "evaluation_source": source,

                    "feature": feature,

                    "sampled_urls": len(selected),

                    "changed_count": int(
                        changed_features[:, index].sum()
                    ),

                    "changed_fraction": float(
                        changed_features[:, index].mean()
                    ),
                })

            # ------------------------------------------------
            # K. DISPLAY PROGRESS
            # ------------------------------------------------

            swapped_scores = checked_scores(
                model,
                X_swapped,
            )

            mean_delta = float(
                np.abs(
                    swapped_scores - original_scores
                ).mean()
            )

            print(
                f"{model_name:<20} "
                f"{source:<11} "
                f"probed={len(selected):>5,} "
                f"mean_abs_delta={mean_delta:.4f}"
            )

    # ========================================================
    # 9. BUILD REPORTS
    # ========================================================

    detailed = pd.DataFrame(
        detail
    )

    features_report = pd.DataFrame(
        feature_changes
    )

    if detailed.empty or features_report.empty:
        raise ValueError(
            "No diagnostic results produced"
        )

    # Aggregate by actual sample counts.

    summary = (
        detailed.assign(
            abs_score_sum=(
                detailed["samples"]
                * detailed["mean_absolute_delta"]
            ),

            signed_score_sum=(
                detailed["samples"]
                * detailed["mean_signed_delta"]
            ),
        )
        .groupby(
            [
                "feature_set",
                "evaluation_source",
                "intervention",
            ],
            as_index=False,
        )
        .agg(
            samples=(
                "samples",
                "sum",
            ),

            flipped_predictions=(
                "flipped_predictions",
                "sum",
            ),

            large_score_changes=(
                "large_score_changes",
                "sum",
            ),

            abs_score_sum=(
                "abs_score_sum",
                "sum",
            ),

            signed_score_sum=(
                "signed_score_sum",
                "sum",
            ),
        )
    )

    summary["flip_fraction"] = (
        summary["flipped_predictions"]
        / summary["samples"]
    )

    summary["mean_absolute_delta"] = (
        summary["abs_score_sum"]
        / summary["samples"]
    )

    summary["mean_signed_delta"] = (
        summary["signed_score_sum"]
        / summary["samples"]
    )

    summary = summary.drop(
        columns=[
            "abs_score_sum",
            "signed_score_sum",
        ]
    )

    return {
        "detail": detailed,
        "summary": summary,
        "feature_changes": features_report,
    }


# ============================================================
# 10. SAVE REPORTS
# ============================================================

def save_reports(
    reports: dict[str, pd.DataFrame],
) -> None:
    """
    Export aggregated statistics only.

    Never export URLs or domain identifiers.
    """

    if set(reports) != set(REPORT_FILES):
        raise ValueError(
            "Incomplete report set"
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
            or forbidden.intersection(
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
# 11. MAIN
# ============================================================

def main():

    print(
        "PHISHGUARD AI - HTTP/HTTPS COUNTERFACTUAL SENSITIVITY"
    )

    print(
        "Training-only leave-one-source-out; "
        "counterfactuals are not labeled data."
    )

    dataframe = load_training_data()

    params = load_best_parameters()

    print(
        f"Training URLs: {len(dataframe):,}"
    )

    reports = run_experiment(
        dataframe,
        params,
    )

    save_reports(
        reports
    )

    # --------------------------------------------------------
    # PROTOCOL SENSITIVITY SUMMARY
    # --------------------------------------------------------

    print("\nPROTOCOL SENSITIVITY SUMMARY")

    print(
        reports["summary"].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # FEATURE CHANGES
    # --------------------------------------------------------

    for model_name in MODELS:

        print(
            f"\nFEATURES CHANGED BY SCHEME SWAP: {model_name}"
        )

        subset = reports[
            "feature_changes"
        ]

        subset = subset.loc[
            subset["feature_set"] == model_name
        ]

        subset = (
            subset.groupby(
                "feature",
                as_index=False,
            )["changed_count"]
            .sum()
        )

        print(
            subset.sort_values(
                "changed_count",
                ascending=False,
            )
            .head(12)
            .to_string(index=False)
        )

    print(
        "\nDiagnostic only: changing URL scheme "
        "may change real-world meaning."
    )

    print(
        "No new training labels were generated. "
        "Validation/test were not accessed."
    )


if __name__ == "__main__":
    main()