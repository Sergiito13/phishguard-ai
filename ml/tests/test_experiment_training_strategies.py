"""
PhishGuard AI - Training Strategies Unit Tests

Tests:
- Experiment configuration
- Dataset source identification
- Group leakage prevention
- Class-balanced sample weights
- Source-and-class-balanced sample weights
- Invalid weighting strategies
- Dataset loading and validation
- Training and evaluation
- Cross-source dataset isolation
- Experiment report generation

All tests use synthetic data.

No real datasets are modified.
No trained model artifacts are overwritten.
The test dataset is never accessed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. IMPORT MODULES
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(SRC_DIR))

import experiment_training_strategies as strategies

from extract_features import FEATURE_COLUMNS


# ============================================================
# 2. SYNTHETIC DATA
# ============================================================

def make_record(
    label: int,
    sources: str,
    group_key: str,
) -> dict:
    """
    Create one synthetic URL feature record.
    """

    record = {
        "label": label,
        "sources": sources,
        "group_key": group_key,
    }

    for feature in FEATURE_COLUMNS:
        record[feature] = 1.0

    return record


@pytest.fixture
def sample_dataframe():
    """
    Synthetic dataset containing multiple sources
    and both classification labels.
    """

    return pd.DataFrame([
        make_record(0, "legitphish", "group-a"),
        make_record(0, "legitphish", "group-b"),
        make_record(0, "phiusiil", "group-c"),
        make_record(1, "phiusiil", "group-d"),
    ])


# ============================================================
# 3. EXPERIMENT CONFIGURATION
# ============================================================

def test_experiment_configuration():

    experiments = strategies.EXPERIMENTS

    assert len(experiments) == 4

    assert "baseline" in experiments
    assert "class_balanced" in experiments

    assert "source_class_balanced" in experiments
    assert "source_class_no_https" in experiments

    baseline = experiments["baseline"]

    assert baseline["features"] == FEATURE_COLUMNS
    assert baseline["weighting"] == "none"

    assert (
        experiments["class_balanced"]["weighting"]
        == "class"
    )

    assert (
        experiments["source_class_balanced"]["weighting"]
        == "source_class"
    )

    without_https = experiments[
        "source_class_no_https"
    ]

    assert without_https["weighting"] == "source_class"

    assert without_https["features"] == [
        feature
        for feature in FEATURE_COLUMNS
        if feature != "is_https"
    ]


# ============================================================
# 4. DATASET SOURCE IDENTIFICATION
# ============================================================

def test_belongs_to_source():

    df = pd.DataFrame({
        "sources": [
            "url_phish",
            "phiusiil|url_phish",
            "legitphish",
            "phishbd",
            "url_phishing",
        ],
    })

    mask = strategies.belongs_to_source(
        df,
        "url_phish",
    )

    assert mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


# ============================================================
# 5. GROUP ISOLATION
# ============================================================

def test_group_isolation_valid():

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(1, "phiusiil", "train-b"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "validation-a"),
        make_record(1, "phishbd", "validation-b"),
    ])

    strategies.verify_group_isolation(
        train_df,
        validation_df,
    )


def test_group_leakage_detection():

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "shared-group"),
    ])

    validation_df = pd.DataFrame([
        make_record(1, "url_phish", "shared-group"),
    ])

    with pytest.raises(
        ValueError,
        match="Group leakage detected",
    ):

        strategies.verify_group_isolation(
            train_df,
            validation_df,
        )


# ============================================================
# 6. BASELINE SAMPLE WEIGHTS
# ============================================================

def test_baseline_sample_weights(sample_dataframe):

    weights = strategies.calculate_sample_weights(
        sample_dataframe,
        "none",
    )

    assert weights is None


# ============================================================
# 7. CLASS-BALANCED WEIGHTS
# ============================================================

def test_class_balanced_weights(sample_dataframe):

    weights = strategies.calculate_sample_weights(
        sample_dataframe,
        "class",
    )

    labels = sample_dataframe["label"].to_numpy()

    assert len(weights) == len(sample_dataframe)

    assert np.isfinite(weights).all()

    assert np.all(weights > 0)

    # Average weight must be 1.

    assert weights.mean() == pytest.approx(1.0)

    # Both classes receive equal total weight.

    legitimate_weight = weights[
        labels == 0
    ].sum()

    phishing_weight = weights[
        labels == 1
    ].sum()

    assert legitimate_weight == pytest.approx(
        phishing_weight
    )


# ============================================================
# 8. SOURCE AND CLASS BALANCING
# ============================================================

def test_source_class_balanced_weights():

    df = pd.DataFrame([
        make_record(
            0,
            "legitphish|url_phish",
            "group-a",
        ),
        make_record(
            0,
            "legitphish",
            "group-b",
        ),
        make_record(
            0,
            "phiusiil",
            "group-c",
        ),
        make_record(
            1,
            "phiusiil",
            "group-d",
        ),
    ])

    weights = strategies.calculate_sample_weights(
        df,
        "source_class",
    )

    assert len(weights) == 4

    assert np.isfinite(weights).all()

    assert np.all(weights > 0)

    assert weights.mean() == pytest.approx(1.0)

    # The first two samples belong to the same
    # primary-source/class combination.

    assert weights[0] == pytest.approx(weights[1])

    # Each source/class combination receives
    # approximately the same total weight.

    legitphish_total = weights[0] + weights[1]

    phiusiil_legitimate = weights[2]

    phiusiil_phishing = weights[3]

    assert legitphish_total == pytest.approx(
        phiusiil_legitimate
    )

    assert legitphish_total == pytest.approx(
        phiusiil_phishing
    )


# ============================================================
# 9. INVALID WEIGHTING STRATEGY
# ============================================================

def test_invalid_weighting_strategy(sample_dataframe):

    with pytest.raises(
        ValueError,
        match="Unknown weighting strategy",
    ):

        strategies.calculate_sample_weights(
            sample_dataframe,
            "invalid_strategy",
        )


# ============================================================
# 10. DATASET LOADING
# ============================================================

def test_load_datasets(tmp_path, monkeypatch):

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(1, "phiusiil", "train-b"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "validation-a"),
        make_record(1, "phishbd", "validation-b"),
    ])

    train_file = tmp_path / "train.csv"

    validation_file = tmp_path / "validation.csv"

    train_df.to_csv(
        train_file,
        index=False,
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        strategies,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        strategies,
        "VALIDATION_FILE",
        validation_file,
    )

    loaded_train, loaded_validation = (
        strategies.load_datasets()
    )

    assert len(loaded_train) == 2

    assert len(loaded_validation) == 2

    assert set(loaded_train["label"]) == {0, 1}

    assert set(loaded_validation["label"]) == {0, 1}

    assert set(
        loaded_train["group_key"]
    ).isdisjoint(
        set(loaded_validation["group_key"])
    )


# ============================================================
# 11. INVALID DATASET INPUT
# ============================================================

@pytest.mark.parametrize(
    "field,value,error",
    [
        (
            "label",
            2,
            "Invalid labels",
        ),
        (
            "group_key",
            "",
            "Missing group keys",
        ),
        (
            "url_length",
            np.inf,
            "Invalid features",
        ),
    ],
)
def test_load_datasets_rejects_invalid_data(
    tmp_path,
    monkeypatch,
    field,
    value,
    error,
):

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(1, "phiusiil", "train-b"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "validation-a"),
        make_record(1, "phishbd", "validation-b"),
    ])

    train_df.loc[0, field] = value

    train_file = tmp_path / "train.csv"

    validation_file = tmp_path / "validation.csv"

    train_df.to_csv(train_file, index=False)

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    monkeypatch.setattr(
        strategies,
        "TRAIN_FILE",
        train_file,
    )

    monkeypatch.setattr(
        strategies,
        "VALIDATION_FILE",
        validation_file,
    )

    with pytest.raises(
        ValueError,
        match=error,
    ):

        strategies.load_datasets()


# ============================================================
# 12. MODEL TRAINING AND EVALUATION
# ============================================================

def test_run_experiment(monkeypatch):

    class FakeModel:
        """
        Controlled classifier for deterministic tests.
        """

        def __init__(self):
            self.sample_weight = None

        def fit(self, X, y, sample_weight=None):

            self.sample_weight = sample_weight

            return self

        def predict_proba(self, X):

            return np.array([
                [0.90, 0.10],
                [0.10, 0.90],
            ])

    fake_model = FakeModel()

    monkeypatch.setattr(
        strategies,
        "create_experiment_model",
        lambda: fake_model,
    )

    train_df = pd.DataFrame([
        make_record(0, "legitphish", "train-a"),
        make_record(0, "legitphish", "train-b"),
        make_record(1, "phiusiil", "train-c"),
    ])

    validation_df = pd.DataFrame([
        make_record(0, "url_phish", "val-a"),
        make_record(1, "url_phish", "val-b"),
    ])

    config = {
        "features": list(FEATURE_COLUMNS),
        "weighting": "class",
    }

    result = strategies.run_experiment(
        experiment_name="test_model",
        config=config,
        train_df=train_df,
        validation_df=validation_df,
        evaluation_source="url_phish",
    )

    assert result["experiment"] == "test_model"

    assert result["evaluation_source"] == "url_phish"

    assert result["training_samples"] == 3

    assert result["validation_samples"] == 2

    assert result["feature_count"] == len(
        FEATURE_COLUMNS
    )

    assert result["accuracy"] == 1.0

    assert result["precision"] == 1.0

    assert result["recall"] == 1.0

    assert result["f1"] == 1.0

    assert result["roc_auc"] == 1.0

    assert result["fpr"] == 0.0

    assert result["false_positives"] == 0

    assert result["false_negatives"] == 0

    # Confirm sample weights were passed
    # to the classifier.

    assert fake_model.sample_weight is not None

    assert len(fake_model.sample_weight) == 3


# ============================================================
# 13. CROSS-SOURCE EVALUATION
# ============================================================

def test_cross_source_evaluation(monkeypatch):

    train_df = pd.DataFrame([
        make_record(
            index % 2,
            source,
            f"train-{index}",
        )
        for index, source in enumerate(strategies.SOURCES)
    ])

    validation_df = pd.DataFrame([
        make_record(
            index % 2,
            source,
            f"validation-{index}",
        )
        for index, source in enumerate(strategies.SOURCES)
    ])

    # Use one configuration to test source filtering
    # without training multiple real models.

    monkeypatch.setattr(
        strategies,
        "EXPERIMENTS",
        {
            "test_strategy": {
                "features": list(FEATURE_COLUMNS),
                "weighting": "none",
            },
        },
    )

    evaluated_sources = []

    def fake_run_experiment(
        experiment_name,
        config,
        train_df,
        validation_df,
        evaluation_source,
    ):

        # The excluded source must not appear
        # in training.

        assert not strategies.belongs_to_source(
            train_df,
            evaluation_source,
        ).any()

        # Validation must contain only
        # the excluded source.

        assert strategies.belongs_to_source(
            validation_df,
            evaluation_source,
        ).all()

        evaluated_sources.append(
            evaluation_source
        )

        return {
            "experiment": experiment_name,
            "evaluation_source": evaluation_source,
        }

    monkeypatch.setattr(
        strategies,
        "run_experiment",
        fake_run_experiment,
    )

    results = strategies.evaluate_cross_source(
        train_df,
        validation_df,
    )

    assert len(results) == len(strategies.SOURCES)

    assert set(evaluated_sources) == set(
        strategies.SOURCES
    )


# ============================================================
# 14. REPORT GENERATION
# ============================================================

def test_generate_reports(tmp_path, monkeypatch):

    monkeypatch.setattr(
        strategies,
        "REPORT_DIR",
        tmp_path,
    )

    def make_result(
        experiment,
        source,
        f1,
        roc_auc,
        fpr,
    ):

        return {
            "experiment": experiment,
            "evaluation_source": source,
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "roc_auc": roc_auc,
            "fpr": fpr,
        }

    results = [
        make_result(
            "baseline",
            "overall",
            1.0,
            1.0,
            0.0,
        ),
        make_result(
            "baseline",
            "legitphish",
            0.8,
            0.9,
            0.1,
        ),
        make_result(
            "baseline",
            "url_phish",
            0.4,
            0.5,
            0.5,
        ),
        make_result(
            "weighted",
            "legitphish",
            0.7,
            0.8,
            0.2,
        ),
        make_result(
            "weighted",
            "url_phish",
            0.5,
            0.7,
            0.3,
        ),
    ]

    strategies.generate_reports(results)

    detailed_file = (
        tmp_path / "training_strategies_results.csv"
    )

    summary_file = (
        tmp_path / "training_strategies_summary.csv"
    )

    assert detailed_file.is_file()

    assert summary_file.is_file()

    detailed_df = pd.read_csv(detailed_file)

    summary_df = pd.read_csv(summary_file)

    assert len(detailed_df) == 5

    assert len(summary_df) == 2

    baseline = summary_df.loc[
        summary_df["experiment"] == "baseline"
    ].iloc[0]

    # Overall validation must be excluded
    # from cross-source macro averages.

    assert baseline["macro_f1"] == pytest.approx(0.6)

    assert baseline["macro_roc_auc"] == pytest.approx(0.7)

    assert baseline["macro_fpr"] == pytest.approx(0.3)

    assert baseline["worst_fpr"] == pytest.approx(0.5)