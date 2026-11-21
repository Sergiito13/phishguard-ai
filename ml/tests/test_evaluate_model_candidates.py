"""Synthetic unit tests for the Extended V2 candidate comparison.

No real datasets or saved model artifacts are used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import evaluate_model_candidates as experiment


@pytest.fixture
def datasets():

    def create(split):

        records = []

        for source_index, source in enumerate(experiment.SOURCES):

            for label in (0, 1):

                for index in range(5):

                    records.append({
                        "url": (
                            f"https://{split}-{source}-"
                            f"{label}-{index}.example/login"
                        ),
                        "sources": source,
                        "group_key": (
                            f"domain:{split}-{source}-"
                            f"{label}-{index}.example"
                        ),
                        "label": label,
                        "is_https": index % 2,

                        **{
                            name: float(
                                source_index
                                + label * 2
                                + index * 0.2
                                + fi * 0.01
                            )
                            for fi, name
                            in enumerate(experiment.FEATURES)
                        },
                    })

        return pd.DataFrame(records)

    return create("train"), create("validation")


class FakeModel:

    classes_ = np.array([0, 1])
    fits = []

    def fit(self, X, y, sample_weight=None):

        assert set(np.unique(y)) == {0, 1}

        self.__class__.fits.append({
            "rows": len(X),
            "weighted": sample_weight is not None,
            "weight_len": (
                len(sample_weight)
                if sample_weight is not None
                else 0
            ),
        })

        return self

    def predict_proba(self, X):

        score = np.clip(
            0.08 + np.asarray(X)[:, 0] * 0.11,
            0.01,
            0.99,
        )

        return np.column_stack([
            1 - score,
            score,
        ])


@pytest.fixture
def reports(monkeypatch, datasets):

    FakeModel.fits = []

    monkeypatch.setattr(
        experiment,
        "create_model",
        lambda params: FakeModel(),
    )

    train, validation = datasets

    generated = experiment.run_experiment(
        train,
        validation,
        {"extended_v2": {}},
    )

    assert len(FakeModel.fits) == 10

    return generated


def test_configuration():

    assert len(experiment.FEATURES) == 31
    assert len(experiment.SOURCES) == 4

    assert experiment.STRATEGIES == (
        "uniform",
        "source_class",
    )

    assert len(experiment.REPORT_FILES) == 4


def test_inputs_valid(datasets):

    experiment.validate_inputs(
        *datasets,
        {"extended_v2": {}},
    )


def test_missing_parameters(datasets):

    with pytest.raises(ValueError, match="optimized"):

        experiment.validate_inputs(
            *datasets,
            {},
        )


def test_missing_features(datasets):

    train, validation = datasets

    with pytest.raises(
        ValueError,
        match="Missing training columns",
    ):

        experiment.validate_inputs(
            train.drop(columns=experiment.FEATURES[0]),
            validation,
            {"extended_v2": {}},
        )


def test_nonfinite_features(datasets):

    train, validation = datasets
    train = train.copy()

    train.loc[0, experiment.FEATURES[0]] = np.inf

    with pytest.raises(
        ValueError,
        match="Invalid training feature values",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_invalid_labels(datasets):

    train, validation = datasets
    train = train.copy()

    train.loc[0, "label"] = 2

    with pytest.raises(
        ValueError,
        match="binary classes",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_invalid_protocol(datasets):

    train, validation = datasets
    validation = validation.copy()

    validation.loc[0, "is_https"] = 3

    with pytest.raises(
        ValueError,
        match="HTTPS flags",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_missing_groups(datasets):

    train, validation = datasets
    train = train.copy()

    train.loc[0, "group_key"] = ""

    with pytest.raises(
        ValueError,
        match="group keys",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_group_leakage(datasets):

    train, validation = datasets
    validation = validation.copy()

    validation.loc[0, "group_key"] = train.loc[
        0,
        "group_key",
    ]

    with pytest.raises(
        ValueError,
        match="Group leakage",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_url_leakage(datasets):

    train, validation = datasets
    validation = validation.copy()

    validation.loc[0, "url"] = train.loc[0, "url"]

    with pytest.raises(
        ValueError,
        match="URL leakage",
    ):

        experiment.validate_inputs(
            train,
            validation,
            {"extended_v2": {}},
        )


def test_score_model_valid():

    scores = experiment.score_model(
        FakeModel(),
        np.array([[2.0], [4.0]]),
    )

    assert scores.shape == (2,)

    assert np.all(
        (scores >= 0) & (scores <= 1)
    )


def test_score_model_wrong_classes():

    class WrongModel(FakeModel):
        classes_ = np.array([1, 0])

    with pytest.raises(
        ValueError,
        match="classes",
    ):

        experiment.score_model(
            WrongModel(),
            np.array([[3.0]]),
        )


def test_score_model_nan():

    class NanModel(FakeModel):

        def predict_proba(self, X):
            return np.array([[np.nan, np.nan]])

    with pytest.raises(
        ValueError,
        match="probabilities",
    ):

        experiment.score_model(
            NanModel(),
            np.array([[3.0]]),
        )


def test_score_model_bad_shape():

    class BrokenModel(FakeModel):

        def predict_proba(self, X):
            return np.array([[0.2]])

    with pytest.raises(
        ValueError,
        match="shape",
    ):

        experiment.score_model(
            BrokenModel(),
            np.array([[3.0]]),
        )


def test_evaluation_metrics():

    metrics = experiment.evaluate_scores(
        [0, 0, 1, 1],
        [0.10, 0.60, 0.80, 0.30],
    )

    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)

    assert 0 <= metrics["roc_auc"] <= 1


def test_ten_fits_and_weighting(reports):

    assert len(FakeModel.fits) == 10

    assert sum(
        row["weighted"]
        for row in FakeModel.fits
    ) == 5

    assert all(
        row["rows"] == row["weight_len"]
        for row in FakeModel.fits
        if row["weighted"]
    )

    assert len(reports["folds"]) == 10
    assert len(reports["weights"]) == 10


def test_reports_have_expected_shape(reports):

    assert set(reports) == set(
        experiment.REPORT_FILES
    )

    assert len(reports["validation_sources"]) == 8
    assert len(reports["comparison"]) == 2

    assert set(
        reports["folds"]["evaluation"]
    ) == {
        "overall",
        *experiment.SOURCES,
    }

    assert set(
        reports["folds"]["strategy"]
    ) == set(experiment.STRATEGIES)


def test_fold_evaluation_counts(reports):

    results = reports["folds"]

    assert (
        results.loc[
            results["evaluation"] == "overall",
            "evaluation_samples",
        ] == 40
    ).all()

    assert (
        results.loc[
            results["evaluation"] != "overall",
            "evaluation_samples",
        ] == 10
    ).all()


def test_weights_are_finite(reports):

    assert np.isfinite(
        reports["weights"]["mean_weight"]
    ).all()

    assert np.isfinite(
        reports["weights"]["max_weight"]
    ).all()

    assert (
        reports["weights"]["mean_weight"] > 0
    ).all()

    assert (
        reports["weights"]["max_weight"] > 0
    ).all()


def test_macro_summary_and_deltas(reports):

    summary = reports[
        "comparison"
    ].set_index("strategy")

    assert summary.loc[
        "uniform",
        "delta_macro_f1_vs_uniform",
    ] == pytest.approx(0)

    assert summary.loc[
        "uniform",
        "delta_macro_fpr_vs_uniform",
    ] == pytest.approx(0)

    assert summary[
        "worst_source_fpr"
    ].between(0, 1).all()

    assert summary[
        "worst_source_recall"
    ].between(0, 1).all()


def test_incomplete_summary_rejected(reports):

    frame = reports["folds"].iloc[:-1]

    with pytest.raises(
        ValueError,
        match="Incomplete",
    ):

        experiment.summarize_results(frame)


def test_duplicate_summary_rejected(reports):

    frame = pd.concat([
        reports["folds"],
        reports["folds"].iloc[[0]],
    ])

    with pytest.raises(
        ValueError,
        match="Incomplete",
    ):

        experiment.summarize_results(frame)


def test_no_sensitive_columns(reports):

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    assert all(
        forbidden.isdisjoint(frame.columns)
        for frame in reports.values()
    )


def test_save_reports(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.save_reports(reports)

    assert len(
        list(
            tmp_path.glob("model_candidates_*.csv")
        )
    ) == 4

    assert len(
        pd.read_csv(
            tmp_path
            / experiment.REPORT_FILES["comparison"]
        )
    ) == 2


def test_reject_private_report(
    reports,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    bad = {
        name: frame.copy()
        for name, frame in reports.items()
    }

    bad["comparison"]["url"] = (
        "http://private.example"
    )

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        experiment.save_reports(bad)

    assert not list(
        tmp_path.glob("*.csv")
    )


def test_main_with_synthetic_data(
    datasets,
    tmp_path,
    monkeypatch,
    capsys,
):

    monkeypatch.setattr(
        experiment,
        "load_datasets",
        lambda: datasets,
    )

    monkeypatch.setattr(
        experiment,
        "load_best_parameters",
        lambda: {"extended_v2": {}},
    )

    monkeypatch.setattr(
        experiment,
        "create_model",
        lambda params: FakeModel(),
    )

    monkeypatch.setattr(
        experiment,
        "REPORT_DIR",
        tmp_path,
    )

    experiment.main()

    output = capsys.readouterr().out

    assert "CANDIDATE COMPARISON" in output

    assert len(
        list(tmp_path.glob("*.csv"))
    ) == 4