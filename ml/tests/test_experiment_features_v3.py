"""Synthetic unit tests for V3 split generation and paired candidate evaluation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import experiment_features_v3 as exp

from extract_features_v2 import (
    FEATURE_COLUMNS_V2,
    extract_url_features_v2,
)

from extract_features_v3 import (
    FEATURE_COLUMNS_EXTENDED_V3,
    extract_url_features_v3,
)


# ============================================================
# 1. SYNTHETIC DATA
# ============================================================

@pytest.fixture
def fake_training():

    rows = []

    for si, source in enumerate(exp.SOURCES):

        for label in (0, 1):

            for i in range(5):

                url = (
                    f"https://{source}-{label}-{i}"
                    f".example.com/login?id={si}{i}"
                )

                rows.append({
                    "url": url,
                    "label": label,
                    "sources": source,

                    "group_key": (
                        f"domain:{source}-{label}-{i}"
                        ".example.com"
                    ),

                    "is_https": 1,

                    **{
                        feature: float(
                            si
                            + label * 4
                            + i / 5
                            + fi / 100
                        )

                        for fi, feature in enumerate(
                            FEATURE_COLUMNS_EXTENDED_V3
                        )
                    },
                })

    return pd.DataFrame(rows)


# ============================================================
# 2. FAKE MODEL
# ============================================================

class FakeModel:

    classes_ = np.array([0, 1])

    fits = []

    def fit(
        self,
        x,
        y,
        sample_weight=None,
    ):

        assert set(y) == {0, 1}

        if sample_weight is not None:

            assert len(sample_weight) == len(y)

            assert np.isfinite(
                sample_weight
            ).all()

        type(self).fits.append(
            (
                x.shape[1],
                sample_weight is not None,
            )
        )

        return self

    def predict_proba(self, x):

        scores = np.clip(
            0.05
            + x[:, 0] * 0.09
            + (x.shape[1] == 37) * 0.01,
            0.001,
            0.999,
        )

        return np.column_stack([
            1 - scores,
            scores,
        ])


# ============================================================
# 3. EXPERIMENT FIXTURE
# ============================================================

@pytest.fixture
def report(
    fake_training,
    monkeypatch,
):

    FakeModel.fits = []

    monkeypatch.setattr(
        exp,
        "create_model",
        lambda params: FakeModel(),
    )

    return exp.compare_candidates(
        fake_training,
        {"extended_v2": {}},
    )


# ============================================================
# 4. CONFIGURATION TESTS
# ============================================================

def test_configuration():

    assert len(
        FEATURE_COLUMNS_V2
    ) == 31

    assert len(
        FEATURE_COLUMNS_EXTENDED_V3
    ) == 37

    assert exp.STRATEGIES == (
        "uniform",
        "source_class",
    )

    assert len(
        exp.REPORT_FILES
    ) == 3


# ============================================================
# 5. DATASET METADATA TESTS
# ============================================================

def test_missing_v2_file(
    tmp_path,
):

    with pytest.raises(
        FileNotFoundError
    ):

        exp.read_metadata(
            tmp_path / "missing.csv"
        )


def test_metadata_rejects_duplicate_urls(
    tmp_path,
):

    file = tmp_path / "train.csv"

    pd.DataFrame([
        {
            "url": "https://a.example",
            "label": 0,
            "group_key": "g1",
            "sources": "phishbd",
        },
        {
            "url": "https://a.example",
            "label": 1,
            "group_key": "g2",
            "sources": "phishbd",
        },
    ]).to_csv(
        file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate URLs",
    ):

        exp.read_metadata(file)


def test_metadata_rejects_missing_group(
    tmp_path,
):

    file = tmp_path / "train.csv"

    pd.DataFrame([
        {
            "url": "https://a.example",
            "label": 0,
            "group_key": "",
            "sources": "phishbd",
        },
        {
            "url": "https://b.example",
            "label": 1,
            "group_key": "g2",
            "sources": "phishbd",
        },
    ]).to_csv(
        file,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="group keys",
    ):

        exp.read_metadata(file)


# ============================================================
# 6. V3 DATASET GENERATION TESTS
# ============================================================

def test_process_split_preserves_metadata_and_v2(
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        exp,
        "CHUNK_SIZE",
        1,
    )

    source = tmp_path / "source.csv"

    dest = (
        tmp_path / "out" / "train.csv"
    )

    urls = [
        "https://example.com/login?a=abcd",
        "http://shop.example.org/a1/b23",
    ]

    frame = pd.DataFrame([
        {
            "url": url,
            "label": label,
            "sources": "phishbd",

            "group_key": (
                f"domain:{label}.example"
            ),

            "is_https": int(
                url.startswith("https")
            ),

            **extract_url_features_v2(url),
        }

        for label, url in enumerate(urls)
    ])

    frame.to_csv(
        source,
        index=False,
    )

    assert exp.process_split(
        source,
        dest,
        2,
    ) == 2

    generated = pd.read_csv(dest)

    assert generated[
        "url"
    ].tolist() == urls

    assert len(generated) == 2

    # Verify original V2 features.

    for feature in FEATURE_COLUMNS_V2:

        assert np.allclose(
            frame[feature],
            generated[feature],
        )

    # Verify new V3 features.

    for i, url in enumerate(urls):

        for name, value in (
            extract_url_features_v3(url).items()
        ):

            assert (
                generated.iloc[i][name]
                == pytest.approx(value)
            )

    # Verify temporary files are removed.

    assert not (
        dest.with_suffix(".csv.tmp")
    ).exists()


# ============================================================
# 7. FILE SAFETY TESTS
# ============================================================

def test_failed_process_preserves_old_file(
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        exp,
        "CHUNK_SIZE",
        1,
    )

    source = tmp_path / "source.csv"

    target = tmp_path / "destination.csv"

    target.write_text(
        "preserved",
        encoding="utf-8",
    )

    pd.DataFrame([
        {
            "url": "https://example.com"
        }
    ]).to_csv(
        source,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Missing V2",
    ):

        exp.process_split(
            source,
            target,
            1,
        )

    assert target.read_text(
        encoding="utf-8"
    ) == "preserved"

    assert not (
        target.with_suffix(".csv.tmp")
    ).exists()


def test_process_rejects_same_path(
    tmp_path,
):

    path = tmp_path / "same.csv"

    with pytest.raises(
        ValueError,
        match="cannot be the same",
    ):

        exp.process_split(
            path,
            path,
            1,
        )


# ============================================================
# 8. MODEL CONFIGURATION TESTS
# ============================================================

def test_compare_needs_parameters(
    fake_training,
):

    with pytest.raises(
        ValueError,
        match="optimized",
    ):

        exp.compare_candidates(
            fake_training,
            {},
        )


def test_compare_rejects_missing_feature(
    fake_training,
):

    with pytest.raises(
        ValueError,
        match="Missing evaluation",
    ):

        exp.compare_candidates(
            fake_training.drop(
                columns="v3_query_value_entropy"
            ),

            {"extended_v2": {}},
        )


# ============================================================
# 9. MODEL TRAINING TESTS
# ============================================================

def test_compare_sixteen_fits(report):

    assert len(
        FakeModel.fits
    ) == 16

    assert sum(
        weighted
        for _, weighted in FakeModel.fits
    ) == 8

    assert sum(
        count == 31
        for count, _ in FakeModel.fits
    ) == 8

    assert sum(
        count == 37
        for count, _ in FakeModel.fits
    ) == 8


# ============================================================
# 10. REPORT STRUCTURE TESTS
# ============================================================

def test_compare_reports_shape(
    report,
):

    assert set(
        report
    ) == set(
        exp.REPORT_FILES
    )

    assert len(
        report["source_results"]
    ) == 16

    assert len(
        report["paired"]
    ) == 8

    assert len(
        report["summary"]
    ) == 4


# ============================================================
# 11. PAIRED METRICS TESTS
# ============================================================

def test_paired_deltas_consistent(
    report,
):

    paired = report["paired"]

    assert np.allclose(
        paired["delta_f1"],

        paired["v3_f1"]
        - paired["v2_f1"],
    )

    assert np.allclose(
        paired["delta_fpr"],

        paired["v3_fpr"]
        - paired["v2_fpr"],
    )

    assert np.allclose(
        paired["delta_recall"],

        paired["v3_recall"]
        - paired["v2_recall"],
    )


# ============================================================
# 12. PRIVACY TESTS
# ============================================================

def test_no_sensitive_columns(
    report,
):

    forbidden = {
        "url",
        "urls",
        "sources",
        "group_key",
        "hostname",
        "domain",
    }

    assert all(
        forbidden.isdisjoint(
            frame.columns
        )

        for frame in report.values()
    )


# ============================================================
# 13. REPORT SAVING TESTS
# ============================================================

def test_save_reports(
    report,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        exp,
        "REPORT_DIR",
        tmp_path,
    )

    exp.save_reports(report)

    assert len(
        list(
            tmp_path.glob("v3_*.csv")
        )
    ) == 3


def test_reject_sensitive_report(
    report,
    tmp_path,
    monkeypatch,
):

    monkeypatch.setattr(
        exp,
        "REPORT_DIR",
        tmp_path,
    )

    altered = {
        key: frame.copy()

        for key, frame in report.items()
    }

    altered[
        "summary"
    ]["url"] = "https://private.example"

    with pytest.raises(
        ValueError,
        match="sensitive",
    ):

        exp.save_reports(altered)

    assert not list(
        tmp_path.glob("*.csv")
    )