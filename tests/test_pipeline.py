"""Pipeline guards: schema, leakage, fold hygiene and submission formats.

    python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import MODEL_FEATURES, Cleaner, build_datasets, daily_market_index, load_train, load_validation, remove_outliers
from src.evaluate import FOLDS, fold_data, held_out_cities


@pytest.fixture(scope="module")
def datasets() -> dict[str, pd.DataFrame]:
    return build_datasets(final=True)


# --- Leakage ---------------------------------------------------------------------------
def test_quote_signal_is_never_a_feature():
    assert not set(C.LEAKY_COLUMNS) & set(MODEL_FEATURES)


def test_market_index_uses_inputs_only():
    """The daily market_index series must not depend on the target."""
    train = load_train()
    shuffled = train.assign(**{C.TARGET: train[C.TARGET].sample(frac=1, random_state=0).to_numpy()})
    a = daily_market_index(train, load_validation())
    b = daily_market_index(shuffled, load_validation())
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("name", ["wf1", "wf2", "wf3"])
def test_walk_forward_folds_train_strictly_before_test(name):
    train, test = fold_data(name)
    assert train["date"].max() < test["date"].min()
    assert train["date"].max() <= pd.Timestamp(FOLDS[name].train_end)


def test_city_holdout_removes_every_training_row_touching_held_out_cities():
    held = set(held_out_cities(load_train()))
    train, test = fold_data("city")
    assert not (train["pickup"].isin(held) | train["delivery"].isin(held)).any()
    assert (test["pickup"].isin(held) | test["delivery"].isin(held)).all()


def test_cleaner_learns_from_training_part_only():
    """Weight medians must come from the rows the Cleaner was fit on, not from later data."""
    raw = load_train()
    early = raw[raw["date"] <= C.TRAIN_END]
    market = daily_market_index(raw, load_validation())
    late = raw[raw["date"] > C.TRAIN_END].assign(weight=np.nan)  # force the median fill
    filled = Cleaner(market).fit(early).transform(late)
    expected = early.assign(weight=early["weight"].abs()).groupby("equipment")["weight"].median()
    got = filled.groupby("equipment")["weight"].first()
    np.testing.assert_allclose(got.sort_index(), expected.clip(5_000, 47_500).sort_index())


# --- Cleaning --------------------------------------------------------------------------
def test_shapes_and_no_missing_model_features(datasets):
    assert len(datasets["validation"]) == 12_000
    assert len(datasets["december"]) == 31
    for name in ("train", "validation", "december"):
        assert not datasets[name][MODEL_FEATURES].isna().any().any(), name


def test_weights_positive_and_clipped(datasets):
    for name in ("train", "validation", "december"):
        w = datasets[name]["weight"]
        assert (w >= 5_000).all() and (w <= 47_500).all(), name


def test_outliers_removed_from_training_rows_only(datasets):
    assert not datasets["train"]["is_outlier"].astype(bool).any()
    assert len(remove_outliers(datasets["train"])) == len(datasets["train"])
    _, test = fold_data("wf3")
    assert test["is_outlier"].sum() > 0  # test folds keep outlier rows (reported separately)


def test_december_gets_coordinates_from_city_table(datasets):
    dec = datasets["december"]
    assert dec[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].nunique().eq(1).all()


# --- Submission files ------------------------------------------------------------------
def test_validation_predictions_format():
    if not C.PREDICTIONS_PATH.exists():
        pytest.skip("run `python -m src.predict` first")
    pred = pd.read_csv(C.PREDICTIONS_PATH)
    template = pd.read_csv(C.TEMPLATE_PATH)
    assert list(pred.columns) == ["load_id", "predicted_rate"]
    assert pred["load_id"].tolist() == template["load_id"].tolist()
    assert np.isfinite(pred["predicted_rate"]).all() and (pred["predicted_rate"] > 0).all()


def test_december_predictions_format():
    dec = pd.read_csv(C.DECEMBER_PATH)
    original = pd.read_csv(C.DECEMBER_ORIGINAL_PATH)
    assert list(dec.columns) == list(C.DECEMBER_COLUMNS)
    pd.testing.assert_frame_equal(dec.drop(columns="predicted_rate"), original.drop(columns="predicted_rate"))
    if dec["predicted_rate"].isna().all():
        pytest.skip("run `python -m src.predict` first")
    assert (dec["predicted_rate"] > 0).all()
