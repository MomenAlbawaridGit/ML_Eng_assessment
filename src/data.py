"""Data pipeline: load -> validate -> clean -> features.

One code path for train, validation and the December chart file:

    market = daily_market_index(load_train(), load_validation())   # inputs only, no labels
    cleaner = Cleaner(market).fit(train_part)                     # learns only from train_part
    train_clean = remove_outliers(cleaner.transform(train_part))  # outlier removal: train only
    test_clean = cleaner.transform(test_part)                     # keeps every row, flags outliers
    val_clean = cleaner.transform(load_validation())
    dec_clean = cleaner.transform(load_december())

Every output frame has the same columns (`OUTPUT_COLUMNS`) and `MODEL_FEATURES` never contain NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config as C

CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
NUMERIC_FEATURES = [
    "distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "market_index_detrended", "day_of_week",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
OUTPUT_COLUMNS = (
    ["load_id", "date", "source"]
    + MODEL_FEATURES
    + ["market_index_trend", "market_index_raw", "quote_signal",  # diagnostics, not features
       C.TARGET, "rate_per_mile", "outlier_reference", "outlier_ratio", "is_outlier"]
)
LANE_KEY = ["pickup", "delivery", "equipment"]


# --- Load ------------------------------------------------------------------------------
def _read(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["source"] = source
    return df


def load_train(path: Path = C.TRAIN_PATH) -> pd.DataFrame:
    """Labelled loads, Jan-Oct 2025."""
    return _read(path, "train")


def load_validation(path: Path = C.VALIDATION_PATH) -> pd.DataFrame:
    """Unlabelled loads to submit, Nov-Dec 2025 (all 12,000 rows must survive)."""
    return _read(path, "validation")


def load_december(path: Path = C.DECEMBER_PATH) -> pd.DataFrame:
    """December chart inputs, padded to the raw input schema.

    Missing inputs (coordinates, market_index, quote_signal) are NaN and filled by `Cleaner`.
    A synthetic `load_id` (DEC-YYYY-MM-DD) is added; the file's own column order is untouched.
    """
    df = _read(path, "december").drop(columns=["predicted_rate"], errors="ignore")
    df["load_id"] = "DEC-" + df["date"].dt.strftime("%Y-%m-%d")
    for col in C.RAW_INPUT_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[[*C.RAW_INPUT_COLUMNS, "source"]]


# --- Validate --------------------------------------------------------------------------
def validate(df: pd.DataFrame, name: str = "frame") -> list[str]:
    """Raise on schema errors; return a list of soft data-quality warnings (with counts)."""
    missing = [c for c in C.RAW_INPUT_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if df["load_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate load_id")
    if df["date"].isna().any():
        raise ValueError(f"{name}: unparseable dates")
    bad_eq = set(df["equipment"].dropna()) - set(C.EQUIPMENT_TYPES)
    if bad_eq:
        raise ValueError(f"{name}: unknown equipment {bad_eq}")
    if (df["distance"] <= 0).any() or df["distance"].isna().any():
        raise ValueError(f"{name}: non-positive or missing distance")

    warnings = []
    checks = {
        "negative weight": (df["weight"] < 0).sum(),
        "missing weight": df["weight"].isna().sum(),
        "weight at 47,500 cap (after abs)": (df["weight"].abs() == 47_500).sum(),
        "missing market_index": df["market_index"].isna().sum(),
        "missing coordinates": df[["pickup_lat", "pickup_lon"]].isna().any(axis=1).sum(),
        "pickup == delivery": (df["pickup"] == df["delivery"]).sum(),
    }
    if C.TARGET in df:
        checks["non-positive posted_rate"] = (df[C.TARGET] <= 0).sum()
    for label, count in checks.items():
        if count:
            warnings.append(f"{name}: {label}: {int(count)}")
    return warnings


# --- Split -----------------------------------------------------------------------------
def time_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp = C.TRAIN_END,
    test_start: pd.Timestamp = C.TEST_START,
    test_end: pd.Timestamp = C.TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on dates <= train_end, test on [test_start, test_end]. No date overlap."""
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    return train, test


# --- market_index ----------------------------------------------------------------------
def daily_market_index(
    *frames: pd.DataFrame,
    window: int = C.MI_DETREND_WINDOW,
    min_periods: int = C.MI_DETREND_MIN_PERIODS,
) -> pd.DataFrame:
    """Daily market_index table from input columns only (no labels).

    `market_index` is a per-day value plus row noise (R^2 of the day mean = 0.978), so every row
    gets its day's mean. Pass train AND validation so the Nov-Dec days (incl. December) exist and
    the centred rolling trend has data on both sides of the Oct/Nov boundary.
    Columns: market_index (daily mean), market_index_trend (centred rolling mean),
    market_index_detrended (difference). Index: every calendar day in range.
    """
    rows = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    daily = rows.groupby("date")["market_index"].mean().asfreq("D")
    daily = daily.interpolate(limit_direction="both")  # no gaps today; guards future inputs
    trend = daily.rolling(window, center=True, min_periods=min_periods).mean()
    return pd.DataFrame({
        "market_index": daily,
        "market_index_trend": trend,
        "market_index_detrended": daily - trend,
    })


# --- Lane reference rate-per-mile (baseline / outlier reference / target encoding) ------
def _distance_bucket(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, list(C.DISTANCE_BUCKETS), right=False).astype(str)


class LaneRateReference:
    """Expected rate-per-mile for a load, from training medians, with a fallback chain.

    Levels (first with enough support wins):
      1. "lane": median rate-per-mile of pickup x delivery x equipment (>= min_support rows)
      2. "city" (if use_city): equipment x distance-bucket median times the geometric mean of the
         pickup-city and delivery-city factors (median ratio to the equipment x distance median,
         each with >= min_support rows)
      3. "equipment_distance": median rate-per-mile of equipment x distance bucket
    """

    def __init__(self, min_support: int = C.LANE_MIN_SUPPORT, use_city: bool = True) -> None:
        self.min_support = min_support
        self.use_city = use_city

    def fit(self, df: pd.DataFrame) -> "LaneRateReference":
        d = df.assign(ppm=df[C.TARGET] / df["distance"], bucket=_distance_bucket(df["distance"]))
        self.eq_dist_ = d.groupby(["equipment", "bucket"])["ppm"].median()
        self.eq_ = d.groupby("equipment")["ppm"].median()
        lane = d.groupby(LANE_KEY)["ppm"].agg(["median", "size"])
        self.lane_ = lane.loc[lane["size"] >= self.min_support, "median"]
        rel = d["ppm"] / d.join(self.eq_dist_.rename("ed"), on=["equipment", "bucket"])["ed"]
        self.city_ = {}
        for end in ("pickup", "delivery"):
            g = rel.groupby(d[end]).agg(["median", "size"])
            self.city_[end] = g.loc[g["size"] >= self.min_support, "median"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return columns `ref_ppm` (expected rate-per-mile) and `ref_level` (which level)."""
        d = df[LANE_KEY + ["distance"]].assign(bucket=_distance_bucket(df["distance"]))
        eq_dist = d.join(self.eq_dist_.rename("v"), on=["equipment", "bucket"])["v"]
        eq_dist = eq_dist.fillna(d["equipment"].map(self.eq_))
        lane = d.join(self.lane_.rename("v"), on=LANE_KEY)["v"]
        ref = lane.copy()
        level = pd.Series(np.where(lane.notna(), "lane", "equipment_distance"), index=d.index)
        if self.use_city:
            factors = pd.concat(
                [np.log(d["pickup"].map(self.city_["pickup"])),
                 np.log(d["delivery"].map(self.city_["delivery"]))], axis=1,
            ).mean(axis=1, skipna=True)
            city = eq_dist * np.exp(factors)
            use = ref.isna() & city.notna()
            ref[use] = city[use]
            level[use] = "city"
        ref = ref.fillna(eq_dist)
        return pd.DataFrame({"ref_ppm": ref.astype(float), "ref_level": level}, index=df.index)

    def oof_transform(self, df: pd.DataFrame, n_splits: int = 5, seed: int = C.SEED) -> pd.DataFrame:
        """Out-of-fold reference for training rows (a row never sees its own target)."""
        out = []
        for fit_idx, pred_idx in KFold(n_splits, shuffle=True, random_state=seed).split(df):
            ref = LaneRateReference(self.min_support, self.use_city).fit(df.iloc[fit_idx])
            out.append(ref.transform(df.iloc[pred_idx]))
        return pd.concat(out).loc[df.index]


# --- Clean -----------------------------------------------------------------------------
class Cleaner:
    """Fit cleaning values on the training part only, then transform any file identically.

    Steps: abs(weight) -> weight median per equipment -> coordinates from the training city
    table where missing (December) -> market_index = same-day mean (+ detrended) ->
    day_of_week -> outlier flag (rows with a target only). quote_signal is passed through
    untouched for diagnostics and is NOT a model feature (month-toggled target leak).
    """

    def __init__(self, market_daily: pd.DataFrame | None = None) -> None:
        self.market_daily = market_daily

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        weight = train["weight"].abs()
        self.weight_median_ = weight.groupby(train["equipment"]).median()
        self.weight_median_all_ = float(weight.median())
        ends = [
            train[[e, f"{e}_lat", f"{e}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            for e in ("pickup", "delivery")
        ]
        self.city_table_ = pd.concat(ends).dropna().groupby("city")[["lat", "lon"]].median()
        if self.market_daily is None:  # fallback: train-only daily series
            self.market_daily = daily_market_index(train)
        self.outlier_ref_ = LaneRateReference(use_city=False).fit(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # 1-2. weight: sign errors, then per-equipment training median
        out["weight"] = out["weight"].abs()
        fill = out["equipment"].map(self.weight_median_).fillna(self.weight_median_all_)
        out["weight"] = out["weight"].fillna(fill)
        # 3. coordinates: fill only where missing (December), never overwrite given ones
        for end in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                col = f"{end}_{axis}"
                out[col] = out[col].fillna(out[end].map(self.city_table_[axis]))
        # 4. market_index: same-day mean for every row (per-row deviation is noise)
        out["market_index_raw"] = out["market_index"]
        daily = self.market_daily.reindex(out["date"])
        for col in ("market_index", "market_index_trend", "market_index_detrended"):
            out[col] = daily[col].to_numpy()
        # 5. calendar
        out["day_of_week"] = out["date"].dt.dayofweek
        # 6. target-derived columns (NaN / NA where there is no target)
        if C.TARGET not in out:
            out[C.TARGET] = np.nan
        out["rate_per_mile"] = out[C.TARGET] / out["distance"]
        out["outlier_reference"] = self.outlier_ref_.transform(out)["ref_ppm"]
        out["outlier_ratio"] = out["rate_per_mile"] / out["outlier_reference"]
        flag = (out["outlier_ratio"] < C.OUTLIER_LOW) | (out["outlier_ratio"] > C.OUTLIER_HIGH)
        out["is_outlier"] = flag.astype("boolean").mask(out[C.TARGET].isna())

        out = out[OUTPUT_COLUMNS]
        bad = out[MODEL_FEATURES].isna().sum()
        if bad.any():
            raise ValueError(f"NaN in model features after cleaning: {bad[bad > 0].to_dict()}")
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop flagged price outliers. Apply to TRAINING rows only, never to test/validation."""
    return df[~df["is_outlier"].fillna(False).astype(bool)].copy()


# --- Convenience -----------------------------------------------------------------------
def build_datasets(final: bool = False) -> dict[str, pd.DataFrame]:
    """Cleaned frames for model selection (final=False: Jan-Aug / Sep-Oct) or the final refit
    (final=True: Jan-Oct). Keys: train (outliers removed), test (all rows, flagged; empty if
    final), validation, december. Cleaner is fit on the training part only."""
    raw_train, raw_val, raw_dec = load_train(), load_validation(), load_december()
    market = daily_market_index(raw_train, raw_val)
    if final:
        train_part = raw_train[raw_train["date"] <= C.FINAL_TRAIN_END]
        test_part = raw_train.iloc[0:0]
    else:
        train_part, test_part = time_split(raw_train)
    cleaner = Cleaner(market).fit(train_part)
    return {
        "train": remove_outliers(cleaner.transform(train_part)),
        "test": cleaner.transform(test_part),
        "validation": cleaner.transform(raw_val),
        "december": cleaner.transform(raw_dec),
    }


def write_validation_predictions(
    load_ids: Sequence[str], predictions: Sequence[float], path: Path = C.PREDICTIONS_PATH
) -> pd.DataFrame:
    """Write `load_id,predicted_rate` in the template's row order."""
    template = pd.read_csv(C.TEMPLATE_PATH)[["load_id"]]
    preds = pd.Series(np.asarray(predictions, dtype=float), index=pd.Index(load_ids, name="load_id"))
    template["predicted_rate"] = template["load_id"].map(preds)
    if template["predicted_rate"].isna().any() or len(template) != 12_000:
        raise ValueError("predictions do not cover every validation load_id")
    template.to_csv(path, index=False)
    return template


def write_december_predictions(
    dates: Sequence[pd.Timestamp], predictions: Sequence[float], path: Path = C.DECEMBER_PATH
) -> pd.DataFrame:
    """Fill `predicted_rate` in the December file, keeping its 7 columns and row order."""
    dec = pd.read_csv(C.DECEMBER_ORIGINAL_PATH)
    preds = pd.Series(np.asarray(predictions, dtype=float), index=pd.to_datetime(dates))
    dec["predicted_rate"] = pd.to_datetime(dec["date"]).map(preds).round(2)
    if dec["predicted_rate"].isna().any() or list(dec.columns) != list(C.DECEMBER_COLUMNS):
        raise ValueError("December predictions incomplete or columns changed")
    dec.to_csv(path, index=False)
    return dec


# --- Smoke test ------------------------------------------------------------------------
if __name__ == "__main__":
    raw_train, raw_val, raw_dec = load_train(), load_validation(), load_december()
    for name, frame in (("train", raw_train), ("validation", raw_val), ("december", raw_dec)):
        for w in validate(frame, name):
            print("WARN", w)

    market = daily_market_index(raw_train, raw_val)

    # Model-selection split: fit on Jan-Aug only.
    tr_part, te_part = time_split(raw_train)
    cl = Cleaner(market).fit(tr_part)
    tr_flagged, te = cl.transform(tr_part), cl.transform(te_part)
    print(f"split: train {len(tr_part)} rows (outliers {int(tr_flagged.is_outlier.sum())}), "
          f"test {len(te)} rows (outliers {int(te.is_outlier.sum())})")

    # Final fit: Jan-Oct.
    cleaner = Cleaner(market).fit(raw_train)
    train = cleaner.transform(raw_train)
    val = cleaner.transform(raw_val)
    dec = cleaner.transform(raw_dec)
    assert len(val) == 12_000 and len(dec) == 31, (len(val), len(dec))
    assert list(train.columns) == list(val.columns) == list(dec.columns) == OUTPUT_COLUMNS
    for name, frame in (("train", train), ("validation", val), ("december", dec)):
        assert not frame[MODEL_FEATURES].isna().any().any(), name
    assert (dec[["pickup", "delivery"]] == ["Lexington", "Fort Wayne"]).all().all()
    assert not set(C.LEAKY_COLUMNS) & set(MODEL_FEATURES)
    print(f"final: train {len(train)} rows, outliers flagged {int(train.is_outlier.sum())}, "
          f"after removal {len(remove_outliers(train))}; validation {len(val)}; december {len(dec)}")

    lane_ref = LaneRateReference().fit(remove_outliers(train))
    print("lane reference levels on validation:",
          lane_ref.transform(val)["ref_level"].value_counts().to_dict())

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(C.PROCESSED_DIR / "train_clean_flagged.csv", index=False)
    val.to_csv(C.PROCESSED_DIR / "validation_clean.csv", index=False)
    dec.to_csv(C.PROCESSED_DIR / "december_clean.csv", index=False)
    market.to_csv(C.PROCESSED_DIR / "market_index_daily.csv", index_label="date")
    cleaner.city_table_.to_csv(C.PROCESSED_DIR / "city_table.csv")
    print(dec[["date", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
               "market_index", "market_index_detrended", "day_of_week"]].head(8).to_string())
    print("OK")
