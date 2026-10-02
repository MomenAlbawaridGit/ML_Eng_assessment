"""Model-facing features built on top of the cleaned frames from `src.data`.

`FeatureBuilder` is fit on a fold's training rows only. It owns the in-fold lane reference
(`LaneRateReference`): training rows get an out-of-fold reference (a row never sees its own
target), and test / validation / December rows get the reference fit on all training rows.

Feature groups (names refer to columns produced by `FeatureBuilder.transform`):
  categorical : pickup, delivery, equipment (+ lane for CatBoost)
  numeric     : distance, log_distance, weight, coordinates, market_index variants, day_of_week
  lane ref    : ref_log_ppm (log expected rate-per-mile), ref_level_code (which fallback level)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config as C
from src.data import LaneRateReference

COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
MI_VARIANTS = {
    "level": ["market_index"],
    "detrended": ["market_index_detrended"],
    "both": ["market_index", "market_index_detrended"],
    "none": [],
}
REF_LEVEL_CODES = {"lane": 0, "city": 1, "equipment_distance": 2}


@dataclass(frozen=True)
class FeatureSpec:
    """Which feature groups a model sees. Defaults = the full locked feature set."""

    market_index: str = "both"  # key of MI_VARIANTS
    day_of_week: bool = True
    coordinates: bool = True
    lane_ref: bool = True  # ref_log_ppm + ref_level_code (in-fold, OOF on training rows)
    cities: bool = True  # pickup / delivery categoricals (one-hot for Ridge, native for GBMs)
    lane_category: bool = False  # pickup|delivery|equipment string (CatBoost only)
    weight: bool = True
    extra: tuple[str, ...] = field(default_factory=tuple)

    def categorical(self) -> list[str]:
        cols = ["equipment"]
        if self.cities:
            cols += ["pickup", "delivery"]
        if self.lane_category:
            cols += ["lane"]
        return cols

    def numeric(self) -> list[str]:
        cols = ["distance", "log_distance"]
        if self.weight:
            cols += ["weight"]
        if self.coordinates:
            cols += COORDS
        cols += MI_VARIANTS[self.market_index]
        if self.day_of_week:
            cols += ["day_of_week"]
        if self.lane_ref:
            cols += ["ref_log_ppm", "ref_level_code"]
        return cols + list(self.extra)

    def columns(self) -> list[str]:
        return self.categorical() + self.numeric()


class FeatureBuilder:
    """Fit on training rows; transform any cleaned frame into model features."""

    def __init__(self, oof_splits: int = 5, seed: int = C.SEED) -> None:
        self.oof_splits = oof_splits
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.ref_ = LaneRateReference().fit(train)
        return self

    def _assemble(self, df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["log_distance"] = np.log(out["distance"])
        dim = out["date"].dt.days_in_month
        out["day_of_month"] = (out["date"].dt.day - 1) / (dim - 1)  # 0 = 1st, 1 = last day (opt-in via extra)
        out["lane"] = out["pickup"] + "|" + out["delivery"] + "|" + out["equipment"]
        out["ref_ppm"] = ref["ref_ppm"].to_numpy()
        out["ref_level"] = ref["ref_level"].to_numpy()
        out["ref_log_ppm"] = np.log(out["ref_ppm"])
        out["ref_level_code"] = out["ref_level"].map(REF_LEVEL_CODES).astype(float)
        return out

    def transform_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Training rows: out-of-fold lane reference (no row sees its own target)."""
        ref = self.ref_.oof_transform(train, n_splits=self.oof_splits, seed=self.seed)
        return self._assemble(train, ref)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Any other rows (test fold, validation, December): reference fit on all training rows."""
        return self._assemble(df, self.ref_.transform(df))
