"""Model factory: baselines and learned models behind one `fit(train) / predict(df)` interface.

All models take cleaned frames (from `src.data`) and return predicted `posted_rate` in dollars.
Learned models predict a log target and convert back with exp():
  target="log_ppm"   : y = log(rate / distance)
  target="log_ratio" : y = log(rate / distance / ref_ppm)   (residual on the in-fold lane reference)

`make_model(name)` returns a fresh, unfitted model from the registry `MODEL_SPECS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src import config as C
from src.data import LaneRateReference, _distance_bucket
from src.features import COORDS, MI_VARIANTS, FeatureBuilder, FeatureSpec


# --- Baselines -------------------------------------------------------------------------
class EquipmentDistanceBaseline:
    """(a) median rate-per-mile of equipment x distance bucket, times distance."""

    name = "baseline_a_eq_dist"

    def fit(self, train: pd.DataFrame) -> "EquipmentDistanceBaseline":
        d = train.assign(ppm=train[C.TARGET] / train["distance"],
                         bucket=_distance_bucket(train["distance"]))
        self.table_ = d.groupby(["equipment", "bucket"])["ppm"].median()
        self.eq_ = d.groupby("equipment")["ppm"].median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = df.assign(bucket=_distance_bucket(df["distance"]))
        ppm = d.join(self.table_.rename("v"), on=["equipment", "bucket"])["v"]
        ppm = ppm.fillna(d["equipment"].map(self.eq_))
        return (ppm * df["distance"]).to_numpy(float)


class LaneReferenceBaseline:
    """(b) lane x equipment median rate-per-mile (city / equipment x distance fallback) x distance.
    (c) with market_adjust=True: times exp(a + b * market_index), fitted by OLS of the OOF
    log(ppm / ref) on the same-day market_index."""

    def __init__(self, market_adjust: bool = False, mi_column: str = "market_index") -> None:
        self.market_adjust = market_adjust
        self.mi_column = mi_column
        self.name = "baseline_c_lane_x_mi" if market_adjust else "baseline_b_lane"

    def fit(self, train: pd.DataFrame) -> "LaneReferenceBaseline":
        self.ref_ = LaneRateReference().fit(train)
        if self.market_adjust:
            oof = self.ref_.oof_transform(train)["ref_ppm"]
            y = np.log(train[C.TARGET] / train["distance"] / oof)
            self.adj_ = LinearRegression().fit(train[[self.mi_column]], y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ppm = self.ref_.transform(df)["ref_ppm"].to_numpy(float)
        if self.market_adjust:
            ppm = ppm * np.exp(self.adj_.predict(df[[self.mi_column]]))
        return ppm * df["distance"].to_numpy(float)


# --- Learned models --------------------------------------------------------------------
class Scale(BaseEstimator, TransformerMixin):
    """Multiply a block of columns by a constant. In Ridge, scaling a one-hot block by s
    raises its effective penalty from alpha to alpha / s^2 (per-group regularisation)."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def fit(self, X: Any, y: Any = None) -> "Scale":
        return self

    def transform(self, X: Any) -> Any:
        return X * self.factor

    def __sklearn_is_fitted__(self) -> bool:  # stateless
        return True


# Extra one-hot keys for Ridge (round 2). Unseen keys -> all-zero row (handle_unknown="ignore").
ONEHOT_KEYS = {
    "lane": lambda X: X["pickup"] + "|" + X["delivery"] + "|" + X["equipment"],
    "ulane": lambda X: (np.minimum(X["pickup"], X["delivery"]) + "|" + np.maximum(X["pickup"], X["delivery"])
                        + "|" + X["equipment"]),
    "pickup_eq": lambda X: X["pickup"] + "|" + X["equipment"],
    "delivery_eq": lambda X: X["delivery"] + "|" + X["equipment"],
}


@dataclass
class RateModel:
    """Feature builder + log target + estimator. `estimator` in {ridge, rf, hgb, catboost}."""

    estimator: str
    target: str = "log_ppm"
    features: FeatureSpec = field(default_factory=FeatureSpec)
    params: dict[str, Any] = field(default_factory=dict)
    halflife_days: float | None = None  # recency sample weights (None = uniform)
    splines: bool = False  # Ridge only: cubic splines on log_distance and weight, equipment x log_distance
    mi_offset: bool = False  # fixed linear market_index_detrended effect (OLS in-fold), model sees no MI
    residual_on: str | None = None  # "ridge": fit `estimator` on the residual of a Ridge base model
    residual_shrink: float = 1.0  # multiplier on the residual model's prediction
    # Ridge-only round-2 options (all off by default; existing registry entries are unchanged)
    onehot_extra: tuple[tuple[str, float], ...] = ()  # ((ONEHOT_KEYS key, scale), ...)
    mi_interactions: tuple[str, ...] = ()  # subset of {"equipment", "log_distance"} x detrended MI
    coord_knots: int = 0  # >0: cubic splines on the 4 coordinates with this many knots
    anchor_weeks: int | None = None  # refit the intercept on the last N weeks of training rows
    name: str = ""
    seed: int = C.SEED

    # -- target ------------------------------------------------------------------------
    def _y(self, X: pd.DataFrame) -> np.ndarray:
        y = np.log(X[C.TARGET] / X["distance"])
        if self.target == "log_ratio":
            y = y - X["ref_log_ppm"]
        elif self.target != "log_ppm":
            raise ValueError(f"unknown target {self.target}")
        return y.to_numpy(float)

    def _offset(self, X: pd.DataFrame) -> np.ndarray:
        if not self.mi_offset:
            return np.zeros(len(X))
        return self.mi_slope_ * X["market_index_detrended"].to_numpy(float)

    def _to_rate(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        log_ppm = pred + self._offset(X)
        if self.target == "log_ratio":
            log_ppm = log_ppm + X["ref_log_ppm"].to_numpy(float)
        return np.exp(log_ppm) * X["distance"].to_numpy(float)

    def _weights(self, train: pd.DataFrame) -> np.ndarray | None:
        if self.halflife_days is None:
            return None
        age = (train["date"].max() - train["date"]).dt.days.to_numpy(float)
        return 0.5 ** (age / self.halflife_days)

    # -- design matrix -----------------------------------------------------------------
    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        cats, nums = self.features.categorical(), self.features.numeric()
        M = X[cats + nums].copy()
        if self.estimator in ("hgb", "rf"):
            for c in cats:  # fixed training categories; unseen -> NaN (missing)
                M[c] = pd.Categorical(M[c], categories=self.categories_[c])
            if self.estimator == "rf":  # RF has no categorical support: ordinal codes, unseen -1
                for c in cats:
                    M[c] = M[c].cat.codes.astype(float)
        elif self.estimator == "ridge":
            M["log_distance_sq"] = M["log_distance"] ** 2
            if self.splines:
                for eq in C.EQUIPMENT_TYPES:
                    M[f"logd_x_{eq}"] = M["log_distance"] * (X["equipment"] == eq)
            if "day_of_week" in M:
                M["day_of_week"] = M["day_of_week"].astype(str)
            for key, _ in self.onehot_extra:
                M[f"oh_{key}"] = ONEHOT_KEYS[key](X)
            mi = X["market_index_detrended"].to_numpy(float) if self.mi_interactions else None
            if "equipment" in self.mi_interactions:
                for eq in C.EQUIPMENT_TYPES:
                    M[f"mi_x_{eq}"] = mi * (X["equipment"] == eq).to_numpy(float)
            if "log_distance" in self.mi_interactions:
                M["mi_x_logd"] = mi * (M["log_distance"].to_numpy(float) - np.log(500.0))
        return M

    def _mi_x_cols(self) -> list[str]:
        cols = []
        if "equipment" in self.mi_interactions:
            cols += [f"mi_x_{eq}" for eq in C.EQUIPMENT_TYPES]
        if "log_distance" in self.mi_interactions:
            cols += ["mi_x_logd"]
        return cols

    def _build(self) -> Any:
        p = dict(self.params)
        cats = self.features.categorical()
        if self.estimator == "ridge":
            nums = [c for c in self.features.numeric() if c != "day_of_week"] + ["log_distance_sq"]
            onehot = cats + (["day_of_week"] if self.features.day_of_week else [])
            parts = [("cat", OneHotEncoder(handle_unknown="ignore"), onehot), ("num", StandardScaler(), nums)]
            if self.splines:
                spl = [c for c in ("log_distance", "weight") if c in nums]
                parts.append(("spl", SplineTransformer(n_knots=p.get("n_knots", 6), degree=3), spl))
                parts.append(("eqd", StandardScaler(), [f"logd_x_{eq}" for eq in C.EQUIPMENT_TYPES]))
            for key, scale in self.onehot_extra:
                parts.append((f"oh_{key}", Pipeline([("oh", OneHotEncoder(handle_unknown="ignore")),
                                                     ("scale", Scale(scale))]), [f"oh_{key}"]))
            if self.mi_interactions:
                parts.append(("mix", StandardScaler(), self._mi_x_cols()))
            if self.coord_knots:
                parts.append(("cspl", SplineTransformer(n_knots=self.coord_knots, degree=3), COORDS))
            pre = ColumnTransformer(parts)
            return Pipeline([("pre", pre), ("model", Ridge(alpha=p.get("alpha", 1.0)))])
        if self.estimator == "rf":
            return RandomForestRegressor(
                n_estimators=p.get("n_estimators", 300), min_samples_leaf=p.get("min_samples_leaf", 3),
                max_features=p.get("max_features", 0.5), n_jobs=-1, random_state=self.seed)
        if self.estimator == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=p.get("max_iter", 600), learning_rate=p.get("learning_rate", 0.05),
                max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 20),
                l2_regularization=p.get("l2_regularization", 0.0), categorical_features="from_dtype",
                monotonic_cst=p.get("monotonic_cst"),
                early_stopping=False, random_state=self.seed)
        if self.estimator == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=p.get("iterations", 1500), learning_rate=p.get("learning_rate", 0.05),
                depth=p.get("depth", 6), l2_leaf_reg=p.get("l2_leaf_reg", 3.0), loss_function="RMSE",
                cat_features=cats, random_seed=self.seed, verbose=0, allow_writing_files=False,
                thread_count=p.get("thread_count", -1))
        raise ValueError(f"unknown estimator {self.estimator}")

    # -- API ---------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "RateModel":
        self.builder_ = FeatureBuilder(seed=self.seed).fit(train)
        X = self.builder_.transform_train(train)
        self.categories_ = {c: sorted(X[c].unique()) for c in self.features.categorical()}
        y = self._y(X)
        if self.mi_offset:
            if MI_VARIANTS[self.features.market_index]:
                raise ValueError("mi_offset=True expects features.market_index='none'")
            det = X["market_index_detrended"].to_numpy(float)
            self.mi_slope_ = float(LinearRegression().fit(det[:, None], y).coef_[0])
            y = y - self._offset(X)
        w = self._weights(train)
        self.base_ = None
        if self.residual_on == "ridge":
            self.base_ = RateModel("ridge", target=self.target, features=self.base_features(),
                                   splines=self.splines, seed=self.seed)
            self.base_.categories_ = self.categories_
            self.base_.model_ = self.base_._build()
            self.base_.model_.fit(self.base_._matrix(X), y)
            y = y - self.base_.model_.predict(self.base_._matrix(X))
        self.model_ = self._build()
        fit_kw = {}
        if w is not None:
            fit_kw = {"model__sample_weight": w} if self.estimator == "ridge" else {"sample_weight": w}
        self.model_.fit(self._matrix(X), y, **fit_kw)
        self.anchor_ = 0.0
        if self.anchor_weeks:
            recent = (X["date"] > X["date"].max() - pd.Timedelta(weeks=self.anchor_weeks)).to_numpy()
            self.anchor_ = float(np.mean(y[recent] - self.predict_log(X[recent])))
        return self

    def base_features(self) -> FeatureSpec:
        """Ridge base of a residual model: same spec, plus detrended MI, without the lane category."""
        mi = self.features.market_index if self.features.market_index != "none" else "detrended"
        return replace(self.features, market_index=mi, lane_category=False)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self.model_.predict(self._matrix(X)), float)
        if self.base_ is not None:
            pred = self.base_.model_.predict(self.base_._matrix(X)) + self.residual_shrink * pred
        return pred + getattr(self, "anchor_", 0.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.builder_.transform(df)
        return self._to_rate(X, self.predict_log(X))


class BlendModel:
    """Geometric mean of member predictions (equal weights)."""

    def __init__(self, members: list[Any], name: str = "blend") -> None:
        self.members, self.name = members, name

    def fit(self, train: pd.DataFrame) -> "BlendModel":
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(np.mean([np.log(m.predict(df)) for m in self.members], axis=0))


# --- Registry --------------------------------------------------------------------------
FULL = FeatureSpec()
CAT_FULL = FeatureSpec(lane_category=True)
RIDGE_SPEC = FeatureSpec(market_index="detrended", day_of_week=False, lane_ref=False)
TREE_SPEC = FeatureSpec(market_index="none", day_of_week=False)
HGB_RESID_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 50}

MODEL_SPECS: dict[str, dict[str, Any]] = {
    # baselines
    "baseline_a_eq_dist": {"kind": "baseline_a"},
    "baseline_b_lane": {"kind": "baseline_b"},
    "baseline_c_lane_x_mi": {"kind": "baseline_c"},
    "baseline_c2_lane_x_mi_detrended": {"kind": "baseline_c2"},
    # model comparison (full locked feature set)
    "ridge_log_ppm": {"estimator": "ridge", "target": "log_ppm", "features": FeatureSpec(lane_ref=False)},
    "ridge_log_ratio": {"estimator": "ridge", "target": "log_ratio", "features": FULL},
    "rf_log_ratio": {"estimator": "rf", "target": "log_ratio", "features": FULL},
    "hgb_log_ppm": {"estimator": "hgb", "target": "log_ppm", "features": FULL},
    "hgb_log_ratio": {"estimator": "hgb", "target": "log_ratio", "features": FULL},
    "catboost_log_ppm": {"estimator": "catboost", "target": "log_ppm", "features": CAT_FULL},
    "catboost_log_ratio": {"estimator": "catboost", "target": "log_ratio", "features": CAT_FULL},
    # corrected market_index handling (level dropped; detrended only, no day_of_week).
    # Ridge sees detrended MI linearly; trees get it as a fixed in-fold linear offset because a
    # per-day-unique MI value lets them memorise day-level price noise.
    "ridge_det": {"estimator": "ridge", "target": "log_ppm", "features": RIDGE_SPEC},
    "ridge_spline": {"estimator": "ridge", "target": "log_ppm", "features": RIDGE_SPEC, "splines": True},
    "rf_mi_offset": {"estimator": "rf", "target": "log_ratio", "features": TREE_SPEC, "mi_offset": True},
    "hgb_mi_offset": {"estimator": "hgb", "target": "log_ppm", "features": TREE_SPEC, "mi_offset": True},
    "catboost_mi_offset": {"estimator": "catboost", "target": "log_ppm",
                           "features": replace(TREE_SPEC, lane_category=True), "mi_offset": True},
    "ridge_spline_hgb_resid": {"estimator": "hgb", "target": "log_ppm", "features": TREE_SPEC,
                               "residual_on": "ridge", "splines": True, "params": HGB_RESID_PARAMS},
}


def make_model(name: str, **overrides: Any) -> Any:
    """Fresh unfitted model by registry name. `overrides` replace RateModel fields
    (e.g. features=FeatureSpec(...), params={...}, halflife_days=60)."""
    spec = dict(MODEL_SPECS[name])
    kind = spec.pop("kind", None)
    if kind == "baseline_a":
        return EquipmentDistanceBaseline()
    if kind == "baseline_b":
        return LaneReferenceBaseline(market_adjust=False)
    if kind == "baseline_c":
        return LaneReferenceBaseline(market_adjust=True)
    if kind == "baseline_c2":
        return LaneReferenceBaseline(market_adjust=True, mi_column="market_index_detrended")
    spec.update(overrides)
    return RateModel(name=name, **spec)


def with_overrides(model: RateModel, **kw: Any) -> RateModel:
    return replace(model, **kw)
