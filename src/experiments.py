"""Ablation, tuning and blend experiments, registered into `src.evaluate` (run there).

`SELECTED` is the model chosen by the selection rule (lowest mean walk-forward MAPE excl. outliers,
beats baseline (b) on every fold, simpler model preferred within ~0.1pp). `src.train` fits it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from src.features import FeatureSpec as FS
from src.models import RIDGE_SPEC, TREE_SPEC, BlendModel, make_model

SELECTED = "ridge_spline"
EXPERIMENTS: list[tuple[str, str, str, Callable[[], Any]]] = []  # (name, group, description, factory)


def add(name: str, group: str, desc: str, factory: Callable[[], Any]) -> None:
    EXPERIMENTS.append((name, group, desc, factory))

# 3. market_index / day_of_week diagnosis (why the locked full feature set fails)
for est, name in (("ridge", "ridge_log_ppm"), ("hgb", "hgb_log_ratio")):
    for mi in ("none", "level", "detrended"):
        for dow in (True, False):
            if est == "hgb" and not dow:
                continue
            add(f"{est}_mi_{mi}{'_dow' if dow else ''}", "3 MI/dow diagnosis",
                f"{name} with market_index={mi}, day_of_week={dow}",
                lambda name=name, mi=mi, dow=dow: make_model(
                    name, features=replace(FS(lane_ref=name != "ridge_log_ppm"), market_index=mi, day_of_week=dow)))

# 4. model comparison with corrected MI handling
add("ridge_det", "4 corrected", "Ridge one-hot cities, y=log ppm; distance(+sq), weight, coords, MI detrended",
    lambda: make_model("ridge_det"))
add("ridge_spline", "4 corrected", "ridge_det + cubic splines (log distance, weight) + equipment x log distance",
    lambda: make_model("ridge_spline"))
add("ridge_spline_log_ratio", "4 corrected", "ridge_spline with y=log(ppm / in-fold lane ref) + lane ref features",
    lambda: make_model("ridge_spline", target="log_ratio", features=replace(RIDGE_SPEC, lane_ref=True)))
add("rf_mi_offset", "4 corrected", "RandomForest, y=log(ppm/lane ref) minus linear MI-detrended offset; no MI/dow",
    lambda: make_model("rf_mi_offset"))
add("hgb_mi_offset", "4 corrected", "HistGB native cats + lane ref, y=log ppm minus linear MI-detrended offset",
    lambda: make_model("hgb_mi_offset"))
add("catboost_mi_offset", "4 corrected", "CatBoost cats+lane + lane ref, y=log ppm minus linear MI-detrended offset",
    lambda: make_model("catboost_mi_offset"))
add("ridge_spline_hgb_resid", "4 corrected", "ridge_spline + shallow HistGB (200x15 leaves) on its residual, no MI in trees",
    lambda: make_model("ridge_spline_hgb_resid"))

# 5. ablations on the best model (ridge_spline)
A = {
    "abl_mi_level": replace(RIDGE_SPEC, market_index="level"),
    "abl_mi_both": replace(RIDGE_SPEC, market_index="both"),
    "abl_mi_none": replace(RIDGE_SPEC, market_index="none"),
    "abl_plus_dow": replace(RIDGE_SPEC, day_of_week=True),
    "abl_mi_none_plus_dow": replace(RIDGE_SPEC, market_index="none", day_of_week=True),
    "abl_plus_lane_ref": replace(RIDGE_SPEC, lane_ref=True),
    "abl_no_coords": replace(RIDGE_SPEC, coordinates=False),
    "abl_no_cities": replace(RIDGE_SPEC, cities=False),
    "abl_no_weight": replace(RIDGE_SPEC, weight=False),
}
for n, spec in A.items():
    add(n, "5 ablation", f"ridge_spline with {n.removeprefix('abl_')}",
        lambda spec=spec: make_model("ridge_spline", features=spec))
for hl in (60, 120, 240):
    add(f"abl_halflife_{hl}d", "5 ablation", f"ridge_spline with recency weights, half-life {hl} days",
        lambda hl=hl: make_model("ridge_spline", halflife_days=hl))

# 6. light tuning (ridge_spline): alpha x spline knots
for alpha in (0.3, 10.0, 30.0):
    add(f"tune_alpha_{alpha:g}", "6 tuning", f"ridge_spline alpha={alpha:g} (default 1)",
        lambda alpha=alpha: make_model("ridge_spline", params={"alpha": alpha}))
for k in (4, 10):
    add(f"tune_knots_{k}", "6 tuning", f"ridge_spline n_knots={k} (default 6)",
        lambda k=k: make_model("ridge_spline", params={"n_knots": k}))

# 7. blend of the top two families
add("blend_ridge_spline_catboost", "7 blend", "geometric mean of ridge_spline and catboost_mi_offset",
    lambda: BlendModel([make_model("ridge_spline"), make_model("catboost_mi_offset")]))
add("blend_ridge_spline_hgb", "7 blend", "geometric mean of ridge_spline and hgb_mi_offset",
    lambda: BlendModel([make_model("ridge_spline"), make_model("hgb_mi_offset")]))

# 8. round 2 (2026-10-02): all REJECTED under the adoption rule (mean wf MAPE must improve
# by >= 0.05pp and no fold, incl. city, may be worse by > 0.02pp). Kept for documentation.
R2 = lambda **kw: (lambda: make_model("ridge_spline", **kw))  # noqa: E731
add("abl_day_of_month", "8 round2 (rejected)", "ridge_spline + day_of_month (0 = 1st, 1 = last day), numeric",
    R2(features=replace(RIDGE_SPEC, extra=("day_of_month",))))
for key, label in (("lane", "pickup x delivery x equipment"), ("ulane", "undirected lane x equipment")):
    for sc in (0.1, 0.3, 1.0):
        add(f"r2_{key}_s{sc:g}", "8 round2 (rejected)",
            f"ridge_spline + one-hot {label}, block scaled x{sc:g} (penalty alpha/{sc * sc:g})",
            R2(onehot_extra=((key, sc),)))
for sc in (0.1, 0.3, 1.0):
    add(f"r2_cityeq_s{sc:g}", "8 round2 (rejected)",
        f"ridge_spline + one-hot pickup x equipment and delivery x equipment, scaled x{sc:g}",
        R2(onehot_extra=(("pickup_eq", sc), ("delivery_eq", sc))))
for mix in (("equipment",), ("log_distance",), ("equipment", "log_distance")):
    add(f"r2_mi_x_{'_'.join(m.replace('log_distance', 'logd').replace('equipment', 'eq') for m in mix)}",
        "8 round2 (rejected)", f"ridge_spline + detrended MI x {' and '.join(mix)}", R2(mi_interactions=mix))
for k in (4, 6, 8):
    add(f"r2_coord_spline_k{k}", "8 round2 (rejected)",
        f"ridge_spline + cubic splines on the 4 coordinates ({k} knots); helps city only", R2(coord_knots=k))
for n in (4, 8):
    add(f"r2_anchor_{n}w", "8 round2 (rejected)",
        f"ridge_spline with intercept refit on the last {n} weeks of training rows", R2(anchor_weeks=n))
add("r2_combo_lane0.3_coord6", "8 round2 (rejected)",
    "ridge_spline + lane one-hot x0.3 + coordinate splines (6 knots)",
    R2(onehot_extra=(("lane", 0.3),), coord_knots=6))
add("r2_combo_lane0.3_coord6_mix", "8 round2 (rejected)",
    "r2_combo_lane0.3_coord6 + detrended MI x equipment and x log distance",
    R2(onehot_extra=(("lane", 0.3),), coord_knots=6, mi_interactions=("equipment", "log_distance")))
