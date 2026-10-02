"""Validation design + experiment table.

Folds (everything — Cleaner, outlier rule, lane reference, encoders — is fit on the fold's
training rows only; outliers are removed from training rows only):
  wf1  train <= Jun  -> test Jul-Aug      (walk-forward, 1-2 month horizon)
  wf2  train <= Jul  -> test Aug-Sep
  wf3  train <= Aug  -> test Sep-Oct      (main fold: breakdowns + saved predictions)
  city train Jan-Aug minus every load touching ~12% held-out cities
       -> test Sep-Oct loads touching those cities (rehearses the 12% unseen-city validation rows)

Metrics on each test fold, with and without outlier rows (headline = without):
MAPE, median APE, MAE, RMSE, R^2.

    python -m src.evaluate            # full experiment table -> reports/experiments.{csv,md}
    python -m src.evaluate --quick    # baselines + model comparison only
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import config as C
from src.data import Cleaner, _distance_bucket, daily_market_index, load_train, load_validation, remove_outliers
from src.features import FeatureSpec
from src.models import make_model

CITY_HOLDOUT_SHARE = 0.12
WALK_FORWARD = ("wf1", "wf2", "wf3")
MAIN_FOLD = "wf3"


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    test_start: str
    test_end: str
    city_holdout: bool = False


FOLDS = {
    "wf1": Fold("wf1", "2025-06-30", "2025-07-01", "2025-08-31"),
    "wf2": Fold("wf2", "2025-07-31", "2025-08-01", "2025-09-30"),
    "wf3": Fold("wf3", "2025-08-31", "2025-09-01", "2025-10-31"),
    "city": Fold("city", "2025-08-31", "2025-09-01", "2025-10-31", city_holdout=True),
}


def held_out_cities(raw_train: pd.DataFrame, share: float = CITY_HOLDOUT_SHARE, seed: int = C.SEED) -> list[str]:
    cities = sorted(set(raw_train["pickup"]) | set(raw_train["delivery"]))
    n = int(round(share * len(cities)))
    return sorted(np.random.default_rng(seed).choice(cities, size=n, replace=False).tolist())


@lru_cache(maxsize=None)
def fold_data(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train without outliers, test with all rows flagged) for one fold. Cached."""
    fold = FOLDS[name]
    raw = load_train()
    market = daily_market_index(raw, load_validation())  # inputs only, no labels
    train_part = raw[raw["date"] <= pd.Timestamp(fold.train_end)]
    test_part = raw[(raw["date"] >= pd.Timestamp(fold.test_start)) & (raw["date"] <= pd.Timestamp(fold.test_end))]
    if fold.city_holdout:
        held = held_out_cities(raw)
        touches = lambda d: d["pickup"].isin(held) | d["delivery"].isin(held)  # noqa: E731
        train_part, test_part = train_part[~touches(train_part)], test_part[touches(test_part)]
    cleaner = Cleaner(market).fit(train_part)
    train = remove_outliers(cleaner.transform(train_part)).reset_index(drop=True)
    test = cleaner.transform(test_part).reset_index(drop=True)
    seen = set(zip(train["pickup"], train["delivery"], train["equipment"]))
    test["lane_seen"] = [k in seen for k in zip(test["pickup"], test["delivery"], test["equipment"])]
    test["is_outlier"] = test["is_outlier"].astype(bool)
    return train, test


# --- Metrics ---------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    return {
        "n": len(y),
        "MAPE": 100 * ape.mean(),
        "MedAPE": 100 * np.median(ape),
        "MAE": np.abs(p - y).mean(),
        "RMSE": np.sqrt(((p - y) ** 2).mean()),
        "R2": 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
    }


def fold_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    clean = ~test["is_outlier"].to_numpy()
    out = {k: v for k, v in metrics(test[C.TARGET][clean], pred[clean]).items()}
    out.update({f"{k}_all": v for k, v in metrics(test[C.TARGET], pred).items() if k != "n"})
    return out


def breakdown(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Main-fold breakdown (outliers excluded) by equipment, distance bucket, lane seen."""
    d = test.assign(pred=pred, bucket=_distance_bucket(test["distance"]))
    d = d[~d["is_outlier"]]
    rows = []
    for col in ("equipment", "bucket", "lane_seen"):
        for key, g in d.groupby(col, observed=True):
            rows.append({"group": col, "value": str(key), **metrics(g[C.TARGET], g["pred"])})
    return pd.DataFrame(rows)


# --- Experiment runner -----------------------------------------------------------------
def run(model_factory: Callable[[], Any], folds: tuple[str, ...] = tuple(FOLDS)) -> tuple[pd.DataFrame, dict]:
    """Fit a fresh model per fold; return per-fold metrics and per-fold predictions."""
    rows, preds = [], {}
    for f in folds:
        train, test = fold_data(f)
        p = model_factory().fit(train).predict(test)
        preds[f] = p
        rows.append({"fold": f, **fold_metrics(test, p)})
    return pd.DataFrame(rows), preds


def summarise(name: str, group: str, desc: str, per_fold: pd.DataFrame, seconds: float) -> dict:
    pf = per_fold.set_index("fold")
    wf = pf.loc[[f for f in WALK_FORWARD if f in pf.index]]
    row = {"experiment": name, "group": group, "description": desc}
    for f in pf.index:
        row[f"MAPE_{f}"] = pf.loc[f, "MAPE"]
    row.update({
        "MAPE_wf_mean": wf["MAPE"].mean(),
        "MAPE_all_wf_mean": wf["MAPE_all"].mean(),
        "MedAPE_wf_mean": wf["MedAPE"].mean(),
        "MAE_wf_mean": wf["MAE"].mean(),
        "RMSE_wf_mean": wf["RMSE"].mean(),
        "R2_wf_mean": wf["R2"].mean(),
        "MAE_all_wf_mean": wf["MAE_all"].mean(),
        "RMSE_all_wf_mean": wf["RMSE_all"].mean(),
        "seconds": seconds,
    })
    return row


def experiments() -> list[tuple[str, str, str, Callable[[], Any]]]:
    """(name, group, description, factory). Ordered as in the report table."""
    E = []

    def add(name, group, desc, factory):
        E.append((name, group, desc, factory))

    # 1. baselines
    add("baseline_a_eq_dist", "1 baseline", "equipment x distance-bucket median ppm x distance",
        lambda: make_model("baseline_a_eq_dist"))
    add("baseline_b_lane", "1 baseline", "lane x equipment median ppm (city / eq x dist fallback) x distance",
        lambda: make_model("baseline_b_lane"))
    add("baseline_c_lane_x_mi", "1 baseline", "(b) x exp(a + b*market_index), OLS on OOF log ratio",
        lambda: make_model("baseline_c_lane_x_mi"))
    add("baseline_c2_lane_x_mi_detrended", "1 baseline", "(b) x exp(a + b*detrended market_index)",
        lambda: make_model("baseline_c2_lane_x_mi_detrended"))
    # 2-3. model comparison on the full locked feature set
    full = "cats + distance + weight + coords + MI level&detrended + dow"
    add("ridge_log_ppm", "2 linear", f"Ridge, one-hot cities, y=log ppm; {full}", lambda: make_model("ridge_log_ppm"))
    add("ridge_log_ratio", "2 linear", f"Ridge, y=log(ppm/lane ref); {full} + lane ref", lambda: make_model("ridge_log_ratio"))
    add("rf_log_ratio", "3 trees", f"RandomForest, y=log(ppm/lane ref); {full} + lane ref", lambda: make_model("rf_log_ratio"))
    add("hgb_log_ppm", "3 trees", f"HistGB native cats, y=log ppm; {full} + lane ref", lambda: make_model("hgb_log_ppm"))
    add("hgb_log_ratio", "3 trees", f"HistGB native cats, y=log(ppm/lane ref); {full} + lane ref", lambda: make_model("hgb_log_ratio"))
    add("catboost_log_ppm", "3 trees", f"CatBoost cats+lane, y=log ppm; {full} + lane ref", lambda: make_model("catboost_log_ppm"))
    add("catboost_log_ratio", "3 trees", f"CatBoost cats+lane, y=log(ppm/lane ref); {full} + lane ref", lambda: make_model("catboost_log_ratio"))
    return E


def to_markdown(table: pd.DataFrame) -> str:
    cols = ["experiment", "group", "MAPE_wf1", "MAPE_wf2", "MAPE_wf3", "MAPE_wf_mean", "MAPE_all_wf_mean",
            "MedAPE_wf_mean", "MAE_wf_mean", "RMSE_wf_mean", "R2_wf_mean", "MAPE_city", "description"]
    t = table[[c for c in cols if c in table]].copy()
    fmt = {c: "{:.3f}" for c in t if c.startswith(("MAPE", "MedAPE"))}
    fmt.update({c: "{:.2f}" for c in ("MAE_wf_mean", "RMSE_wf_mean")})
    fmt["R2_wf_mean"] = "{:.4f}"
    for c, f in fmt.items():
        if c in t:
            t[c] = t[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    header = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in t.itertuples(index=False))
    return header + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="baselines + model comparison only")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    args = ap.parse_args()

    from src import experiments as X  # ablation / tuning / blend experiments + SELECTED

    todo = experiments() + ([] if args.quick else X.EXPERIMENTS)
    if args.only:
        todo = [e for e in todo if e[0] in args.only]
    rows, per_fold_all = [], []
    for name, group, desc, factory in todo:
        t0 = time.time()
        pf, preds = run(factory)
        secs = time.time() - t0
        rows.append(summarise(name, group, desc, pf, secs))
        per_fold_all.append(pf.assign(experiment=name))
        r = rows[-1]
        print(f"{name:40s} wf {r['MAPE_wf1']:.3f} {r['MAPE_wf2']:.3f} {r['MAPE_wf3']:.3f} "
              f"mean {r['MAPE_wf_mean']:.3f} | all {r['MAPE_all_wf_mean']:.3f} | city {r['MAPE_city']:.3f} "
              f"({secs:.0f}s)", flush=True)
        if name == X.SELECTED:
            _write_main_fold_outputs(name, preds)

    table = pd.DataFrame(rows)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    if not args.only:
        table.to_csv(C.REPORTS_DIR / "experiments.csv", index=False)
        pd.concat(per_fold_all).to_csv(C.REPORTS_DIR / "experiments_per_fold.csv", index=False)
        md = ["# Experiment table (generated by `python -m src.evaluate`)", "",
              "MAPE in %, outlier rows excluded unless the column says `_all`. "
              "wf1: <=Jun -> Jul-Aug, wf2: <=Jul -> Aug-Sep, wf3: <=Aug -> Sep-Oct; "
              "city: Jan-Aug minus 12% held-out cities -> Sep-Oct loads touching them. "
              "`_wf_mean` = mean over wf1-wf3.", "", to_markdown(table)]
        (C.REPORTS_DIR / "experiments.md").write_text("\n".join(md) + "\n")
    print(to_markdown(table))


def _write_main_fold_outputs(name: str, preds: dict) -> None:
    """Fold-level predictions + breakdowns for the selected model on the main fold and city fold."""
    _, test = fold_data(MAIN_FOLD)
    out = test[["load_id", "date", "equipment", "distance", C.TARGET, "is_outlier", "lane_seen"]].rename(
        columns={C.TARGET: "actual"})
    out.insert(5, "predicted", np.round(preds[MAIN_FOLD], 2))
    out["model"] = name
    out.to_csv(C.REPORTS_DIR / "oof_sep_oct.csv", index=False)
    bd = breakdown(test, preds[MAIN_FOLD]).assign(fold=MAIN_FOLD)
    _, ctest = fold_data("city")
    bd_city = breakdown(ctest, preds["city"]).query("group == 'lane_seen'").assign(fold="city")
    pd.concat([bd, bd_city]).to_csv(C.REPORTS_DIR / "breakdown_selected.csv", index=False)


if __name__ == "__main__":
    main()
