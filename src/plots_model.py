"""Model-diagnostic figures and the model-comparison table for the report.

Run from the repo root:
    .venv/bin/python -m src.plots_model

Reads the modeling outputs written by src.evaluate / src.train / src.predict:
    reports/experiments.csv, reports/experiments_per_fold.csv,
    reports/oof_sep_oct.csv (wf3 = Sep-Oct predictions of the selected ridge_spline),
    data/december_chart_inputs.csv (filled)
and writes figures 10-13 to reports/figures/ plus reports/model_comparison_table.md.
Style matches src/plots_eda.py (Okabe-Ito palette, 6.5 in wide, 170 dpi).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.plots_eda import (  # noqa: E402
    BLACK, BLUE, EQUIP_COLORS, EQUIP_ORDER, GREY, ROOT, SKY, SUB_COLOR, VERMIL, WIDTH, load,
    money, save, setup_style, titles,
)

REPORTS = ROOT / "reports"
DATA = ROOT / "data"
SELECTED = "ridge_spline"

# Rows shown in fig 10 and the markdown table: (experiment id, display label).
COMPARISON = [
    ("baseline_a_eq_dist", "(a) Equipment × distance median"),
    ("baseline_b_lane", "(b) Lane × equipment median"),
    ("baseline_c2_lane_x_mi_detrended", "(c′) (b) × detrended market_index"),
    ("rf_mi_offset", "Random forest (MI offset)"),
    ("hgb_mi_offset", "HistGB (MI offset)"),
    ("ridge_det", "Ridge, linear terms"),
    ("catboost_mi_offset", "CatBoost (MI offset)"),
    (SELECTED, "Ridge + splines (selected)"),
]
# Extra rows for the markdown table only (context for the decisions log).
TABLE_EXTRA = [
    ("ridge_log_ppm", "Ridge, MI level + detrended + day of week (locked set)"),
    ("ridge_spline_hgb_resid", "Ridge + splines + HistGB on residual"),
]


def pct_err(df: pd.DataFrame) -> pd.Series:
    """Signed % error, positive = over-prediction."""
    return (df["predicted"] - df["actual"]) / df["actual"] * 100


# --------------------------------------------------------------------------- table
def write_table(exp: pd.DataFrame) -> None:
    rows = []
    for key, label in COMPARISON + TABLE_EXTRA:
        r = exp.loc[key]
        name = f"**{label}**" if key == SELECTED else label
        vals = [r.MAPE_wf1, r.MAPE_wf2, r.MAPE_wf3, r.MAPE_wf_mean, r.MAPE_all_wf_mean, r.MAPE_city]
        cells = [f"{v:.2f}" for v in vals]
        if key == SELECTED:
            cells = [f"**{c}**" for c in cells]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join([
        "MAPE (%) by validation fold. wf1: train ≤Jun → test Jul–Aug; wf2: ≤Jul → Aug–Sep; "
        "wf3: ≤Aug → Sep–Oct. Outlier rows (1.4%) excluded except in *Incl. outliers* "
        "(mean of wf1–wf3). *City holdout*: trained Jan–Aug without 8 held-out cities, "
        "tested on Sep–Oct loads touching them. Source: `reports/experiments.csv`.",
        "",
        "| Model | wf1 | wf2 | wf3 | Mean | Incl. outliers | City holdout |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ])
    path = REPORTS / "model_comparison_table.md"
    path.write_text(text)
    print(f"  saved {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- figures
# Rows shown in fig 10: (experiment id, plain label). Top to bottom.
FIG10_ROWS = [
    ("baseline_b_lane", "Simple route average\n(baseline)"),
    ("rf_mi_offset", "Random forest"),
    ("hgb_mi_offset", "Gradient boosting\n(HistGB)"),
    ("catboost_mi_offset", "CatBoost"),
    (SELECTED, "Ridge (our model)"),
]


def fig10_comparison(exp: pd.DataFrame) -> None:
    keys = [k for k, _ in FIG10_ROWS]
    vals = exp.loc[keys, "MAPE_wf_mean"].values
    print("[10] " + ", ".join(f"{k} {v:.2f}" for k, v in zip(keys, vals)))
    y = np.arange(len(keys))[::-1]
    colors = [GREY if k.startswith("baseline") else (VERMIL if k == SELECTED else SKY)
              for k in keys]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, k, v in zip(y, keys, vals):
        ax.text(v + 0.05, yi, f"{v:.2f}%", va="center", fontsize=11,
                fontweight="bold" if k == SELECTED else "normal")
    ax.set_yticks(y, [lab for _, lab in FIG10_ROWS])
    for tl, k in zip(ax.get_yticklabels(), keys):
        if k == SELECTED:
            tl.set_fontweight("bold")
    ax.set_xlim(0, vals.max() * 1.18)
    ax.set_xlabel("Average error (%)")
    ax.grid(axis="y", visible=False)
    titles(ax, "Our model nearly halves the baseline's error",
           "Average over three test periods, Jul to Oct 2025. Lower is better.")
    fig.tight_layout()
    save(fig, "10_model_comparison.png")


def fig11_pred_vs_actual(oof: pd.DataFrame) -> None:
    clean, out = oof[~oof.is_outlier], oof[oof.is_outlier]
    mape_c = pct_err(clean).abs().mean()
    print(f"[11] n={len(oof)}, wrong prices={len(out)}, average error normal loads {mape_c:.2f}%")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.85, 5.0))
    ax.scatter(clean.actual, clean.predicted, s=4, alpha=0.3, color=BLUE, lw=0,
               rasterized=True)
    ax.scatter(out.actual, out.predicted, s=16, alpha=0.9, color=VERMIL, marker="x", lw=1.0)
    lo, hi = 100, 15000
    ax.plot([lo, hi], [lo, hi], color=BLACK, lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator([100, 300, 1000, 3000, 10000]))
        axis.set_major_formatter(money)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.text(700, 230, f"{len(clean):,} normal loads:\n{mape_c:.1f}% average error",
            color=BLUE, fontsize=11, fontweight="bold", ha="left", va="top")
    ax.text(130, 7000, f"{len(out)} wrong prices\nin the data", color=VERMIL, fontsize=11,
            fontweight="bold", ha="left", va="top")
    ax.set_xlabel("Real price ($ per load, log scale)")
    ax.set_ylabel("Predicted price ($ per load, log scale)")
    titles(ax, "Predictions match real prices closely",
           "Test loads from Sep and Oct 2025")
    fig.tight_layout()
    save(fig, "11_pred_vs_actual.png")


def fig12_residuals(oof: pd.DataFrame) -> None:
    c = oof[~oof.is_outlier].copy()
    c["ape"] = pct_err(c).abs()
    eq = c.groupby("equipment").ape.mean().loc[EQUIP_ORDER]
    print("[12] average error by truck type: " + ", ".join(f"{e} {v:.2f}" for e, v in eq.items()))
    x = np.arange(len(eq))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5))
    ax.bar(x, eq.values, color=[EQUIP_COLORS[e] for e in eq.index], width=0.6)
    for xi, v in zip(x, eq.values):
        ax.text(xi, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold")
    ax.set_xticks(x, eq.index, fontsize=11.5)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Average error (%)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Error is low for every truck type",
           "Test loads from Sep and Oct 2025, wrong prices excluded")
    fig.tight_layout()
    save(fig, "12_residuals_over_time.png")


def fig13_december() -> None:
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    # Historical band: middle half of past Lexington/Fort Wayne Dry Van loads (scaled to
    # 360 mi) at market_index levels like December's (same rule as before, fig 09 data).
    tr, va, _ = load()
    lane = tr[(((tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")) |
               ((tr.pickup == "Fort Wayne") & (tr.delivery == "Lexington")))
              & (tr.equipment == "Dry Van") & ~tr.outlier].copy()
    lane["rate360"] = lane.ppm * 360
    dec_mi = va[va.date.dt.month == 12]["market_index"]
    mlo, mhi = dec_mi.quantile([0.05, 0.95])
    like = lane[lane.market_index.between(mlo, mhi)]
    q25, q75 = like.rate360.quantile([0.25, 0.75])
    p = dec.predicted_rate
    thu = dec[dec.date.dt.dayofweek == 3]
    print(f"[13] band ${q25:.0f} to ${q75:.0f} (n={len(like)}); predicted ${p.min():.0f} to "
          f"${p.max():.0f}, mean ${p.mean():.0f}; Thursdays {thu.predicted_rate.round(0).tolist()}")

    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    ax.axhspan(q25, q75, color=GREY, alpha=0.18, lw=0)
    ax.text(dec.date.iloc[0], q25 + 2, f"Normal range for past loads: ${q25:.0f} to ${q75:.0f}",
            fontsize=10.5, color=SUB_COLOR, va="bottom")
    ax.plot(dec.date, p, color=BLUE, lw=2.2, marker="o", ms=3.5, zorder=4)
    ax.scatter(thu.date, thu.predicted_rate, s=70, color=VERMIL, zorder=5)
    for d, v in zip(thu.date, thu.predicted_rate):
        ax.annotate("Thu", (d, v), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=10, color=VERMIL, fontweight="bold")
    xmas = dec[dec.date == "2025-12-25"]
    if len(xmas):
        ax.annotate("Dec 25 is a Thursday", (xmas.date.iloc[0], xmas.predicted_rate.iloc[0]),
                    xytext=(pd.Timestamp("2025-12-27"), q25 + 14), fontsize=10,
                    color=SUB_COLOR, ha="center",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
    ax.set_ylim(q25 - 6, max(q75, p.max()) + 12)
    ax.yaxis.set_major_formatter(money)
    ax.set_ylabel("Predicted price ($ per load)")
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("Dec %-d"))
    ax.set_xlim(mdates.date2num(dec.date.min()) - 0.8, mdates.date2num(dec.date.max()) + 0.8)
    ax.grid(axis="x", visible=False)
    titles(ax, "December prices follow a weekly pattern, highest on Thursdays",
           f"Lexington to Fort Wayne, Dry Van, 360 miles: ${p.min():.0f} to ${p.max():.0f} a load")
    ax.title.set_fontsize(13)
    fig.tight_layout()
    save(fig, "13_december_explained.png")


def main() -> None:
    setup_style()
    exp = pd.read_csv(REPORTS / "experiments.csv").set_index("experiment")
    oof = pd.read_csv(REPORTS / "oof_sep_oct.csv", parse_dates=["date"])
    assert set(oof.model) == {SELECTED}, oof.model.unique()
    write_table(exp)
    fig10_comparison(exp)
    fig11_pred_vs_actual(oof)
    fig12_residuals(oof)
    fig13_december()


if __name__ == "__main__":
    main()
