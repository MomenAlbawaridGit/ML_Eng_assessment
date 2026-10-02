"""Model-diagnostic figures and the model-comparison table for the report.

Run from the repo root:
    .venv/bin/python -m src.plots_model

Reads the modeling outputs written by src.evaluate / src.train / src.predict:
    reports/experiments.csv, reports/experiments_per_fold.csv,
    reports/oof_sep_oct.csv (wf3 = Sep-Oct predictions of the selected ridge_spline),
    data/december_chart_inputs.csv (filled), data/processed/market_index_daily.csv
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
    BLACK, BLUE, DIST_BINS, DIST_LABELS, EQUIP_COLORS, EQUIP_ORDER, GREY, ORANGE, ROOT,
    SKY, VERMIL, WIDTH, load, save, setup_style,
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
def fig10_comparison(exp: pd.DataFrame) -> None:
    keys = [k for k, _ in COMPARISON]
    labels = [lab for _, lab in COMPARISON]
    d = exp.loc[keys]
    y = np.arange(len(keys))[::-1]  # first row at the top
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    colors = [VERMIL if k == SELECTED else (GREY if k.startswith("baseline") else SKY)
              for k in keys]
    ax.barh(y, d.MAPE_wf_mean, color=colors, height=0.62, zorder=2)
    for f, mk in [("wf1", "o"), ("wf2", "s"), ("wf3", "^")]:
        ax.scatter(d[f"MAPE_{f}"], y, marker=mk, s=16, facecolor="white", edgecolor=BLACK,
                   lw=0.8, zorder=4, label=f"Fold {f}")
    ax.scatter(d.MAPE_city, y, marker="D", s=22, color=ORANGE, edgecolor=BLACK, lw=0.6,
               zorder=5, label="City holdout")
    right = d[["MAPE_wf_mean", "MAPE_wf1", "MAPE_wf2", "MAPE_wf3", "MAPE_city"]].max(axis=1)
    for yi, k, v, xr in zip(y, keys, d.MAPE_wf_mean, right):
        ax.text(xr + 0.1, yi, f"{v:.2f}%", va="center", fontsize=8,
                fontweight="bold" if k == SELECTED else "normal")
    ax.set_yticks(y, labels)
    for tl, k in zip(ax.get_yticklabels(), keys):
        if k == SELECTED:
            tl.set_fontweight("bold")
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("MAPE, % (outlier rows excluded); bar = mean of 3 walk-forward folds")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1, fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor="#cccccc")
    sel, base = exp.loc[SELECTED], exp.loc["baseline_b_lane"]
    fig.suptitle(f"Ridge + splines cuts error from {base.MAPE_wf_mean:.2f}% (lane median) to "
                 f"{sel.MAPE_wf_mean:.2f}%,\nbetter on every fold and on unseen cities; "
                 "tuned trees are no better", fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, "10_model_comparison.png")


def fig11_pred_vs_actual(oof: pd.DataFrame) -> None:
    clean, out = oof[~oof.is_outlier], oof[oof.is_outlier]
    mape_c = pct_err(clean).abs().mean()
    mape_a = pct_err(oof).abs().mean()
    print(f"[11] n={len(oof)}, outliers={len(out)}, MAPE excl {mape_c:.2f}% incl {mape_a:.2f}%")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.8, 4.6))
    ax.scatter(clean.actual, clean.predicted, s=3, alpha=0.25, color=BLUE, lw=0,
               label=f"Normal loads (n={len(clean):,})", rasterized=True)
    ax.scatter(out.actual, out.predicted, s=12, alpha=0.9, color=VERMIL, marker="x", lw=0.9,
               label=f"Flagged price outliers (n={len(out)})")
    lo = min(oof.actual.min(), oof.predicted.min()) * 0.8
    hi = max(oof.actual.max(), oof.predicted.max()) * 1.25
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=BLACK, lw=0.9, label="Perfect prediction")
    ax.fill_between(xs, xs / 1.05, xs * 1.05, color=GREY, alpha=0.2, lw=0, label="±5%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator([100, 300, 1000, 3000, 10000]))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Actual posted rate ($ per load, log scale)")
    ax.set_ylabel("Predicted rate ($ per load, log scale)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title(f"Sep–Oct test fold: MAPE {mape_c:.2f}% on normal loads, {mape_a:.2f}% "
                 "with outliers;\noutliers are mispriced loads the model correctly ignores",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "11_pred_vs_actual.png")


def fig12_residuals(oof: pd.DataFrame) -> None:
    c = oof[~oof.is_outlier].copy()
    c["pe"] = pct_err(c)
    c["ape"] = c.pe.abs()
    daily = c.groupby("date").pe.agg(["mean", "count",
                                      lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    daily.columns = ["mean", "n", "q25", "q75"]
    bias = c.pe.mean()
    by_month = c.groupby(c.date.dt.month).pe.mean()
    print(f"[12] overall bias {bias:+.2f}%, by month {by_month.round(2).to_dict()}, "
          f"daily mean range {daily['mean'].min():+.2f}..{daily['mean'].max():+.2f}")
    # linear drift over the fold, % per 30 days
    t = (daily.index - daily.index[0]).days.values
    slope = np.polyfit(t, daily["mean"].values, 1)[0] * 30
    print(f"[12] trend in daily bias: {slope:+.2f} pp per 30 days")

    fig = plt.figure(figsize=(WIDTH, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, 1.7], hspace=0.55,
                          wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(daily.index, daily.q25, daily.q75, color=SKY, alpha=0.3, lw=0,
                    label="Daily IQR of load errors")
    ax.plot(daily.index, daily["mean"], color=BLUE, lw=1.3, marker="o", ms=2.2,
            label="Daily mean error")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.axhline(bias, color=VERMIL, lw=1, ls="--", label=f"Fold mean {bias:+.2f}%")
    ax.set_ylabel("Prediction error, %\n(pred − actual) / actual")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7)
    ax.set_title(f"Daily mean error (bias): Sep {by_month[9]:+.2f}%, Oct {by_month[10]:+.2f}%",
                 fontsize=9, fontweight="normal", loc="left")
    worst = daily["mean"].idxmin()
    ax.annotate("Actual prices climb ~4% in late Sep and\nreset on Oct 1; market_index "
                "does not move,\nso no input can explain it",
                xy=(worst, daily.loc[worst, "mean"]), xytext=(pd.Timestamp("2025-10-19"), -3.0),
                fontsize=7, color="#333333", ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.legend(loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    ax.set_ylim(min(-4, daily.q25.min() - 1.6), max(3, daily.q75.max() + 0.3))

    # error by equipment
    ax1 = fig.add_subplot(gs[1, 0])
    eq = c.groupby("equipment").agg(mape=("ape", "mean"), bias=("pe", "mean"), n=("pe", "size"))
    eq = eq.loc[EQUIP_ORDER]
    xb = np.arange(len(eq))
    ax1.bar(xb, eq.mape, color=[EQUIP_COLORS[e] for e in eq.index], width=0.65)
    for xi, (m, b) in enumerate(zip(eq.mape, eq.bias)):
        ax1.text(xi, m + 0.05, f"{m:.2f}%\nbias {b:+.1f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(xb, [f"{e}\n(n={n:,})" for e, n in zip(eq.index, eq.n)], fontsize=7)
    ax1.set_ylabel("MAPE, %")
    ax1.set_ylim(0, eq.mape.max() * 1.45)
    ax1.set_title("By equipment", fontsize=9, fontweight="normal", loc="left")
    ax1.grid(axis="x", visible=False)

    # error by distance bucket
    ax2 = fig.add_subplot(gs[1, 1])
    c["bucket"] = pd.cut(c.distance, DIST_BINS, right=False, labels=DIST_LABELS)
    db = c.groupby("bucket", observed=True).agg(mape=("ape", "mean"), bias=("pe", "mean"),
                                                 n=("pe", "size"))
    xb = np.arange(len(db))
    ax2.bar(xb, db.mape, color=GREY, width=0.65)
    for xi, (m, b) in enumerate(zip(db.mape, db.bias)):
        ax2.text(xi, m + 0.05, f"{m:.2f}%\n{b:+.1f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(xb, [f"{lab}\n{n / 1000:.1f}k" for lab, n in zip(db.index, db.n)], fontsize=6.5)
    ax2.set_xlabel("Distance bucket (miles) and number of loads")
    ax2.set_ylim(0, db.mape.max() * 1.45)
    ax2.set_title("By distance (label: MAPE, bias %)", fontsize=9, fontweight="normal",
                  loc="left")
    ax2.grid(axis="x", visible=False)
    print("[12] by equipment:\n" + eq.round(2).to_string())
    print("[12] by distance:\n" + db.round(2).to_string())

    fig.suptitle(f"Sep–Oct: {bias:+.1f}% mean bias from month-level price moves the inputs do not "
                 "explain\n(worst: a late-Sep ramp); errors are even across equipment and distance "
                 f"(all ≤{max(eq.mape.max(), db.mape.max()):.1f}% MAPE, outliers excluded)",
                 fontsize=9.5, y=1.0)
    fig.subplots_adjust(top=0.86)
    save(fig, "12_residuals_over_time.png")


def fig13_december(mi: pd.DataFrame) -> None:
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    dec = dec.merge(mi[["date", "market_index_detrended"]], on="date", how="left")
    assert dec.market_index_detrended.notna().all(), "missing detrended MI for December"

    # Historical reference from the training data (same rule as fig 09).
    tr, va, _ = load()
    lane = tr[(((tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")) |
               ((tr.pickup == "Fort Wayne") & (tr.delivery == "Lexington")))
              & (tr.equipment == "Dry Van") & ~tr.outlier].copy()
    lane["rate360"] = lane.ppm * 360
    dec_mi = va[va.date.dt.month == 12]["market_index"]
    mlo, mhi = dec_mi.quantile([0.05, 0.95])
    like = lane[lane.market_index.between(mlo, mhi)]
    q25, q75 = like.rate360.quantile([0.25, 0.75])
    so = lane[lane.date >= "2025-09-01"]
    so_mean = so.rate360.mean()
    print(f"[13] Dec-like lane IQR ${q25:.0f}–${q75:.0f} (n={len(like)}); "
          f"Sep–Oct lane mean ${so_mean:.0f} (n={len(so)})")
    p = dec.predicted_rate
    r = np.corrcoef(p, dec.market_index_detrended)[0, 1]
    thu = dec[dec.date.dt.dayofweek == 3]
    print(f"[13] predicted mean ${p.mean():.0f}, range ${p.min():.0f}–${p.max():.0f}; "
          f"corr with detrended MI {r:.3f}; Thursdays {thu.predicted_rate.round(0).tolist()}")
    by_dow = dec.groupby(dec.date.dt.dayofweek).predicted_rate.mean()
    print(f"[13] mean by weekday (Mon=0): {by_dow.round(1).to_dict()}")

    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    ax.axhspan(q25, q75, color=GREY, alpha=0.18, lw=0,
               label=f"History at Dec-like market_index: IQR ${q25:.0f}–${q75:.0f} (n={len(like)})")
    ax.axhline(so_mean, color=GREY, lw=1, ls="--",
               label=f"Sep–Oct lane average ${so_mean:.0f} (n={len(so)})")
    ax.plot(dec.date, p, color=BLUE, lw=1.8, marker="o", ms=3, zorder=4,
            label=f"Predicted rate (mean ${p.mean():.0f})")
    ax.scatter(thu.date, thu.predicted_rate, s=55, facecolor="none", edgecolor=VERMIL, lw=1.4,
               zorder=5, label="Thursday peaks")
    xmas = dec[dec.date == "2025-12-25"]
    if len(xmas):
        ax.annotate("Dec 25 is a Thursday:\npeak comes from the weekly\nmarket_index cycle, "
                    "not a holiday", xy=(xmas.date.iloc[0], xmas.predicted_rate.iloc[0]),
                    xytext=(pd.Timestamp("2025-12-17"), q25 - 8), fontsize=7, color="#333333",
                    ha="center", arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.set_ylabel("Predicted rate ($ per load)")
    lo = min(q25, p.min()) - 22
    hi = max(q75, p.max()) + 8
    ax.set_ylim(lo, hi)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(dec.date, dec.market_index_detrended, color=ORANGE, lw=1, ls=":", marker="s",
             ms=2, label="Detrended market_index (right axis)")
    ax2.set_ylabel("Detrended market_index (index units)", color="#8a5d00")
    ax2.tick_params(axis="y", colors="#8a5d00")
    ax2.grid(False)
    mi_d = dec.market_index_detrended
    span = mi_d.max() - mi_d.min()
    ax2.set_ylim(mi_d.min() - span * 1.1, mi_d.max() + span * 0.15)

    ax.set_xticks(dec.date)
    ax.set_xticklabels([f"{d.day}\n{d.strftime('%a')[0]}" for d in dec.date], fontsize=6.5)
    for tl, d in zip(ax.get_xticklabels(), dec.date):
        if d.dayofweek == 3:
            tl.set_color(VERMIL)
            tl.set_fontweight("bold")
    ax.set_xlim(mdates.date2num(dec.date.min()) - 0.6, mdates.date2num(dec.date.max()) + 0.6)
    ax.set_xlabel("December 2025 (day, weekday initial; Thursdays in red)")
    ax.grid(axis="x", visible=False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              fontsize=7.5)
    fig.suptitle(f"Lexington→Fort Wayne Dry Van in December: a flat ${p.min():.0f}–${p.max():.0f} "
                 "weekly cycle\nthat tracks market_index (Thursday peaks); "
                 "the data contain no holiday spike", fontsize=10, y=1.0)
    ax.set_title("360 mi, 32,000 lb. Level sits inside the lane's historical range.",
                 fontsize=7.5, fontweight="normal", color="#444444")
    fig.tight_layout()
    save(fig, "13_december_explained.png")


def main() -> None:
    setup_style()
    exp = pd.read_csv(REPORTS / "experiments.csv").set_index("experiment")
    oof = pd.read_csv(REPORTS / "oof_sep_oct.csv", parse_dates=["date"])
    assert set(oof.model) == {SELECTED}, oof.model.unique()
    mi = pd.read_csv(DATA / "processed" / "market_index_daily.csv", parse_dates=["date"])
    write_table(exp)
    fig10_comparison(exp)
    fig11_pred_vs_actual(oof)
    fig12_residuals(oof)
    fig13_december(mi)


if __name__ == "__main__":
    main()
