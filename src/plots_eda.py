"""EDA and data-quality figures for the freight-rate report.

Run from the repo root:
    .venv/bin/python -m src.plots_eda

Works from the raw CSVs in data/ (independent of src/data.py) and writes PNGs to
reports/figures/. Each figure is sized for a 6.5in-wide Word page at 170 dpi.
Key numbers are printed to stdout so they can be quoted in the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "reports" / "figures"
DPI = 170
WIDTH = 6.5

# Outlier cutoffs on ppm / reference ppm (matches src/config.py OUTLIER_LOW/HIGH).
RATIO_LO, RATIO_HI = 0.5, 2.0
LANE_MIN_SUPPORT = 5  # lanes with fewer rows fall back to equipment x distance bucket

# Okabe-Ito colorblind-safe palette
BLUE, ORANGE, GREEN, VERMIL = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PINK, GREY, BLACK = "#56B4E9", "#CC79A7", "#999999", "#000000"
EQUIP_COLORS = {"Dry Van": BLUE, "Reefer": ORANGE, "Flatbed": GREEN}
EQUIP_ORDER = ["Dry Van", "Reefer", "Flatbed"]

# Real coordinates. The three report cities and Hartford are given by the brief/team;
# the remainder are approximate public city-centre coordinates (+/- ~0.05 deg), used only
# for the "all cities" error summary.
REAL_COORDS = {
    "Los Angeles": (34.0522, -118.2437), "Tampa": (27.9506, -82.4572),
    "Providence": (41.8240, -71.4128), "Hartford": (41.7658, -72.6734),
    "Albany": (42.6526, -73.7562), "Albuquerque": (35.0844, -106.6504),
    "Amarillo": (35.2220, -101.8313), "Atlanta": (33.7490, -84.3880),
    "Austin": (30.2672, -97.7431), "Bakersfield": (35.3733, -119.0187),
    "Baltimore": (39.2904, -76.6122), "Baton Rouge": (30.4515, -91.1871),
    "Birmingham": (33.5186, -86.8104), "Boston": (42.3601, -71.0589),
    "Buffalo": (42.8864, -78.8784), "Charleston": (32.7765, -79.9311),
    "Chattanooga": (35.0456, -85.3097), "Cincinnati": (39.1031, -84.5120),
    "Columbia": (34.0007, -81.0348), "Corpus Christi": (27.8006, -97.3964),
    "Dallas": (32.7767, -96.7970), "Dayton": (39.7589, -84.1916),
    "Detroit": (42.3314, -83.0458), "El Paso": (31.7619, -106.4850),
    "Fort Wayne": (41.0793, -85.1394), "Fresno": (36.7378, -119.7871),
    "Grand Rapids": (42.9634, -85.6681), "Green Bay": (44.5192, -88.0198),
    "Greensboro": (36.0726, -79.7920), "Harrisburg": (40.2732, -76.8867),
    "Houston": (29.7604, -95.3698), "Indianapolis": (39.7684, -86.1581),
    "Jacksonville": (30.3322, -81.6557), "Kansas City": (39.0997, -94.5786),
    "Las Vegas": (36.1699, -115.1398), "Lexington": (38.0406, -84.5037),
    "Little Rock": (34.7465, -92.2896), "Louisville": (38.2527, -85.7585),
    "Lubbock": (33.5779, -101.8552), "Madison": (43.0731, -89.4012),
    "Memphis": (35.1495, -90.0490), "Milwaukee": (43.0389, -87.9065),
    "Mobile": (30.6954, -88.0399), "Montgomery": (32.3792, -86.3077),
    "Nashville": (36.1627, -86.7816), "New Orleans": (29.9511, -90.0715),
    "New York": (40.7128, -74.0060), "Oklahoma City": (35.4676, -97.5164),
    "Philadelphia": (39.9526, -75.1652), "Phoenix": (33.4484, -112.0740),
    "Raleigh": (35.7796, -78.6382), "Reno": (39.5296, -119.8138),
    "Richmond": (37.5407, -77.4360), "Salt Lake City": (40.7608, -111.8910),
    "San Antonio": (29.4241, -98.4936), "San Francisco": (37.7749, -122.4194),
    "Savannah": (32.0809, -81.0912), "Shreveport": (32.5252, -93.7502),
    "St. Louis": (38.6270, -90.1994), "Syracuse": (43.0481, -76.1474),
    "Toledo": (41.6528, -83.5379), "Tucson": (32.2226, -110.9747),
    "Tulsa": (36.1540, -95.9928), "Washington": (38.9072, -77.0369),
    # validation-only cities
    "Allentown": (40.6084, -75.4902), "Charlotte": (35.2271, -80.8431),
    "Chicago": (41.8781, -87.6298), "Jackson": (32.2988, -90.1848),
    "Knoxville": (35.9606, -83.9207), "Laredo": (27.5306, -99.4803),
    "Norfolk": (36.8508, -76.2859), "San Diego": (32.7157, -117.1611),
}
REPORT_CITIES = ["Los Angeles", "Tampa", "Providence"]


# --------------------------------------------------------------------------- helpers
def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "text.parse_math": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def subtitle(fig: plt.Figure, text: str, y: float = 0.995) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=8.5, color="#444444")


def haversine_mi(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    for d in (tr, va):
        d["weight_fixed"] = d["weight"].abs()  # sign errors
    tr["ppm"] = tr["posted_rate"] / tr["distance"]
    # Same rule as src/data.py (reports/data_findings.md 3d), re-implemented so this script
    # runs from the raw CSVs: lane (pickup x delivery x equipment) median when the lane has
    # >= LANE_MIN_SUPPORT rows, otherwise equipment x distance-bucket median.
    lane = tr.groupby(["pickup", "delivery", "equipment"])["ppm"]
    bucket = pd.cut(tr["distance"], DIST_BINS, right=False)
    fallback = tr.groupby([tr["equipment"], bucket], observed=True)["ppm"].transform("median")
    ref = lane.transform("median").where(lane.transform("size") >= LANE_MIN_SUPPORT, fallback)
    tr["ratio"] = tr["ppm"] / ref
    tr["outlier"] = (tr["ratio"] < RATIO_LO) | (tr["ratio"] > RATIO_HI)
    return tr, va, dec


def city_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    b = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([a, b]).groupby("city")[["lat", "lon"]].first()


DIST_BINS = [0, 250, 500, 1000, 1500, 2000, np.inf]
DIST_LABELS = ["<250", "250–500", "500–1k", "1k–1.5k", "1.5k–2k", "2k+"]


# --------------------------------------------------------------------------- figures
def fig01_outliers(tr: pd.DataFrame) -> None:
    n_out = int(tr["outlier"].sum())
    lo_n = int((tr["ratio"] < RATIO_LO).sum())
    hi_n = int((tr["ratio"] > RATIO_HI).sum())
    print(f"[01] flagged {n_out} rows ({n_out / len(tr):.2%}): {lo_n} below {RATIO_LO}, "
          f"{hi_n} above {RATIO_HI}")
    clean = tr.loc[~tr["outlier"]]
    kept = tr.loc[~tr["outlier"], "ratio"]
    print(f"[01] kept rows ratio range {kept.min():.2f}-{kept.max():.2f}; flagged ranges "
          f"{tr.loc[tr.ratio < RATIO_LO, 'ratio'].agg(['min', 'max']).round(2).tolist()} and "
          f"{tr.loc[tr.ratio > RATIO_HI, 'ratio'].agg(['min', 'max']).round(2).tolist()}")
    bins = np.logspace(np.log10(0.1), np.log10(10), 120)

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.4), sharex=True)
    for ax, d, lab, col in [(axes[0], tr, f"Before: all {len(tr):,} training loads", GREY),
                            (axes[1], clean, f"After: {len(clean):,} loads ({n_out:,} removed)", BLUE)]:
        ax.hist(d["ratio"], bins=bins, color=col, edgecolor="none")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylabel("Loads (log scale)")
        ax.set_title(lab, fontsize=9, fontweight="normal", loc="left")
        for c in (RATIO_LO, RATIO_HI):
            ax.axvline(c, color=VERMIL, ls="--", lw=1)
    axes[0].text(RATIO_LO, axes[0].get_ylim()[1] * 0.5, f"  {lo_n} loads < {RATIO_LO}×",
                 color=VERMIL, ha="right", va="top", fontsize=8)
    axes[0].text(RATIO_HI, axes[0].get_ylim()[1] * 0.5, f"  {hi_n} loads > {RATIO_HI}×",
                 color=VERMIL, ha="left", va="top", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    axes[1].set_xticklabels(["0.1×", "0.2×", "0.5×", "1×", "2×", "5×", "10×"])
    axes[1].set_xlabel("Rate per mile ÷ typical rate per mile for the same lane × equipment")
    fig.suptitle(f"{n_out} loads ({n_out / len(tr):.1%}) are priced at 0.2–0.5× or 2–5× their lane's "
                 "norm; the gap is clean,\nso removing them leaves a tight distribution", y=1.08)
    subtitle(fig, f"Cutoffs: ratio < {RATIO_LO} or > {RATIO_HI} (dashed). "
                  "Typical = training median of the lane (≥5 loads) or of equipment × distance band.", y=1.0)
    fig.tight_layout()
    save(fig, "01_outliers_before_after.png")


def fig02_quote_signal(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    """quote_signal vs rate per mile, split by month regime.

    In some months quote_signal ~= ppm (target copy), in others ~= 2c - ppm (mirror),
    and in August plus all of validation Nov-Dec it is unrelated noise.
    """
    d = tr.loc[~tr["outlier"]].copy()
    d["month"] = d["date"].dt.month
    r_all = d["quote_signal"].corr(d["ppm"])
    d["diff"] = d["quote_signal"] - d["ppm"]
    # classify each month by its behaviour
    copy_share = d.groupby("month")["diff"].apply(lambda x: (x.abs() < 0.03).mean())
    print("[02] share of loads with |quote_signal - ppm| < $0.03 by month: "
          + ", ".join(f"{m}:{v:.2f}" for m, v in copy_share.items()))

    def signature(df: pd.DataFrame) -> pd.Series:
        """Mean quote_signal of short (<300 mi) minus long (>2000 mi) hauls, per month."""
        g = df.groupby(df["date"].dt.to_period("M"))
        return g.apply(lambda s: s.loc[s.distance < 300, "quote_signal"].mean()
                       - s.loc[s.distance > 2000, "quote_signal"].mean())

    sig_tr, sig_va = signature(tr), signature(va)
    print("[02] short-minus-long quote_signal: train "
          + ", ".join(f"{k.strftime('%b')} {v:+.2f}" for k, v in sig_tr.items())
          + " | val " + ", ".join(f"{k.strftime('%b')} {v:+.2f}" for k, v in sig_va.items()))
    print(f"[02] quote_signal sd: Aug {tr.loc[tr.date.dt.month == 8, 'quote_signal'].std():.3f}, "
          f"validation {va['quote_signal'].std():.3f}, train other months "
          f"{tr.loc[tr.date.dt.month != 8, 'quote_signal'].std():.3f}")
    copy_m = [m for m in range(1, 11) if sig_tr.iloc[m - 1] > 0.3]
    mirror_m = [m for m in range(1, 11) if sig_tr.iloc[m - 1] < -0.3]
    noise_m = [m for m in range(1, 11) if m not in copy_m + mirror_m]
    mn = lambda ms: ", ".join(pd.Timestamp(2025, m, 1).strftime("%b") for m in ms)  # noqa: E731
    groups = [(copy_m, f"Copy months ({mn(copy_m)})", BLUE),
              (mirror_m, f"Mirror months ({mn(mirror_m)})", ORANGE),
              (noise_m, f"Noise month ({mn(noise_m)})", GREY)]

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.4))
    axes = axes.ravel()
    xx = np.array([1.4, 3.4])
    for ax, (ms, lab, col) in zip(axes[:3], groups):
        s = d[d["month"].isin(ms)]
        s = s.sample(min(6000, len(s)), random_state=0)
        ax.scatter(s["ppm"], s["quote_signal"], s=2, alpha=0.25, color=col, edgecolor="none")
        ax.plot(xx, xx, color=BLACK, lw=0.7, ls=":")
        ax.plot(xx, 4.15 - xx, color=BLACK, lw=0.7, ls=":")
        r = d.loc[d["month"].isin(ms), "quote_signal"].corr(d.loc[d["month"].isin(ms), "ppm"])
        ax.set_title(f"{lab}: r = {r:+.2f}", fontsize=8.5, fontweight="normal")
        ax.set_xlim(1.4, 3.4)
        ax.set_ylim(0.7, 3.6)
        ax.set_xlabel("Rate per mile ($/mile)")
        ax.set_ylabel("quote_signal")
    axes[0].text(3.3, 3.35, "quote_signal = $/mile", ha="right", fontsize=7, rotation=0)
    axes[1].text(3.3, 0.85, "quote_signal = 4.15 − $/mile", ha="right", fontsize=7)
    ax = axes[3]
    labels = [k.strftime("%b")[0] for k in sig_tr.index] + [k.strftime("%b")[0] for k in sig_va.index]
    labels = [f"{l}\n{i + 1}" for i, l in enumerate(labels)]
    vals = list(sig_tr.values) + list(sig_va.values)
    cols = [BLUE if v > 0.3 else ORANGE if v < -0.3 else GREY for v in sig_tr.values] + [VERMIL] * len(sig_va)
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        if abs(v) < 0.3:
            ax.text(i, 0.06, "≈0", ha="center", fontsize=7, color=cols[i])
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.axvline(9.5, color=BLACK, lw=0.6, ls="--")
    ax.text(10.5, 0.62, "validation", ha="center", fontsize=7.5, color=VERMIL)
    ax.set_ylim(-1, 1)
    ax.tick_params(axis="x", labelsize=6.5, rotation=0)
    ax.set_xlabel("Month of 2025")
    ax.set_ylabel("quote_signal: short − long hauls")
    ax.set_title("Nov–Dec behave like August: no signal", fontsize=8.5, fontweight="normal")
    fig.suptitle("quote_signal copies or mirrors $/mile in 9 of 10 training months, but is noise in "
                 "August\nand in Nov–Dec validation; overall r = "
                 f"{r_all:.2f} hides this. Drop it or the model will over-trust it.",
                 fontsize=9.5, y=1.01)
    fig.text(0.01, -0.02, "Training loads Jan–Oct 2025 (outliers removed), 6,000-load sample per "
             "panel. Short < 300 mi, long > 2,000 mi. Dotted lines: y = x and y = 4.15 − x.",
             fontsize=7, color="#444444")
    fig.tight_layout()
    save(fig, "02_quote_signal_vs_ppm.png")


def fig03_monthly(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].copy()
    daily = d.groupby("date").agg(ppm=("ppm", "mean"), mi=("market_index", "mean"))
    r_daily = daily["ppm"].corr(daily["mi"])
    mi_all = pd.concat([d[["date", "market_index"]], va[["date", "market_index"]]]) \
        .groupby("date")["market_index"].mean().asfreq("D")
    # 7-day centred rolling mean: removes the strong weekly cycle without partial-week artifacts
    mi_roll = mi_all.rolling(7, center=True, min_periods=7).mean()
    ppm_roll = daily["ppm"].asfreq("D").rolling(7, center=True, min_periods=7).mean()
    mo = d.set_index("date").resample("MS")["ppm"].mean()
    mi_mo_tr = d.set_index("date")["market_index"].resample("MS").mean()
    mi_mo_va = va.set_index("date")["market_index"].resample("MS").mean()
    print("[03] monthly ppm: " + ", ".join(f"{k:%b} {v:.2f}" for k, v in mo.items()))
    print("[03] monthly market_index train: " + ", ".join(f"{k:%b} {v:.3f}" for k, v in mi_mo_tr.items()))
    print("[03] monthly market_index val:   " + ", ".join(f"{k:%b} {v:.3f}" for k, v in mi_mo_va.items()))
    print(f"[03] daily corr(ppm, market_index) = {r_daily:.3f}")
    cut = pd.Timestamp("2025-11-01")

    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    ax.plot(ppm_roll.index, ppm_roll.values, color=BLUE, lw=1.0, alpha=0.6,
            label="$/mile, 7-day mean")
    ax.plot(mo.index + pd.DateOffset(days=14), mo.values, color=BLUE, lw=2.2, marker="o", ms=4,
            label="$/mile, monthly mean")
    ax.set_ylabel("Mean rate per mile ($/mile)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.plot(mi_all.loc[cut:].index, mi_all.loc[cut:].values, color=ORANGE, lw=0.7, alpha=0.6,
             label="market_index Nov–Dec, daily (weekly cycle)")
    ax2.plot(mi_roll.loc[:cut].index, mi_roll.loc[:cut].values, color=ORANGE, lw=1.6,
             label="market_index, 7-day mean (train)")
    ax2.plot(mi_roll.loc[cut - pd.DateOffset(days=1):].index,
             mi_roll.loc[cut - pd.DateOffset(days=1):].values, color=ORANGE, lw=1.8, ls="--",
             label="market_index, 7-day mean (to predict)")
    ax2.set_ylabel("market_index", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.axvspan(cut, pd.Timestamp("2025-12-31"), color=GREY, alpha=0.12)
    ax.text(pd.Timestamp("2025-12-01"), ax.get_ylim()[1], "Nov–Dec:\nno prices", ha="center",
            va="top", fontsize=8, color="#444444")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("2025")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17),
              fontsize=7.5)
    ax.set_title(f"$/mile peaked in June and eased; market_index tracks it (daily r = {r_daily:.2f})\n"
                 f"and sits near its Sep–Oct level ({mi_mo_va.iloc[0]:.2f}–{mi_mo_va.iloc[1]:.2f}) "
                 "in Nov–Dec", fontsize=9.5)
    fig.tight_layout()
    save(fig, "03_monthly_ppm_market_index.png")


def fig04_distance_equipment(tr: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].copy()
    d["dbin"] = pd.cut(d["distance"], DIST_BINS, labels=DIST_LABELS, right=False)
    g = d.groupby(["dbin", "equipment"], observed=True)["ppm"].agg(["median", "count"]).unstack()
    print("[04] median ppm by distance bucket x equipment:\n" + g["median"].round(2).to_string())
    eq = d.groupby("equipment")["ppm"].median()
    prem = {e: eq[e] / eq["Dry Van"] - 1 for e in EQUIP_ORDER}
    print(f"[04] overall median ppm premium vs Dry Van: {prem}")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    x = np.arange(len(DIST_LABELS))
    w = 0.26
    for i, e in enumerate(EQUIP_ORDER):
        ax.bar(x + (i - 1) * w, g["median"][e].values, w, color=EQUIP_COLORS[e],
               label=f"{e} (+{prem[e]:.0%})" if e != "Dry Van" else e)
    ax.set_xticks(x, DIST_LABELS)
    ax.set_xlabel("Distance bucket (miles)")
    ax.set_ylabel("Median rate per mile ($/mile)")
    ax.set_ylim(0, g["median"].values.max() * 1.15)
    ax.legend(ncol=3, loc="upper right")
    s, l = g["median"]["Dry Van"].iloc[0], g["median"]["Dry Van"].iloc[-1]
    ax.set_title(f"Short hauls cost more per mile (Dry Van ${s:.2f} <250 mi vs ${l:.2f} 2k+ mi); "
                 "Reefer > Flatbed > Dry Van", fontsize=9.5)
    fig.tight_layout()
    save(fig, "04_ppm_by_distance_equipment.png")


def fig05_coordinates(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    ct = city_table(pd.concat([tr, va]))
    known = ct.index.intersection(list(REAL_COORDS))
    real = pd.DataFrame(REAL_COORDS, index=["rlat", "rlon"]).T.loc[known]
    ct = ct.join(real)
    ct["err_mi"] = haversine_mi(ct["lat"], ct["lon"], ct["rlat"], ct["rlon"])
    ct["dlat"] = ct["lat"] - ct["rlat"]
    ct["dlon"] = ct["lon"] - ct["rlon"]
    print(f"[05] coordinate error over {ct['err_mi'].notna().sum()} cities: "
          f"median {ct['err_mi'].median():.0f} mi, mean {ct['err_mi'].mean():.0f}, "
          f"min {ct['err_mi'].min():.0f}, max {ct['err_mi'].max():.0f}")
    print(f"[05] mean lat shift {ct['dlat'].mean():+.2f} deg (sd {ct['dlat'].std():.2f}); "
          f"mean lon shift {ct['dlon'].mean():+.2f} deg (sd {ct['dlon'].std():.2f})")
    for c in REPORT_CITIES:
        r = ct.loc[c]
        print(f"[05] {c}: data ({r.lat:.4f}, {r.lon:.4f}) real ({r.rlat:.4f}, {r.rlon:.4f}) "
              f"-> {r.err_mi:.0f} mi off")
    print("[05] worst 5:\n" + ct.sort_values("err_mi").tail(5)[["err_mi"]].round(0).to_string())

    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    # faint arrows for all cities
    for c, r in ct.iterrows():
        if c in REPORT_CITIES or np.isnan(r.rlat):
            continue
        ax.annotate("", xy=(r.lon, r.lat), xytext=(r.rlon, r.rlat),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6, alpha=0.7))
    ax.scatter(ct["rlon"], ct["rlat"], s=10, facecolor="white", edgecolor=GREY, lw=0.8,
               label="Real position", zorder=3)
    ax.scatter(ct["lon"], ct["lat"], s=10, color=GREY, label="Position in data", zorder=3)
    offs = {"Los Angeles": (-6, -22), "Tampa": (-8, -22), "Providence": (-10, -18)}
    for c in REPORT_CITIES:
        r = ct.loc[c]
        ax.annotate("", xy=(r.lon, r.lat), xytext=(r.rlon, r.rlat),
                    arrowprops=dict(arrowstyle="-|>", color=VERMIL, lw=1.6), zorder=4)
        ax.scatter([r.rlon], [r.rlat], s=40, facecolor="white", edgecolor=VERMIL, lw=1.5, zorder=5)
        ax.scatter([r.lon], [r.lat], s=40, color=VERMIL, zorder=5)
        ax.annotate(f"{c}\n{r.err_mi:.0f} mi off", (r.lon, r.lat), xytext=offs[c],
                    textcoords="offset points", fontsize=8, color=VERMIL, fontweight="bold",
                    ha="left" if c == "Los Angeles" else "right")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_ylim(23.5, 46)
    ax.set_title(f"City coordinates in the data are shifted from reality "
                 f"(median {ct['err_mi'].median():.0f} mi across {len(known)} cities)", fontsize=10)
    fig.text(0.01, -0.06, "Arrows point from the real city position to the position in the data. "
             "Shifts point in random directions\n(not one consistent offset). "
             "Real positions: public city-centre coordinates.", fontsize=7, color="#444444")
    fig.tight_layout()
    save(fig, "05_coordinates_check.png")


def fig06_distance_haversine(tr: pd.DataFrame) -> None:
    d = tr.copy()
    d["hav_data"] = haversine_mi(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["ratio_data"] = d["distance"] / d["hav_data"]
    rp = d["pickup"].map(lambda c: REAL_COORDS.get(c, (np.nan, np.nan)))
    rd = d["delivery"].map(lambda c: REAL_COORDS.get(c, (np.nan, np.nan)))
    d["hav_real"] = haversine_mi(rp.str[0], rp.str[1], rd.str[0], rd.str[1])
    d["ratio_real"] = d["distance"] / d["hav_real"]
    med_data = d["ratio_data"].median()
    iqr_data = d["ratio_data"].quantile([0.25, 0.75]).values
    med_real = d["ratio_real"].median()
    iqr_real = d["ratio_real"].quantile([0.25, 0.75]).values
    print(f"[06] distance / haversine(data coords): median {med_data:.3f}, IQR {iqr_data.round(3)}, "
          f"corr {d['distance'].corr(d['hav_data']):.4f}")
    print(f"[06] distance / haversine(real coords): median {med_real:.3f}, IQR {iqr_real.round(3)}, "
          f"corr {d['distance'].corr(d['hav_real']):.4f}")
    pairs = [("Providence", "Hartford"), ("Hartford", "Providence"), ("Los Angeles", "Tampa"),
             ("Tampa", "Los Angeles"), ("Providence", "Tampa"), ("Hartford", "Tampa"),
             ("Los Angeles", "Providence"), ("Hartford", "Los Angeles"),
             ("Lexington", "Fort Wayne")]
    rows = []
    for a, b in pairs:
        s = d[(d.pickup == a) & (d.delivery == b)]
        if len(s):
            rows.append((f"{a}→{b}", len(s), s.distance.median(), s.hav_data.iloc[0],
                         s.hav_real.iloc[0]))
    pt = pd.DataFrame(rows, columns=["lane", "n", "distance", "hav_data", "hav_real"])
    pt["ratio_data"] = pt.distance / pt.hav_data
    pt["ratio_real"] = pt.distance / pt.hav_real
    print("[06] lane check:\n" + pt.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True)
    samp = d.sample(min(8000, len(d)), random_state=0)
    xmax = max(d.hav_data.max(), d.hav_real.max()) * 1.03
    for ax, col, med, lab, colr in [(axes[0], "hav_data", med_data, "data coordinates", BLUE),
                                    (axes[1], "hav_real", med_real, "real coordinates", VERMIL)]:
        ax.scatter(samp[col], samp["distance"], s=2, alpha=0.25, color=colr, edgecolor="none")
        xx = np.array([0, xmax])
        ax.plot(xx, med * xx, color=BLACK, lw=1, ls="--", label=f"distance = {med:.2f} × straight line")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, None)
        ax.set_xlabel(f"Straight-line miles, {lab}")
        ax.legend(loc="upper left")
        corr = d["distance"].corr(d[col])
        q = d["ratio_data" if col == "hav_data" else "ratio_real"].quantile([0.05, 0.95])
        ax.set_title(f"Using {lab}: r = {corr:.3f}\n90% of ratios in {q.iloc[0]:.2f}–{q.iloc[1]:.2f}",
                     fontsize=9, fontweight="normal")
    # highlight Providence->Hartford in both panels
    ph = d[(d.pickup == "Providence") & (d.delivery == "Hartford")]
    if len(ph):
        for ax, col in [(axes[0], "hav_data"), (axes[1], "hav_real")]:
            r = ph.iloc[0]
            ax.scatter([r[col]], [ph.distance.median()], s=30, color=BLACK, zorder=5)
            ax.annotate(f"Providence→Hartford\n{ph.distance.median():.0f} mi vs "
                        f"{r[col]:.0f} straight\n= {ph.distance.median() / r[col]:.1f}×",
                        (r[col], ph.distance.median()), xytext=(1250, 250), textcoords="data",
                        fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6),
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
    lt1 = (d["ratio_real"] < 1).mean()
    print(f"[06] share of loads where distance < straight line using real coords: {lt1:.1%}; "
          f"data coords: {(d['ratio_data'] < 1).mean():.1%}")
    axes[1].text(0.03, 0.83, f"{lt1:.0%} of loads would have a road\nshorter than the straight line",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=7.5, color=VERMIL)
    axes[0].set_ylabel("distance column (miles)")
    fig.suptitle(f"`distance` matches the data's coordinates (≈{med_data:.2f}× straight line, a normal "
                 "road factor),\nnot the real ones", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, "06_distance_vs_haversine.png")


def fig07_dow_holidays(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].copy()
    d["dow"] = d["date"].dt.dayofweek
    # ppm relative to route x equipment median removes lane mix
    d["ppm_rel"] = d["ppm"] / d.groupby(["pickup", "delivery", "equipment"])["ppm"].transform("median")
    dm = d.dropna(subset=["market_index"])
    b = np.polyfit(dm["market_index"], dm["ppm_rel"], 1)
    d["resid"] = d["ppm_rel"] - np.polyval(b, d["market_index"]) + 1
    dow = d.groupby("dow").agg(raw=("ppm_rel", "mean"), sem=("ppm_rel", "sem"),
                               res=("resid", "mean"), mi=("market_index", "mean"))
    dow_va = va.groupby(va["date"].dt.dayofweek)["market_index"].mean()
    print("[07] by weekday (raw rel ppm, residual after market_index, train mi, val mi):\n"
          + dow.assign(mi_val=dow_va).round(4).to_string())
    daily = d.groupby("date").agg(rel=("ppm_rel", "mean"), res=("resid", "mean"))
    holidays = {"Memorial Day": "2025-05-26", "July 4": "2025-07-04", "Labor Day": "2025-09-01"}
    for name, h in holidays.items():
        h = pd.Timestamp(h)
        win = daily.loc[h - pd.DateOffset(days=7): h + pd.DateOffset(days=7), "res"].drop(h)
        print(f"[07] {name} ({h:%a}): residual day-of {daily.loc[h, 'res']:.3f}, "
              f"+/-7d mean {win.mean():.3f} sd {win.std():.3f}")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    raw_sp = (dow["raw"].max() - dow["raw"].min()) * 100
    res_sp = (dow["res"].max() - dow["res"].min()) * 100

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    x = np.arange(7)
    ax.bar(x - 0.2, dow["raw"], 0.4, color=SKY, label=f"Raw (spread {raw_sp:.1f}%)")
    ax.bar(x + 0.2, dow["res"], 0.4, color=BLUE, label=f"After market_index ({res_sp:.1f}%)")
    ax.set_xticks(x, names)
    ax.set_ylim(0.98, 1.02)
    ax.axhline(1, color=BLACK, lw=0.6)
    ax.set_ylabel("$/mile ÷ route median")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.plot(x, dow["mi"], color=ORANGE, marker="o", ms=3, lw=1.2, label="market_index (train)")
    ax2.plot(x, dow_va.values, color=ORANGE, marker="o", ms=3, lw=1.2, ls="--",
             label="market_index (Nov–Dec)")
    ax2.set_ylim(0.6, 1.3)
    ax2.set_ylabel("Mean market_index", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, ncol=1)
    ax.set_title("By weekday", fontsize=9, fontweight="normal")
    ax = axes[1]
    sub = daily.loc["2025-05-01":"2025-09-30"]
    ax.plot(sub.index, sub["rel"], color=SKY, lw=0.8, label="Raw daily mean")
    ax.plot(sub.index, sub["res"], color=BLUE, lw=1.1, label="After market_index")
    for name, h in holidays.items():
        h = pd.Timestamp(h)
        ax.axvline(h, color=VERMIL, ls="--", lw=0.9)
        ax.text(h, 1.055, name, ha="center", fontsize=7, color=VERMIL,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.axhline(1, color=BLACK, lw=0.6)
    ax.set_ylim(0.95, 1.065)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("2025")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=2)
    ax.annotate("Step on Jul 1 (month start),\nnot Jul 4", xy=(pd.Timestamp("2025-07-01"), 1.0),
                xytext=(pd.Timestamp("2025-07-20"), 0.965), fontsize=7, color="#333333",
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_title("Daily, May–Sep 2025: no spike at holidays", fontsize=9, fontweight="normal")
    fig.suptitle("Prices rise midweek only because market_index does; with it accounted for, "
                 "there is no weekday or holiday effect", y=1.03, fontsize=9.5)
    fig.text(0.01, -0.1, "Each load's $/mile ÷ its route × equipment median (removes lane mix). "
             "'After market_index' removes a linear fit on market_index.\nOutliers removed. "
             "The remaining slow drift (month-level steps) is a separate effect, not weekday/holiday.",
             fontsize=7.5, color="#444444")
    fig.tight_layout()
    save(fig, "07_daily_ppm_dow.png")


def fig08_shift(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.8))
    axes = axes.ravel()
    specs = [("distance", "Distance (miles)", np.linspace(0, 3500, 50)),
             ("weight_fixed", "Weight (lb, sign fixed)", np.linspace(0, 48000, 49)),
             ("market_index", "market_index", np.linspace(0.75, 1.5, 61)),
             ("quote_signal", "quote_signal", 50)]
    for ax, (col, lab, bins) in zip(axes, specs):
        a, b = tr[col].dropna(), va[col].dropna()
        if isinstance(bins, int):
            lo, hi = np.nanpercentile(pd.concat([a, b]), [0.2, 99.8])
            bins = np.linspace(lo, hi, bins)
        ax.hist(a, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=0.35,
                label="Train Jan–Oct")
        ax.hist(b, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.5,
                label="Validation Nov–Dec")
        ax.set_xlabel(lab)
        ax.set_yticks([])
        print(f"[08] {col}: train mean {a.mean():.3f} sd {a.std():.3f} min {a.min():.3f} "
              f"max {a.max():.3f} | val mean {b.mean():.3f} sd {b.std():.3f} "
              f"min {b.min():.3f} max {b.max():.3f}")
    axes[2].set_title("Validation sits at the low end", fontsize=8.5, fontweight="normal",
                      color=VERMIL)
    axes[0].set_ylabel("Density")
    axes[3].set_ylabel("Density")
    ax = axes[4]
    st = tr["equipment"].value_counts(normalize=True).reindex(EQUIP_ORDER)
    sv = va["equipment"].value_counts(normalize=True).reindex(EQUIP_ORDER)
    x = np.arange(3)
    ax.bar(x - 0.2, st, 0.4, color=BLUE, alpha=0.35, label="Train Jan–Oct")
    ax.bar(x + 0.2, sv, 0.4, facecolor="none", edgecolor=ORANGE, lw=1.5, label="Validation Nov–Dec")
    ax.set_xticks(x, ["Dry Van", "Reefer", "Flatbed"])
    ax.set_ylabel("Share of loads")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    print(f"[08] equipment share train {st.round(3).to_dict()} val {sv.round(3).to_dict()}")
    ax = axes[5]
    lanes_tr = set(zip(tr.pickup, tr.delivery))
    seen_lane = np.array([l in lanes_tr for l in zip(va.pickup, va.delivery)])
    cities_tr = set(tr.pickup) | set(tr.delivery)
    new_city = ~(va.pickup.isin(cities_tr) & va.delivery.isin(cities_tr))
    cats = ["Seen\nlane", "New lane,\nknown cities", "New\ncity"]
    vals = [seen_lane.mean(), (~seen_lane & ~new_city).mean(), new_city.mean()]
    ax.bar(cats, vals, color=[BLUE, SKY, VERMIL])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}" if v < 0.01 else f"{v:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylabel("Share of validation loads")
    print(f"[08] validation rows: lane seen {vals[0]:.4f}, unseen lane w/ known cities {vals[1]:.4f}, "
          f"new city {vals[2]:.4f}")
    lpd_tr = tr.groupby("date").size().mean()
    lpd_va = va.groupby("date").size().mean()
    print(f"[08] loads/day train {lpd_tr:.0f} val {lpd_va:.0f}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Validation matches training except market_index (mean "
                 f"{va.market_index.mean():.2f} vs {tr.market_index.mean():.2f}),\n"
                 f"volume ({lpd_tr:.0f}→{lpd_va:.0f} loads/day) and {vals[2]:.0%} of loads "
                 "touching a city unseen in training", fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "08_train_vs_validation_shift.png")


def fig09_lex_fw(tr: pd.DataFrame, va: pd.DataFrame, dec: pd.DataFrame) -> None:
    lane = tr[((tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")) |
              ((tr.pickup == "Fort Wayne") & (tr.delivery == "Lexington"))].copy()
    lane["dir"] = np.where(lane.pickup == "Lexington", "Lexington→Fort Wayne", "Fort Wayne→Lexington")
    lane["rate360"] = lane["ppm"] * 360  # normalise to the December file's 360 mi
    dv = lane[lane.equipment == "Dry Van"]
    fwd = dv[dv.dir == "Lexington→Fort Wayne"]
    print(f"[09] lane rows: {len(lane)} total; Dry Van {len(dv)} "
          f"(L→FW {len(fwd)}, FW→L {len(dv) - len(fwd)})")
    print("[09] Dry Van L→FW posted_rate: " + fwd.posted_rate.describe().round(1).to_string()
          .replace("\n", "; "))
    print(f"[09] Dry Van L→FW distance range {fwd.distance.min()}–{fwd.distance.max()}, "
          f"weight median {fwd.weight_fixed.median():.0f}; quote_signal median "
          f"{fwd.quote_signal.median():.3f}")
    print(f"[09] outliers flagged on lane: {int(lane.outlier.sum())} "
          f"{lane.loc[lane.outlier, ['date', 'equipment', 'dir', 'ratio']].round(2).values.tolist()}")
    dvc = dv[~dv.outlier]
    med = dvc.rate360.median()
    q25, q75 = dvc.rate360.quantile([0.25, 0.75])
    dec_mi = va[va.date.dt.month == 12]["market_index"]
    mlo, mhi = dec_mi.quantile([0.05, 0.95])
    like = dvc[dvc.market_index.between(mlo, mhi)]
    print(f"[09] Dry Van both dirs rate@360mi: median {med:.0f}, IQR {q25:.0f}–{q75:.0f}; "
          f"with market_index in Dec 5–95% range [{mlo:.2f},{mhi:.2f}]: n={len(like)}, "
          f"median {like.rate360.median():.0f}, IQR {like.rate360.quantile(.25):.0f}–"
          f"{like.rate360.quantile(.75):.0f}")
    r_mi = dvc.rate360.corr(dvc.market_index)
    print(f"[09] lane Dry Van corr(rate360, market_index) = {r_mi:.2f}")

    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    ax.axhspan(q25, q75, color=BLUE, alpha=0.1, label=f"Dry Van IQR ${q25:.0f}–${q75:.0f}")
    ax.axhline(med, color=BLUE, lw=1, ls="--", label=f"Dry Van median ${med:.0f}")
    other = lane[(lane.equipment != "Dry Van") & ~lane.outlier]
    for e, mk in [("Reefer", "s"), ("Flatbed", "^")]:
        o = other[other.equipment == e]
        ax.scatter(o.date, o.rate360, s=14, marker=mk, color=EQUIP_COLORS[e], alpha=0.6,
                   label=f"{e}, context (n={len(o)})")
    for dname, fc in [("Lexington→Fort Wayne", BLUE), ("Fort Wayne→Lexington", "white")]:
        s = dvc[dvc.dir == dname]
        ax.scatter(s.date, s.rate360, s=26, facecolor=fc, edgecolor=BLUE, lw=1.2,
                   zorder=4, label=f"Dry Van {dname} (n={len(s)})")
    lo = min(lane[~lane.outlier].rate360.min(), q25) * 0.92
    hi = max(lane[~lane.outlier].rate360.max(), q75) * 1.06
    out = lane[lane.outlier]
    if len(out):
        ax.scatter(out.date, out.rate360.clip(lo * 1.01, hi * 0.99), s=22, marker="x",
                   color=VERMIL, label=f"Flagged outlier, clipped (n={len(out)})", zorder=5)
    ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31"), color=GREY, alpha=0.15)
    ax.text(pd.Timestamp("2025-12-16"), hi * 0.985,
            f"December\nwindow\n\nDry Van loads\nat Dec-like\nmarket_index:\n"
            f"median ${like.rate360.median():.0f}\n(n={len(like)})",
            ha="center", va="top", fontsize=7, color="#333333")
    ax.set_xlim(pd.Timestamp("2024-12-25"), pd.Timestamp("2026-01-05"))
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("2025")
    ax.set_ylabel("Rate scaled to 360 mi ($ per load)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    fig.suptitle(f"Lexington↔Fort Wayne Dry Van: about ${med:.0f} per 360-mi load; "
                 f"December should land near ${like.rate360.median():.0f}", fontsize=10, y=0.99)
    ax.set_title("Each load's $/mile × 360 to match the December file (360 mi, 32,000 lb, Dry Van). "
                 "Reefer/Flatbed for context only.", fontsize=7.5, fontweight="normal",
                 color="#444444")
    fig.tight_layout()
    save(fig, "09_lexington_fortwayne_history.png")


def main() -> None:
    setup_style()
    tr, va, dec = load()
    print(f"train {tr.shape}, validation {va.shape}, december {dec.shape}")
    fig01_outliers(tr)
    fig02_quote_signal(tr, va)
    fig03_monthly(tr, va)
    fig04_distance_equipment(tr)
    fig05_coordinates(tr, va)
    fig06_distance_haversine(tr)
    fig07_dow_holidays(tr, va)
    fig08_shift(tr, va)
    fig09_lex_fw(tr, va, dec)


if __name__ == "__main__":
    main()
