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
TITLE_PT, SUB_PT, LABEL_PT, TICK_PT, NOTE_PT = 14, 11, 11.5, 10.5, 10.5
SUB_COLOR = "#444444"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": NOTE_PT, "axes.titlesize": TITLE_PT, "axes.titleweight": "bold",
        "axes.labelsize": LABEL_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": NOTE_PT, "legend.frameon": False,
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


def titles(ax: plt.Axes, title: str, sub: str | None = None) -> None:
    """Left-aligned bold title with an optional one-line plain subtitle beneath it."""
    ax.set_title(title, loc="left", pad=26 if sub else 10, fontsize=TITLE_PT,
                 fontweight="bold")
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=SUB_PT, color=SUB_COLOR)


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
DIST_LABELS = ["Under\n250", "250 to\n500", "500 to\n1,000", "1,000 to\n1,500",
               "1,500 to\n2,000", "Over\n2,000"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
money = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")


# --------------------------------------------------------------------------- figures
def fig01_outliers(tr: pd.DataFrame) -> None:
    n_out = int(tr["outlier"].sum())
    lo_n = int((tr["ratio"] < RATIO_LO).sum())
    hi_n = int((tr["ratio"] > RATIO_HI).sum())
    print(f"[01] flagged {n_out} rows ({n_out / len(tr):.2%}): {lo_n} below {RATIO_LO}, "
          f"{hi_n} above {RATIO_HI}")
    bins = np.logspace(np.log10(0.12), np.log10(8), 110)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    ax.hist(tr.loc[~tr.outlier, "ratio"], bins=bins, color=BLUE, edgecolor="none")
    ax.hist(tr.loc[tr.outlier, "ratio"], bins=bins, color=VERMIL, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks([0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.2×", "0.5×", "1×", "2×", "5×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.12, 8)
    ax.set_ylim(0.8, 2e4)
    ax.set_xlabel("Price ÷ normal price for that route (log scale)")
    ax.set_ylabel("Number of loads (log scale)")
    ax.text(0.28, 60, f"{lo_n} priced\nfar too low", color=VERMIL, ha="center", fontsize=11,
            fontweight="bold")
    ax.text(3.4, 60, f"{hi_n} priced\nfar too high", color=VERMIL, ha="center", fontsize=11,
            fontweight="bold")
    ax.text(1.0, 1.25e4, f"{len(tr) - n_out:,} normal loads", color=BLUE, ha="center",
            fontsize=11, fontweight="bold")
    ax.text(0.5, -0.30, f"{n_out} loads ({n_out / len(tr):.1%}) with wrong prices removed",
            transform=ax.transAxes, ha="center", fontsize=11.5, color=VERMIL)
    ax.grid(axis="x", visible=False)
    titles(ax, "Some prices in the data are clearly wrong",
           "Normal loads sit close to the usual price for their route")
    fig.tight_layout()
    save(fig, "01_outliers_before_after.png")


def fig02_quote_signal(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    """Short-minus-long quote_signal by month: >0 copies price, <0 mirrors it, ~0 no link.

    Validation has no prices, so a direct correlation is impossible for Nov and Dec; this
    measure works without prices because real price per mile is higher on short trips.
    """
    def signature(df: pd.DataFrame) -> pd.Series:
        g = df.groupby(df["date"].dt.to_period("M"))
        return g.apply(lambda s: s.loc[s.distance < 300, "quote_signal"].mean()
                       - s.loc[s.distance > 2000, "quote_signal"].mean())

    vals = list(signature(tr).values) + list(signature(va).values)
    print("[02] short-minus-long quote_signal: "
          + ", ".join(f"{m} {v:+.2f}" for m, v in zip(MONTHS, vals)))
    d = tr.loc[~tr.outlier]
    r_m = d.groupby(d.date.dt.month).apply(lambda s: s.quote_signal.corr(s.ppm))
    print("[02] corr(quote_signal, ppm) by month: " + ", ".join(f"{m}:{v:+.2f}" for m, v in r_m.items()))

    def kind(v):
        return "copy" if v > 0.3 else "mirror" if v < -0.3 else "none"
    col = {"copy": BLUE, "mirror": ORANGE, "none": GREY}
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    ax.bar(x, vals, color=[col[kind(v)] for v in vals], width=0.7)
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.axvline(9.5, color=BLACK, lw=0.8, ls=":")
    ax.text(10.5, 1.0, "To\npredict", ha="center", va="top", fontsize=10.5, color=SUB_COLOR)
    ax.text(1, 0.86, "copies price", ha="center", color=BLUE, fontsize=11, fontweight="bold")
    ax.text(4, -0.98, "mirrors price", ha="center", color="#a66f00", fontsize=11,
            fontweight="bold")
    ax.text(7, 0.08, "no link", ha="center", color="#666666", fontsize=11, fontweight="bold")
    ax.text(10.5, 0.08, "no link", ha="center", color="#666666", fontsize=11, fontweight="bold")
    ax.set_xticks(x, MONTHS)
    ax.set_ylim(-1.1, 1.15)
    ax.set_ylabel("quote_signal, short trips\nminus long trips")
    ax.grid(axis="x", visible=False)
    titles(ax, "quote_signal leaks the price, but not in Nov and Dec",
           "Short trips cost more per mile, so a copy of the price is higher on short trips")
    fig.tight_layout()
    save(fig, "02_quote_signal_vs_ppm.png")


def fig03_monthly(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].set_index("date")
    mo = d.resample("MS")["ppm"].mean()
    mi = d.resample("MS")["market_index"].mean()
    print("[03] monthly ppm: " + ", ".join(f"{k:%b} {v:.2f}" for k, v in mo.items()))
    print("[03] monthly market_index: " + ", ".join(f"{k:%b} {v:.3f}" for k, v in mi.items()))
    x = np.arange(len(mo))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    ax2 = ax.twinx()
    ax2.plot(x, mi.values, color=ORANGE, lw=1.6, alpha=0.55)
    ax2.set_ylabel("market_index", color="#a66f00")
    ax2.tick_params(axis="y", colors="#a66f00")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax2.set_ylim(0.6, 1.45)
    ax2.text(x[3], mi.iloc[3] + 0.05, "market_index", color="#a66f00", fontsize=10.5,
             ha="right")
    ax.plot(x, mo.values, color=BLUE, lw=2.6, marker="o", ms=6, zorder=5)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.text(2.6, 2.14, "Price per mile", color=BLUE, fontsize=11, fontweight="bold",
            ha="center", va="top")
    pk = int(np.argmax(mo.values))
    ax.annotate(f"${mo.iloc[pk]:.2f}", (x[pk], mo.iloc[pk]), xytext=(0, 9),
                textcoords="offset points", ha="center", color=BLUE, fontsize=10.5)
    ax.annotate(f"${mo.iloc[0]:.2f}", (x[0], mo.iloc[0]), xytext=(10, -10),
                textcoords="offset points", ha="left", color=BLUE, fontsize=10.5)
    ax.set_xticks(x, [f"{k:%b}" for k in mo.index])
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_ylim(1.98, 2.38)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.set_ylabel("Price per mile ($)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Prices rose until June, then eased",
           "Monthly average price per mile, Jan to Oct 2025")
    fig.tight_layout()
    save(fig, "03_monthly_ppm_market_index.png")


def fig04_distance_equipment(tr: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].copy()
    d["dbin"] = pd.cut(d["distance"], DIST_BINS, labels=DIST_LABELS, right=False)
    g = d.groupby(["dbin", "equipment"], observed=True)["ppm"].median().unstack()
    print("[04] median ppm by distance band x equipment:\n" + g.round(2).to_string())
    x = np.arange(len(DIST_LABELS))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    for e in ["Reefer", "Flatbed", "Dry Van"]:
        ax.plot(x, g[e].values, color=EQUIP_COLORS[e], lw=2.4, marker="o", ms=5)
        ax.text(x[-1] + 0.15, g[e].iloc[-1], e, color=EQUIP_COLORS[e], fontsize=11,
                fontweight="bold", va="center")
    ax.set_xticks(x, DIST_LABELS)
    ax.set_xlim(-0.3, len(x) - 0.1)
    ax.set_xlabel("Trip distance (miles)")
    ax.set_ylabel("Price per mile ($)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.grid(axis="x", visible=False)
    titles(ax, "Short trips cost more per mile",
           "Typical price per mile by trip length and truck type")
    fig.tight_layout()
    save(fig, "04_ppm_by_distance_equipment.png")


def fig05_coordinates(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    ct = city_table(pd.concat([tr, va]))
    known = ct.index.intersection(list(REAL_COORDS))
    ct = ct.loc[known].join(pd.DataFrame(REAL_COORDS, index=["rlat", "rlon"]).T)
    ct["err_mi"] = haversine_mi(ct["lat"], ct["lon"], ct["rlat"], ct["rlon"])
    med = ct["err_mi"].median()
    print(f"[05] coordinate error over {len(ct)} cities: median {med:.0f} mi, "
          f"mean {ct['err_mi'].mean():.0f}, max {ct['err_mi'].max():.0f}")
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    ax.scatter(ct["rlon"], ct["rlat"], s=12, color="#cccccc", zorder=2)
    offs = {"Los Angeles": (8, -4, "left"), "Tampa": (-10, -4, "right"),
            "Providence": (-10, -4, "right")}
    for c in REPORT_CITIES:
        r = ct.loc[c]
        print(f"[05] {c}: {r.err_mi:.0f} mi off")
        ax.annotate("", xy=(r.lon, r.lat), xytext=(r.rlon, r.rlat),
                    arrowprops=dict(arrowstyle="-|>", color=VERMIL, lw=1.8,
                                    shrinkA=5, shrinkB=5), zorder=4)
        ax.scatter([r.rlon], [r.rlat], s=55, facecolor="white", edgecolor=BLACK, lw=1.5,
                   zorder=5)
        ax.scatter([r.lon], [r.lat], s=55, color=VERMIL, zorder=5)
        dx, dy, ha = offs[c]
        ax.annotate(f"{c}\n{r.err_mi:.0f} miles off", (r.lon, r.lat), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.5, color=VERMIL,
                    fontweight="bold", ha=ha, va="top")
    la = ct.loc["Los Angeles"]
    ax.annotate("real", (la.rlon, la.rlat), xytext=(9, 0), textcoords="offset points",
                fontsize=10, va="center", color=BLACK)
    ax.annotate("in data", (la.lon, la.lat), xytext=(9, 6), textcoords="offset points",
                fontsize=10, va="center", color=VERMIL)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_aspect(1.25)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 46)
    titles(ax, f"City locations in the data are off by about {med:.0f} miles",
           f"Typical gap across {len(ct)} cities. Grey dots: real city locations.")
    fig.tight_layout()
    save(fig, "05_coordinates_check.png")


def fig06_distance_haversine(tr: pd.DataFrame) -> None:
    d = tr.copy()
    d["straight"] = haversine_mi(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    ratio = (d["distance"] / d["straight"]).median()
    print(f"[06] distance / straight line (data coords): median {ratio:.3f}, "
          f"corr {d['distance'].corr(d['straight']):.4f}")
    samp = d.sample(8000, random_state=0)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    ax.scatter(samp["straight"], samp["distance"], s=4, alpha=0.3, color=BLUE, lw=0,
               rasterized=True)
    xmax = d["straight"].max() * 1.03
    xx = np.array([0, xmax])
    ax.plot(xx, ratio * xx, color=BLACK, lw=1.2, ls="--")
    ax.annotate(f"distance = {ratio:.2f} × straight line", (1000, ratio * 1000),
                xytext=(150, 2900), fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Straight-line distance from the data's locations (miles)")
    ax.set_ylabel("Distance column (miles)")
    titles(ax, "The distance column matches the data's own locations",
           f"Roads run about {ratio:.2f} times the straight line, a normal ratio")
    fig.tight_layout()
    save(fig, "06_distance_vs_haversine.png")


def fig07_dow_holidays(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    d = tr.loc[~tr["outlier"]].copy()
    d["dow"] = d["date"].dt.dayofweek
    # price relative to route x equipment median removes lane mix
    d["ppm_rel"] = d["ppm"] / d.groupby(["pickup", "delivery", "equipment"])["ppm"].transform("median")
    dm = d.dropna(subset=["market_index"])
    b = np.polyfit(dm["market_index"], dm["ppm_rel"], 1)
    d["resid"] = d["ppm_rel"] - np.polyval(b, d["market_index"]) + 1
    dow = d.groupby("dow").agg(raw=("ppm_rel", "mean"), res=("resid", "mean"))
    raw = (dow["raw"] / dow["raw"].mean() - 1) * 100
    res = (dow["res"] / dow["res"].mean() - 1) * 100
    print("[07] weekday price vs average, %: raw " + ", ".join(f"{v:+.2f}" for v in raw)
          + " | after market_index " + ", ".join(f"{v:+.2f}" for v in res))
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    ax.bar(x - 0.2, raw, 0.4, color=SKY)
    ax.bar(x + 0.2, res, 0.4, color=BLUE)
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.annotate("Raw price", (3 - 0.2, raw.iloc[3]), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom", color="#2a7fb0",
                fontsize=11, fontweight="bold")
    ax.annotate("After market_index", (6 + 0.2, res.iloc[6]), xytext=(5.6, 0.6),
                ha="center", va="bottom", color=BLUE, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylim(-1.2, 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.set_ylabel("Price vs weekly average (%)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Midweek prices are higher only because market_index is",
           "Once market_index is accounted for, every weekday costs the same")
    fig.tight_layout()
    save(fig, "07_daily_ppm_dow.png")


def fig08_shift(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    for col in ["distance", "weight_fixed", "market_index"]:
        print(f"[08] {col}: train mean {tr[col].mean():.3f} | val mean {va[col].mean():.3f}")
    cities_tr = set(tr.pickup) | set(tr.delivery)
    new_city = (~(va.pickup.isin(cities_tr) & va.delivery.isin(cities_tr))).mean()
    print(f"[08] validation loads touching a new city: {new_city:.4f}")
    bins = np.linspace(0.65, 1.5, 52)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    a, b = tr["market_index"].dropna(), va["market_index"].dropna()
    ax.hist(a, bins=bins, weights=np.full(len(a), 100 / len(a)), color=BLUE, alpha=0.35,
            histtype="stepfilled")
    ax.hist(b, bins=bins, weights=np.full(len(b), 100 / len(b)), color=ORANGE, lw=2,
            histtype="step")
    ax.text(1.27, 3.7, "Jan to Oct\n(training)", color=BLUE, fontsize=11, fontweight="bold",
            ha="left")
    ax.text(1.07, 6.3, "Nov and Dec\n(to predict)", color="#a66f00", fontsize=11,
            fontweight="bold", ha="left")
    ax.set_xlabel("market_index")
    ax.set_ylabel("Share of loads (%)")
    ax.grid(axis="x", visible=False)
    titles(ax, "Nov and Dec market_index sits at the low end",
           f"Distance, weight and truck mix do not change; {new_city:.0%} of loads touch a new city")
    fig.tight_layout()
    save(fig, "08_train_vs_validation_shift.png")


def fig09_lex_fw(tr: pd.DataFrame, va: pd.DataFrame, dec: pd.DataFrame) -> None:
    lane = tr[(((tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")) |
               ((tr.pickup == "Fort Wayne") & (tr.delivery == "Lexington")))
              & (tr.equipment == "Dry Van")].copy()
    lane["rate360"] = lane["ppm"] * 360  # scale to the December file's 360 mi
    dvc = lane[~lane.outlier]
    med = dvc.rate360.median()
    print(f"[09] Dry Van both directions: {len(lane)} loads, {int(lane.outlier.sum())} wrong "
          f"price removed; rate@360mi median {med:.0f}, range {dvc.rate360.min():.0f} to "
          f"{dvc.rate360.max():.0f}")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    ax.scatter(dvc.date, dvc.rate360, s=30, color=BLUE, alpha=0.8, zorder=4)
    ax.axhline(med, color=BLACK, lw=1, ls="--")
    ax.text(pd.Timestamp("2025-12-16"), med + 4, f"Typical\n${med:.0f}", fontsize=11,
            ha="center", va="bottom", fontweight="bold")
    ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31"), color=GREY, alpha=0.15)
    ax.text(pd.Timestamp("2025-12-16"), dvc.rate360.max(), "December\n(to predict)",
            ha="center", va="top", fontsize=10.5, color=SUB_COLOR)
    ax.set_xlim(pd.Timestamp("2024-12-28"), pd.Timestamp("2026-01-03"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.set_major_formatter(money)
    ax.set_ylabel("Price for 360 miles ($)")
    ax.set_xlabel("2025")
    titles(ax, f"Lexington and Fort Wayne Dry Van loads cost about ${round(med, -1):.0f}",
           f"All {len(dvc)} past loads, both directions, scaled to the 360 mile December trip")
    ax.title.set_fontsize(13)
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
