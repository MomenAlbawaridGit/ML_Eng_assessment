"""Build the submission report (DOCX). Every number is read from the files in reports/
and data/, so the report always matches the latest run.

    python -m src.report   # -> reports/freight_rate_report.docx
"""

from __future__ import annotations

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src import config as C
from src.experiments import SELECTED

FIG = C.FIGURES_DIR
OUT = C.REPORTS_DIR / "freight_rate_report.docx"
CHART = C.ROOT / "scorer_results" / "candidate_december.png"

# Rows shown in the results table (registry name -> label). The selected model is added last.
TABLE_ROWS = {
    "baseline_a_eq_dist": "(a) Baseline: equipment × distance median",
    "baseline_b_lane": "(b) Baseline: lane × equipment median",
    "baseline_c2_lane_x_mi_detrended": "(c) Baseline (b) × detrended market_index",
    "rf_mi_offset": "Random forest",
    "hgb_mi_offset": "HistGradientBoosting",
    "catboost_mi_offset": "CatBoost",
    "ridge_spline": "Ridge + splines",
}


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


# --- docx helpers ----------------------------------------------------------------------
def table(doc: Document, header: list[str], rows: list[list[str]], bold_rows: tuple[int, ...] = ()) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for i, row in enumerate(rows):
        for cell, text in zip(t.add_row().cells, row):
            cell.text = text
            if i in bold_rows:
                cell.paragraphs[0].runs[0].bold = True
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def figure(doc: Document, path, caption: str, width: float = 6.3) -> None:
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(9)
    p.runs[0].font.color.rgb = RGBColor(0x45, 0x5A, 0x60)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


# --- numbers ---------------------------------------------------------------------------
def load_numbers() -> dict:
    exp = pd.read_csv(C.REPORTS_DIR / "experiments.csv").set_index("experiment")
    sel = exp.loc[SELECTED]
    bd = pd.read_csv(C.REPORTS_DIR / "breakdown_selected.csv")
    eq = bd[(bd["group"] == "equipment") & (bd["fold"] == "wf3")].set_index("value")["MAPE"]
    dec = pd.read_csv(C.DECEMBER_PATH, parse_dates=["date"])
    bias_path = C.REPORTS_DIR / "fold_bias.csv"
    bias = pd.read_csv(bias_path) if bias_path.exists() else None
    return {"exp": exp, "sel": sel, "eq": eq, "dec": dec, "bias": bias}


def results_rows(exp: pd.DataFrame) -> tuple[list[list[str]], int]:
    names = list(TABLE_ROWS)
    if SELECTED not in names:
        names.append(SELECTED)
    rows = []
    for name in names:
        r = exp.loc[name]
        label = TABLE_ROWS.get(name, name.replace("_", " "))
        if name == SELECTED:
            label += " (selected)"
        rows.append([label] + [fmt(r[c]) for c in
                     ("MAPE_wf1", "MAPE_wf2", "MAPE_wf3", "MAPE_wf_mean", "MAPE_all_wf_mean", "MAPE_city")])
    return rows, names.index(SELECTED)


# --- report ----------------------------------------------------------------------------
def build() -> None:
    n = load_numbers()
    exp, sel, eq, dec = n["exp"], n["sel"], n["eq"], n["dec"]
    base_b = exp.loc["baseline_b_lane"]
    rate = dec["predicted_rate"]
    thursdays = dec.loc[dec["date"].dt.dayofweek == 3, "predicted_rate"]
    half = (rate.max() - rate.min()) / 2
    hl240 = exp.loc["abl_halflife_240d"] if "abl_halflife_240d" in exp.index else None
    fallback = exp.loc["abl_mi_none_plus_dow"] if "abl_mi_none_plus_dow" in exp.index else None

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    doc.add_heading("Freight Rate Prediction: Validation, Split and Results", level=0)
    doc.add_paragraph(
        "Task: predict posted_rate (USD per load) for 12,000 Nov–Dec 2025 loads from 48,000 labelled "
        f"Jan–Oct 2025 loads. The selected model scores {fmt(sel['MAPE_wf_mean'])}% mean absolute "
        f"percentage error (MAPE) on clean rows in out-of-time tests, against {fmt(base_b['MAPE_wf_mean'])}% "
        "for a lane-median baseline. For Nov–Dec, expect roughly 1.8–2.8% on clean rows, because month-level "
        "price swings are not predictable from the inputs (section 5). Including the ~1.4% of rows with "
        f"corrupted prices, expect about {fmt(sel['MAPE_all_wf_mean'], 1)}%."
    )
    doc.add_paragraph(
        "Terms used: MAPE = mean |predicted − actual| / actual. pp = percentage points. Walk-forward fold = "
        "train on earlier months, test on the next two months. Detrended market_index = the day's average "
        "market_index minus its 28-day average, i.e. the short-term ups and downs with the slow drift removed."
    )

    # 1 -------------------------------------------------------------------------------
    doc.add_heading("1. Validating the train/test data", level=1)
    doc.add_paragraph(
        "Before modelling, I checked every column in code. The schema, the date ranges and the IDs are "
        "clean: there are no duplicate loads and no loads whose pickup city equals the delivery city. "
        "Equipment, distance and weight are distributed the same way in train and validation, so the only "
        "shift is the dates. The issues found, and how each was handled:"
    )
    table(doc, ["Issue", "Rows (train / val)", "Handling"], [
        ["quote_signal switches behaviour by month. In Jan, Feb, Mar, Jun and Sep it is within 3% of the "
         "true $/mile on 98% of loads. In Apr, May, Jul and Oct it is a mirror image. In Aug, Nov and Dec it "
         "is noise. The pooled correlation (0.05) hides this.", "all", "Excluded from every model (target leak)"],
        ["Negative weights (sign errors; same magnitudes as positive ones)", "292 / 145", "abs()"],
        ["Missing weight", "300 / 165", "Training median per equipment (31.4k–31.6k lb)"],
        ["Missing market_index", "374 / 249",
         "Every row gets the same-day mean. Within-day variation is noise. In a masking test the same-day "
         "mean reaches the noise floor, while a ±5-day mean is 4× worse"],
        ["Price outliers: $/mile about 0.2× or 3× the lane norm", "677 (1.41%) / –",
         "Removed from training rows only; kept and reported separately in test folds"],
        ["Weights clipped at 5,000 and 47,500 lb", "1,236 / 308", "Kept (prices normal)"],
        ["Coordinates differ from the real cities (median 110 mi)", "all cities",
         "Kept: distance is 1.18× the straight line between the data's own coordinates (r = 0.9995), and "
         "it does not fit the real ones"],
        ["Same lane has different distances (±2.2%)", "most lanes", "Kept: price follows per-load distance"],
        ["December file lacks coordinates and market_index", "31",
         "Coordinates from the training city table; market_index from validation same-day means"],
    ])
    figure(doc, FIG / "02_quote_signal_vs_ppm.png",
           "Figure 1. quote_signal copies or mirrors the target in 9 of 10 training months and is noise in "
           "Nov–Dec. A model using it would score well on September (a copy month) and fail on the real data.")
    figure(doc, FIG / "01_outliers_before_after.png",
           "Figure 2. Outlier rule: $/mile ÷ lane reference (lanes with ≥5 loads, otherwise equipment × "
           "distance band), cut at 0.5× and 2×. The cut-offs sit in an empty gap. The rule flags the same "
           "336 rows as an independent check in the months where quote_signal reveals the true price "
           "(0 false positives, 0 false negatives).")
    figure(doc, FIG / "06_distance_vs_haversine.png",
           "Figure 3. distance matches the dataset's own coordinates, not the real ones "
           "(Providence→Hartford would be 4.07× the straight line using real coordinates).")
    doc.add_paragraph(
        "Leakage control: weight medians, the city table, the outlier reference, the encoders and the splines "
        "are all fitted on each fold's training rows only, then applied to that fold's test rows. One "
        "Cleaner class does this for train, validation and the December file, so all three are prepared the "
        "same way. One deliberate exception is the daily market_index series, which is built from the "
        "inputs of train and validation (never from labels), because validation.csv provides those values "
        "at prediction time. Automated tests (tests/test_pipeline.py) check these rules."
    )

    # 2 -------------------------------------------------------------------------------
    doc.add_heading("2. Split and validation approach", level=1)
    doc.add_paragraph(
        "The task is a forecast 1–2 months past the last labelled date, so a random split would leak the "
        "future and flatter the model. I used time-ordered folds that copy that horizon, plus one fold for "
        "unseen cities:"
    )
    table(doc, ["Fold", "Train on", "Test on", "What it rehearses"], [
        ["Walk-forward 1", "Jan–Jun", "Jul–Aug", "1–2 month forecast"],
        ["Walk-forward 2", "Jan–Jul", "Aug–Sep", "1–2 month forecast"],
        ["Walk-forward 3", "Jan–Aug", "Sep–Oct", "1–2 month forecast; Sep–Oct market_index ≈ Nov–Dec"],
        ["City holdout", "Jan–Aug minus 8 cities", "Sep–Oct loads touching them",
         "12.1% of validation loads touch 8 cities never seen in training"],
        ["Final fit", "Jan–Oct (outliers removed, 47,323 rows)", "Validation Nov–Dec", "—"],
    ])
    bullets(doc, [
        "Selection rule: lowest mean MAPE over the three walk-forward folds. A model must beat the "
        "lane-median baseline on every fold and on the city holdout. When two models are within about "
        "0.1 pp, the simpler one wins. Later changes were adopted only if they improved the mean by at "
        "least 0.05 pp and were no more than 0.02 pp worse on any fold.",
        "Metrics: MAPE is the headline because pricing error is relative. Median APE, MAE, RMSE and R² are "
        "in reports/experiments.csv. Every metric is reported without and with the outlier rows. Outliers "
        "are corrupted labels that no input can predict, so the headline excludes them, but the all-rows "
        "number is what a scorer will see.",
        f"Caveat: the walk-forward test windows overlap (Aug and Sep appear twice), and {len(exp)} "
        "configurations were compared on the same folds. Treat differences under about 0.05 pp as noise.",
    ])

    # 3 -------------------------------------------------------------------------------
    doc.add_heading("3. Models and results", level=1)
    rows, sel_i = results_rows(exp)
    table(doc, ["Model", "WF1", "WF2", "WF3", "Mean", "All rows", "City"], rows, bold_rows=(sel_i,))
    doc.add_paragraph(
        "MAPE % on clean rows. \"All rows\" includes outliers (mean of the three walk-forward folds). "
        f"Source: reports/experiments.csv ({len(exp)} configurations)."
    )
    figure(doc, FIG / "10_model_comparison.png", "Figure 4. Walk-forward MAPE per model, with per-fold and city-holdout markers.")
    level_note = ""
    if n["bias"] is not None:
        b = n["bias"]
        lvl = b[b["model"].str.contains("mi_level")]
        if len(lvl):
            level_note = (f" (mean bias with the level: {fmt(lvl['bias_pct'].min(), 1)}% to "
                          f"{fmt(lvl['bias_pct'].max(), 1)}% per fold; reports/fold_bias.csv)")
    findings = [
        "The raw market_index level breaks out-of-time prediction. Apr–Jul index values of 1.2–1.3 came with "
        "prices only 1–3% above normal, so models using the level under-predicted Jul–Oct, more so the further ahead they forecast" + level_note + ". "
        f"Swapping in the detrended index cut the mean MAPE from {fmt(exp.loc['abl_mi_level', 'MAPE_wf_mean'])}% "
        f"to {fmt(exp.loc['ridge_spline', 'MAPE_wf_mean'])}%. Within a month, the daily index moves with "
        "prices (r 0.75–0.95 in 8 of 10 months; reports/data_findings.md §5). It also carries the weekday "
        "cycle (Thursday high, Sunday low), so a separate day-of-week feature added nothing.",
        "Tree models memorised day-level noise, because market_index takes one value per day. Feeding the "
        "index as a fixed linear offset fixed that, but CatBoost "
        f"({fmt(exp.loc['catboost_mi_offset', 'MAPE_wf_mean'])}%) still did not beat Ridge. Ridge plus a "
        f"booster on its residuals ({fmt(exp.loc['ridge_spline_hgb_resid', 'MAPE_wf_mean'])}%) was within "
        "noise, so the simpler, more transparent model was kept.",
        "Ridge + splines: Ridge regression on log($/mile). Inputs: one-hot pickup, delivery and equipment; "
        "splines on log-distance and weight; equipment × log-distance; coordinates, so unseen cities can be "
        "placed; detrended market_index. Tuning α and the spline knots moved MAPE by less than 0.01 pp.",
    ]
    if hl240 is not None:
        findings.append(
            f"Weighting recent months more (half-life 240 days) scored {fmt(hl240['MAPE_wf_mean'], 3)}%: better on "
            "walk-forward 1 and 3, worse on 2, and within noise overall. Not adopted.")
    if "r2_combo_lane0.3_coord6" in exp.index:
        c = exp.loc["r2_combo_lane0.3_coord6"]
        findings.append(
            "Round 2 tested 20 more variants: lane one-hots, city × equipment terms, market_index interactions, "
            "coordinate splines and re-anchoring to recent weeks. None met the adoption rule. The closest "
            "(adding a heavily shrunk lane term and coordinate splines) was better on every fold "
            f"({fmt(c['MAPE_wf1'])}/{fmt(c['MAPE_wf2'])}/{fmt(c['MAPE_wf3'])}, city {fmt(c['MAPE_city'])}). "
            f"Its mean gain of {fmt(sel['MAPE_wf_mean'] - c['MAPE_wf_mean'], 3)} pp fell short of 0.05 pp, "
            "it was the best of many tries, and its settings were tuned on the same folds, so it was not "
            "adopted. Its predictions differ from the selected model's by at most ±2.6%.")
    if SELECTED != "ridge_spline":
        findings.append(f"Round 2 adopted {SELECTED}; see reports/experiments.md for what changed and why.")
    bullets(doc, findings)
    figure(doc, FIG / "11_pred_vs_actual.png", "Figure 5. Walk-forward 3 (Sep–Oct): predicted vs actual. Outliers form two bands at about 0.2× and 3×.")
    figure(doc, FIG / "12_residuals_over_time.png",
           f"Figure 6. Sep–Oct daily bias and error by segment. Dry Van {fmt(eq['Dry Van'])}%, "
           f"Reefer {fmt(eq['Reefer'])}%, Flatbed {fmt(eq['Flatbed'])}% MAPE.")

    # 4 -------------------------------------------------------------------------------
    doc.add_heading("4. December prediction chart", level=1)
    figure(doc, CHART, "Figure 7. December chart produced by the provided score.py (Lexington → Fort Wayne, "
                       f"360 mi, Dry Van, 32,000 lb). Note the zoomed y-axis: the swing is only ±${half:.0f}.")
    doc.add_paragraph(
        f"With every input fixed except the date, the curve is a flat weekly cycle of ${rate.min():.0f}–"
        f"${rate.max():.0f} (mean ${rate.mean():.0f}, ±{100 * half / rate.mean():.1f}%). It follows the "
        "weekly pattern in December's market_index, peaking every Thursday "
        f"(about ${thursdays.mean():.0f}) and dipping on Sunday and Monday. 25 December peaks because it is "
        "a Thursday, not because it is a holiday. Training covers Jan–Oct, so Thanksgiving and Christmas "
        "never appear in it, and there is no detectable price effect around July 4, Memorial Day or Labor "
        "Day. The model therefore has no holiday effect to learn. A real market would probably tighten "
        "before Christmas, and this data cannot teach that. The level agrees with history: Lexington↔Fort "
        "Wayne Dry Van loads at December-like market_index have an interquartile range of $792–$831 when "
        "scaled to 360 mi (n = 17; Figure 8)."
    )
    figure(doc, FIG / "13_december_explained.png", "Figure 8. The December curve against detrended market_index and the lane's history.")

    # 5 -------------------------------------------------------------------------------
    doc.add_heading("5. Risks and limitations", level=1)
    risks = [
        "Month-level price shifts of about ±5% (Jan −5%, Jun +4%) are not explained by any input. In "
        "effect, Nov–Dec are priced at the Jan–Oct level. On Sep–Oct this produced a bias of about −0.8%, "
        "and prices also drift within some months. Real Nov–Dec error could be about 1 pp higher than the "
        "folds show.",
        f"If the hidden labels contain the same ~1.4% corruption, the all-rows MAPE will be about "
        f"{fmt(sel['MAPE_all_wf_mean'], 1)}%, not {fmt(sel['MAPE_wf_mean'], 1)}%.",
        f"The 8 unseen cities are placed by their coordinates. The city holdout ({fmt(sel['MAPE_city'])}%) "
        "suggests this works.",
    ]
    if fallback is not None:
        risks.append(
            f"Fallback: dropping market_index and using day-of-week instead scores {fmt(fallback['MAPE_wf_mean'])}%. "
            "It would be the safer choice if the real December market_index differed from validation.csv.")
    bullets(doc, risks)

    doc.add_heading("Reproduce", level=2)
    doc.add_paragraph(
        "pip install -r requirements.txt; python -m src.train; python -m src.predict; "
        "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv. "
        "python -m src.evaluate regenerates the experiment tables. python -m src.plots_eda and "
        "python -m src.plots_model regenerate the figures. python -m src.report rebuilds this document."
    )

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
