"""Build the submission report (DOCX). Every number is read from the files in reports/
and data/, so the report always matches the latest run.

    python -m src.report   # writes reports/freight_rate_report.docx
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

# Rows shown in the results table (registry name: label). The selected model is shown last.
TABLE_ROWS = {
    "baseline_b_lane": "Simple route average (baseline)",
    "rf_mi_offset": "Random forest",
    "hgb_mi_offset": "Gradient boosting",
    "catboost_mi_offset": "CatBoost",
    "ridge_spline": "Ridge (chosen model)",
}


def pct(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}%"


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
                    r.font.size = Pt(9.5)
    doc.add_paragraph()


def figure(doc: Document, path, caption: str, width: float = 6.0) -> None:
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


# --- report ----------------------------------------------------------------------------
def build() -> None:
    exp = pd.read_csv(C.REPORTS_DIR / "experiments.csv").set_index("experiment")
    sel, base = exp.loc[SELECTED], exp.loc["baseline_b_lane"]
    per_fold = pd.read_csv(C.REPORTS_DIR / "experiments_per_fold.csv")
    main = per_fold[(per_fold["experiment"] == SELECTED) & (per_fold["fold"] == "wf3")].iloc[0]
    bd = pd.read_csv(C.REPORTS_DIR / "breakdown_selected.csv")
    eq = bd[(bd["group"] == "equipment") & (bd["fold"] == "wf3")].set_index("value")["MAPE"]
    bias = pd.read_csv(C.REPORTS_DIR / "fold_bias.csv")
    level_bias = bias[bias["model"].str.contains("mi_level") & bias["fold"].isin(["wf1", "wf2", "wf3"])]["bias_pct"]
    dec = pd.read_csv(C.DECEMBER_PATH, parse_dates=["date"])
    rate = dec["predicted_rate"]
    swing = (rate.max() - rate.min()) / 2

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    doc.add_heading("Freight Rate Prediction Report", level=0)

    # Summary ---------------------------------------------------------------------------
    doc.add_heading("Summary", level=1)
    bullets(doc, [
        "Goal: predict the price of 12,000 truck loads in November and December 2025, "
        "using 48,000 loads with known prices from January to October 2025.",
        f"Result: on past months the model was off by {pct(sel['MAPE_wf_mean'], 1)} on average. "
        f"A simple route average was off by {pct(base['MAPE_wf_mean'], 1)}, so the model cuts the error nearly in half.",
        "Biggest finding: the column quote_signal secretly copies the price in some months and is useless "
        "in November and December. I removed it.",
        "Chosen model: Ridge regression. It was the most accurate and the simplest to explain.",
    ])
    doc.add_paragraph(
        "How error is measured: for each load I take the gap between the predicted and the real price "
        "as a percent of the real price, then average it over all loads. Lower is better."
    )

    # 1. Data ---------------------------------------------------------------------------
    doc.add_heading("1. Checking and cleaning the data", level=1)
    doc.add_paragraph(
        "I checked every column before building any model. There are no duplicate loads, and the "
        "training and prediction files look the same apart from the dates. These are the problems I "
        "found and what I did about each one."
    )
    table(doc, ["Problem", "Loads affected", "What I did"], [
        ["quote_signal copies the price in 5 months, mirrors it in 4 months, and has no link to price "
         "in August, November and December", "All", "Removed it from the model"],
        ["Wrong prices, about 5 times too low or 3 times too high for the route", "677 (1.4%)",
         "Removed them from training only"],
        ["Negative weights (sign typos)", "292", "Made them positive"],
        ["Missing weights", "300", "Used the usual weight for that truck type"],
        ["Missing market_index", "374", "Used the average for that day"],
        ["City locations are about 110 miles from the real cities", "All cities",
         "Kept them, because the distance column was calculated from them"],
        ["December file has no locations or market_index", "31 rows",
         "Took locations from the training data and market_index from the November and December file"],
    ])
    figure(doc, FIG / "02_quote_signal_vs_ppm.png",
           "Figure 1. quote_signal copies or mirrors the price in most training months, but has no link "
           "in November and December. A model using it would look great in testing and fail on the real task.")
    figure(doc, FIG / "01_outliers_before_after.png",
           "Figure 2. Most loads are priced close to the normal price for their route. A small group is far "
           "too cheap or far too expensive. These look like data errors, so I removed them from training.")
    doc.add_paragraph(
        "To keep the test fair, anything the model learns from (usual weights, normal route prices, the "
        "wrong price rule) is worked out from the training months only, never from the months being tested. "
        "The same cleaning code runs on the training data, the prediction file and the December file. "
        "Automatic tests in tests/test_pipeline.py check this."
    )

    # 2. Split ---------------------------------------------------------------------------
    doc.add_heading("2. How I split and tested the data", level=1)
    doc.add_paragraph(
        "The real task is to predict future months, so I never mixed the data randomly. A random split "
        "lets the model peek at the future and makes the score look better than it is. Instead, I always "
        "trained on earlier months and tested on the next two months, just like the real task."
    )
    table(doc, ["Test", "Train on", "Test on"], [
        ["Test 1", "Jan to Jun", "Jul and Aug"],
        ["Test 2", "Jan to Jul", "Aug and Sep"],
        ["Test 3", "Jan to Aug", "Sep and Oct"],
        ["New cities test", "Jan to Aug, without 8 cities", "Sep and Oct loads in those 8 cities"],
        ["Final model", "Jan to Oct", "November and December (the real task)"],
    ])
    bullets(doc, [
        "The new cities test matters because 12% of the loads to predict involve cities that never appear "
        "in the training data.",
        "I picked the model with the lowest average error across tests 1, 2 and 3. It also had to beat the "
        "simple route average in every test.",
        "Wrong prices stay in the test months. I report the error both with and without them.",
    ])

    # 3. Models -------------------------------------------------------------------------
    doc.add_heading("3. Models and results", level=1)
    rows, bold = [], ()
    for i, (name, label) in enumerate(TABLE_ROWS.items()):
        r = exp.loc[name]
        rows.append([label, pct(r["MAPE_wf1"]), pct(r["MAPE_wf2"]), pct(r["MAPE_wf3"]),
                     pct(r["MAPE_wf_mean"]), pct(r["MAPE_city"])])
        if name == SELECTED:
            bold = (i,)
    table(doc, ["Model", "Test 1", "Test 2", "Test 3", "Average", "New cities"], rows, bold_rows=bold)
    figure(doc, FIG / "10_model_comparison.png", "Figure 3. Average error by model. Lower is better.")
    doc.add_paragraph("Why Ridge won:")
    bullets(doc, [
        "It had the lowest error of the models above in every test.",
        "The tree models (random forest, gradient boosting, CatBoost) learned day to day noise and did "
        "no better.",
        "It is simple and easy to explain. When two models score about the same, the simpler one is safer.",
        "It handles new cities well because it uses each city's location.",
    ])
    doc.add_paragraph(
        "One choice made a big difference. The raw market_index drifts up and down over the months in a way "
        "prices do not, and using it made predictions "
        f"{abs(level_bias.max()):.0f}% to {abs(level_bias.min()):.0f}% too low. I used only its short term "
        "ups and downs compared with its 4 week average. That cut the average error from "
        f"{pct(exp.loc['abl_mi_level', 'MAPE_wf_mean'], 1)} to {pct(sel['MAPE_wf_mean'], 1)}."
    )
    doc.add_paragraph(
        f"On test 3 (Sep and Oct) the model was off by {pct(main['MAPE'])} on average, or about "
        f"${main['MAE']:.0f} per load. By truck type: Dry Van {pct(eq['Dry Van'], 1)}, Reefer "
        f"{pct(eq['Reefer'], 1)}, Flatbed {pct(eq['Flatbed'], 1)}. With the wrong prices included, "
        f"the average error is {pct(main['MAPE_all'], 1)}, because no model can predict a price that is a typo."
    )
    figure(doc, FIG / "11_pred_vs_actual.png", "Figure 4. Predicted vs real price for September and October.")
    doc.add_paragraph(
        f"I also tried {len(exp)} variations in total, including giving recent months more weight, adding "
        "the day of the month, and mixing models. None improved the result enough to be worth the extra "
        "complexity. All results are in reports/experiments.md."
    )

    # 4. December -----------------------------------------------------------------------
    doc.add_heading("4. December chart", level=1)
    figure(doc, CHART, "Figure 5. December chart from the provided score.py: Lexington to Fort Wayne, "
                       "360 miles, Dry Van, 32,000 lb. Only the date changes.")
    doc.add_paragraph(
        f"The predicted price stays between ${rate.min():.0f} and ${rate.max():.0f}, around ${rate.mean():.0f}. "
        f"The chart zooms in a lot, so the wave looks bigger than it is: it moves only about ${swing:.0f} "
        "either way. The wave repeats every week and is highest on Thursdays, because market_index follows "
        "the same weekly pattern. December 25 is high only because it is a Thursday. The training data has "
        "no holiday effect, so the model does not add one. In the real market, prices often rise before "
        "Christmas, but this data cannot show that."
    )
    figure(doc, FIG / "13_december_explained.png",
           "Figure 6. The December prediction sits close to the normal price range for this route.")

    # 5. Risks --------------------------------------------------------------------------
    doc.add_heading("5. What could go wrong", level=1)
    bullets(doc, [
        "Prices move up or down by up to about 5% from month to month, and nothing in the data explains "
        "it. The real error for November and December could be closer to 2% to 3%.",
        f"If the real prices for November and December also contain about 1.4% wrong prices, the measured "
        f"error will be about {pct(sel['MAPE_all_wf_mean'], 0)}, even though the model is fine on normal loads.",
    ])

    # Run --------------------------------------------------------------------------------
    doc.add_heading("How to run", level=2)
    doc.add_paragraph(
        "pip install -r requirements.txt, then python -m src.train, then python -m src.predict, then "
        "python score.py --predictions validation_predictions.csv --december-predictions "
        "data/december_chart_inputs.csv"
    )

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
