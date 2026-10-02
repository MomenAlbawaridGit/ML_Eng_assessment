# Freight Rate Prediction

Predicts `posted_rate` (USD per truckload) for 12,000 Nov–Dec 2025 loads, and draws the fixed December chart for Lexington → Fort Wayne.

## Quick start

Requires Python 3.11+ (developed on 3.13.1, macOS).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data        # clean + validate all inputs (smoke test)
python -m src.train       # fit the final model on Jan–Oct 2025 -> models/
python -m src.predict     # write validation_predictions.csv + fill data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` validates both files and writes `scorer_results/candidate_december.png`.

Optional:

```bash
python -m src.evaluate    # walk-forward + city-holdout experiments -> reports/experiments.{md,csv} (slow: every experiment x 4 folds)
python -m src.plots_eda   # EDA / data-quality figures -> reports/figures/
python -m src.plots_model # model diagnostic figures -> reports/figures/
python -m src.report      # rebuild reports/freight_rate_report.docx
python -m pytest -q       # leakage / fold-hygiene / output-format tests
```

Every step is deterministic (fixed seeds). A clean install from `requirements.txt` on the same platform reproduced identical output files.

## Approach in one paragraph

The data is cleaned by one `Cleaner`, which is fit on the training rows only:
- negative weights are made positive (`abs()`), and missing weights get the median for their equipment type;
- `market_index` is replaced by its same-day mean;
- coordinates for the December file are looked up from a city table;
- training rows whose price-per-mile is less than 0.5× or more than 2× their lane reference are dropped (1.4% of rows).

`quote_signal` is **excluded** because it leaks the target. In 9 of the 10 training months it copies or mirrors price-per-mile, and in Nov–Dec it is pure noise.

The model is a **Ridge regression on log(price per mile)**, using:
- one-hot pickup city, delivery city and equipment;
- distance and weight splines;
- coordinates, so the model can place cities it never saw in training;
- the **detrended** daily `market_index`.

Models were selected with **walk-forward validation**: train ≤Jun → test Jul–Aug, train ≤Jul → test Aug–Sep, train ≤Aug → test Sep–Oct. A separate **city-holdout** fold rehearses the 12% of validation loads that touch unseen cities.

| Model | Walk-forward MAPE (excl. outliers) | City holdout |
|---|---|---|
| Baseline: equipment × distance median | 3.87% | 3.82% |
| Baseline: lane × equipment median | 3.40% | 3.68% |
| **Ridge + splines (selected)** | **1.84%** | **1.96%** |

All rows, figures and the reasoning behind these choices are in `reports/`. Start with `reports/data_findings.md` and `reports/experiments.md`.

## Layout

```
data/                     inputs (README names) + data/processed/ intermediates
src/config.py             paths, seeds, split dates
src/data.py               loading, validation, Cleaner, outlier rule, lane reference
src/features.py           model feature builders
src/models.py             model factory (Ridge / RF / HistGB / CatBoost, baselines)
src/evaluate.py           walk-forward + city-holdout evaluation, metrics
src/experiments.py        experiment registry, SELECTED model
src/train.py, predict.py  final fit and prediction
src/plots_*.py            figures
reports/                  findings, experiment tables, figures, report
models/                   final model (created by `python -m src.train`)
score.py                  provided scorer (unchanged)
```

## Notes

- LightGBM was not used because it needs the system `libomp` library on macOS. scikit-learn's `HistGradientBoostingRegressor` is evaluated in its place. CatBoost is only needed for `src.evaluate`.
