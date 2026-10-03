# Freight Rate Prediction

This project predicts the price of 12,000 truck loads in November and December 2025. It also produces the December chart for Lexington to Fort Wayne.

## How to run

You need Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train      # train the final model on January to October 2025
python -m src.predict    # write validation_predictions.csv and fill the December file
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` checks both files and saves the chart to `scorer_results/candidate_december.png`.

Other commands:

```bash
python -m src.evaluate     # rerun all model tests (slow)
python -m src.plots_eda    # redraw the data figures
python -m src.plots_model  # redraw the model figures
python -m src.report       # rebuild the report
python -m pytest -q        # run the automatic checks
```

The results are the same every time you run it.

## What I did

**Cleaning the data**
* Removed `quote_signal`. It copies the price in some months and is useless in November and December.
* Removed 677 loads (1.4%) with clearly wrong prices from training.
* Made negative weights positive and filled missing weights with the usual weight for that truck type.
* Filled missing `market_index` values with the average for that day.

**Testing**
* I trained on earlier months and tested on the next two months, three times, just like the real task.
* An extra test checks cities the model has never seen.

**Results** (average error, lower is better)

| Model | Average error | New cities |
|---|---|---|
| Simple route average | 3.40% | 3.68% |
| Ridge (chosen model) | **1.84%** | **1.96%** |

Full details are in `reports/freight_rate_report.docx`.

## Folders

```
data/       input files
src/        all code
tests/      automatic checks
reports/    report, figures and test results
```
