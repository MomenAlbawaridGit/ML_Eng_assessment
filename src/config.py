"""Project-wide configuration: paths, seeds, split dates and cleaning constants.

Every path and magic number used by the pipeline lives here so it is changed in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Paths -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
DECEMBER_ORIGINAL_PATH = DATA_DIR / "december_chart_inputs.original.csv"  # pristine backup, read-only
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

# --- Reproducibility -------------------------------------------------------------------
SEED = 42

# --- Time split (train on the past, predict the next period) ---------------------------
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-08-31")  # model-selection training part: Jan-Aug
TEST_START = pd.Timestamp("2025-09-01")  # model-selection test part: Sep-Oct
TEST_END = pd.Timestamp("2025-10-31")
FINAL_TRAIN_END = pd.Timestamp("2025-10-31")  # final refit: Jan-Oct

# --- Schema ----------------------------------------------------------------------------
TARGET = "posted_rate"
EQUIPMENT_TYPES = ("Dry Van", "Reefer", "Flatbed")
RAW_INPUT_COLUMNS = (
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date", "market_index", "quote_signal",
)
DECEMBER_COLUMNS = ("pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate")

# quote_signal is a month-toggled target leak (see reports/data_findings.md): never a feature.
LEAKY_COLUMNS = ("quote_signal",)

# --- Cleaning / feature constants ------------------------------------------------------
OUTLIER_LOW = 0.5  # flag if rate-per-mile / reference < 0.5 ...
OUTLIER_HIGH = 2.0  # ... or > 2.0 (symmetric in log space; the gap 0.5-0.8 / 1.25-2.0 is empty)
LANE_MIN_SUPPORT = 5  # min training rows for a lane x equipment median to be trusted
DISTANCE_BUCKETS = (0, 250, 500, 1000, 1500, 2000, float("inf"))
MI_DETREND_WINDOW = 28  # days; multiple of 7 so the weekly market_index cycle cancels
MI_DETREND_MIN_PERIODS = 7
