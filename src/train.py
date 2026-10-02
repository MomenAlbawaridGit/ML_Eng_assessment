"""Fit the selected model on Jan-Oct 2025 (outliers removed) and save it to models/.

    python -m src.train
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from src import config as C
from src.data import build_datasets
from src.experiments import SELECTED
from src.models import make_model

MODEL_PATH = C.MODELS_DIR / "final_model.joblib"
META_PATH = C.MODELS_DIR / "final_model.json"


def main() -> None:
    data = build_datasets(final=True)  # Cleaner fit on Jan-Oct; outliers dropped from train only
    train = data["train"]
    model = make_model(SELECTED).fit(train)
    fitted = model.predict(train)
    ape = np.abs(fitted - train[C.TARGET]) / train[C.TARGET]
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": SELECTED,
        "train_rows": int(len(train)),
        "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
        "in_sample_mape_pct": round(100 * float(ape.mean()), 3),
        "seed": C.SEED,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
