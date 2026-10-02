"""Predict validation loads and the December chart with the saved model.

    python -m src.predict
Writes validation_predictions.csv (repo root; load_id,predicted_rate) and fills
data/december_chart_inputs.csv (7 original columns, order kept).
"""

from __future__ import annotations

import joblib
import numpy as np

from src.data import build_datasets, write_december_predictions, write_validation_predictions
from src.train import MODEL_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    data = build_datasets(final=True)  # same deterministic Cleaner fit (Jan-Oct) as in training
    val, dec = data["validation"], data["december"]

    p_val = model.predict(val)
    p_dec = model.predict(dec)
    for name, p in (("validation", p_val), ("december", p_dec)):
        if not (np.isfinite(p).all() and (p > 0).all()):
            raise ValueError(f"{name}: non-finite or non-positive predictions")

    out = write_validation_predictions(val["load_id"], p_val)
    print(f"validation: {len(out)} rows; predicted_rate quantiles "
          f"{np.percentile(p_val, [1, 25, 50, 75, 99]).round(0).tolist()}")
    d = write_december_predictions(dec["date"], p_dec)
    r = d["predicted_rate"]
    print(f"december: min {r.min():.2f} max {r.max():.2f} mean {r.mean():.2f} "
          f"range {100 * (r.max() / r.min() - 1):.2f}%")
    print(d.assign(dow=d["date"].pipe(lambda s: s.astype("datetime64[ns]").dt.day_name().str[:3]))
          [["date", "dow", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
