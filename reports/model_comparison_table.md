MAPE (%) by validation fold. wf1: train ≤Jun → test Jul–Aug; wf2: ≤Jul → Aug–Sep; wf3: ≤Aug → Sep–Oct. Outlier rows (1.4%) excluded except in *Incl. outliers* (mean of wf1–wf3). *City holdout*: trained Jan–Aug without 8 held-out cities, tested on Sep–Oct loads touching them. Source: `reports/experiments.csv`.

| Model | wf1 | wf2 | wf3 | Mean | Incl. outliers | City holdout |
|---|---:|---:|---:|---:|---:|---:|
| (a) Equipment × distance median | 3.97 | 3.83 | 3.82 | 3.87 | 6.19 | 3.82 |
| (b) Lane × equipment median | 3.57 | 3.35 | 3.29 | 3.40 | 5.73 | 3.68 |
| (c′) (b) × detrended market_index | 3.41 | 3.28 | 3.25 | 3.32 | 5.65 | 3.65 |
| Random forest (MI offset) | 2.24 | 2.04 | 2.03 | 2.11 | 4.45 | 2.20 |
| HistGB (MI offset) | 2.16 | 1.90 | 1.89 | 1.98 | 4.32 | 2.42 |
| Ridge, linear terms | 2.04 | 1.84 | 1.86 | 1.91 | 4.26 | 2.05 |
| CatBoost (MI offset) | 2.05 | 1.82 | 1.80 | 1.89 | 4.23 | 1.97 |
| **Ridge + splines (selected)** | **1.99** | **1.76** | **1.78** | **1.84** | **4.19** | **1.96** |
| Ridge, MI level + detrended + day of week (locked set) | 2.24 | 4.31 | 4.80 | 3.78 | 6.03 | 4.75 |
| Ridge + splines + HistGB on residual | 1.96 | 1.73 | 1.74 | 1.81 | 4.16 | 1.99 |
