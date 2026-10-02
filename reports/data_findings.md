# Data findings

All numbers were reproduced in code with `.venv/bin/python` on `data/train_test.csv` (48,000 rows), `data/validation.csv` (12,000) and `data/december_chart_inputs.csv` (31). The pipeline is in `src/data.py` and the constants are in `src/config.py`.

## 0. Headline: `quote_signal` is a month-toggled target leak, so it is dropped as a feature

| Month | Rows | quote_signal within 3% of rate-per-mile | within 1% | corr(QS, ppm) | corr(QS, lane×eq median ppm) | median QS/ppm |
|---|---|---|---|---|---|---|
| Jan | 4,918 | 97.8% | 64.2% | 0.45 | **0.81** | 1.000 |
| Feb | 4,337 | 98.0% | 65.0% | 0.48 | **0.87** | 1.000 |
| Mar | 5,036 | 98.1% | 66.3% | 0.49 | **0.85** | 1.000 |
| Apr | 4,819 | 10.9% | 3.6% | −0.47 | **−0.79** | 0.934 |
| May | 4,913 | 10.3% | 3.6% | −0.46 | **−0.75** | 0.895 |
| Jun | 4,783 | 98.4% | 68.1% | 0.48 | **0.87** | 1.000 |
| Jul | 4,912 | 9.6% | 3.2% | −0.47 | **−0.87** | 0.897 |
| Aug | 4,759 | 14.7% | 5.3% | −0.01 | −0.02 | 0.961 |
| Sep | 4,670 | 98.2% | 66.6% | 0.48 | **0.87** | 1.000 |
| Oct | 4,853 | 10.3% | 3.1% | −0.43 | **−0.81** | 0.915 |
| Nov (val) | 5,836 | n/a | n/a | n/a | −0.008 | n/a |
| Dec (val) | 6,164 | n/a | n/a | n/a | −0.004 | n/a |

- Pooled over all months, the signs cancel. The overall corr(QS, ppm) is only 0.049, which is why it looked "weak".
- In "leak" months, QS equals the *uncorrupted* rate-per-mile. For outlier rows, QS/ppm is 0.44 while QS/lane median is 0.986. That gives us ground truth for the outlier rule (§3d).
- In Nov–Dec it carries no lane information (corr −0.005). A model that learns it on Jan–Oct would get a test score that cannot be trusted, and its validation predictions would break.
- **Decision:** `quote_signal` is excluded from `MODEL_FEATURES` (`config.LEAKY_COLUMNS`). The pipeline passes it through for diagnostics only, so no December fill is needed (resolves OPEN b).

## 1. Verification of initial EDA findings

| Claim | Status | Actual value |
|---|---|---|
| Negative weights: train 292, val 145 | confirmed | 292 / 145; range −5,000 to −47,500, same distribution as positives |
| Missing weight: train 300, val 165 | confirmed | 300 / 165 |
| Weight medians per equipment ~31.4–31.6k | confirmed | DV 31,444 · Flatbed 31,532 · Reefer 31,577 (after abs) |
| Missing market_index: train 374, val 249 | confirmed | 374 / 249; no day has fewer than 114 known values |
| 47,500 lb cap: 1,191 rows | confirmed, with a nuance | 1,191 raw; **1,204 after abs()** (13 were −47,500). Val 297 → 301. There is also a floor at 5,000 lb (32 train / 7 val rows). Weights are clipped to [5,000, 47,500] |
| Missing values have normal prices (no flags) | confirmed | median ppm/lane ratio = 1.00 for missing-MI, missing-weight and negative-weight rows |
| ~630 price outliers (1.3%) | **differs** | **677 (1.41%)**: 337 low (×0.17–0.47), 340 high (×2.18–5.41). "~630" came from lane medians on 1–2-row lanes, where the outlier itself contaminates the median. See §3d |
| Coordinates ~112 mi off | confirmed (≈) | median 110 mi over all 72 cities (127 mi over the 64 train cities); worst: Salt Lake City 402, Phoenix 395, LA 390, Providence 324 |
| LA / Tampa / Providence data coordinates | recorded | LA (28.566, −116.702) vs real (34.052, −118.244); Tampa (28.358, −86.948) vs (27.951, −82.457); Providence (37.373, −69.500) vs (41.824, −71.413) |
| distance fits data coords (~1.2× haversine) | confirmed | distance / haversine(data): median 1.182, IQR 1.165–1.205, corr 0.9995. With real coords: IQR 1.05–1.34, p1–p99 0.55–2.77, corr 0.971 |
| Providence→Hartford would be 4× with real coords | confirmed | median distance 264.5 mi vs real haversine 65.1 mi (4.07×), data haversine 209 mi |
| Coordinates are clipped | **new** | lat floor 25.5 (Laredo, val only), lon ceiling −69.5 (Providence, Boston). Each city has exactly one coordinate pair, and train and val agree |
| 8 val cities unseen in train, 12% of rows | confirmed | Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego; 1,447 rows (12.1%). Unseen lanes 12.2%, unseen lane×equipment 15.4% |
| Loads/day 158 train vs 197 val | confirmed | 157.9 / 196.7 |
| ppm Jan $2.03 → Jun $2.25 → Oct $2.16 | confirmed (monthly median ppm) | 2.029, 2.061, 2.139, 2.147, 2.190, **2.248**, 2.187, 2.121, 2.160, **2.164** |
| market_index tracks daily prices, corr 0.64 | confirmed (definition-dependent) | daily mean MI vs daily median ppm 0.642; vs daily median lane-normalised ppm 0.685; vs daily mean ppm 0.577. Row-level corr is only 0.08 |
| MI Sep–Oct ~0.89–0.96 vs Nov–Dec ~0.92–0.94 | confirmed (monthly means) | Sep 0.893, Oct 0.958, Nov 0.919, Dec 0.935 |
| MI rises slightly through December | confirmed | Dec weekly peaks 1.024 → 1.034 → 1.045 → 1.027 (31st); weekly troughs 0.831 → 0.838 → 0.844 → 0.858 |
| quote_signal weak (corr 0.08) | **differs** | pooled corr 0.049, but it is a month-toggled leak (§0) |
| No weekday effect | **differs** | MI has a strong weekday cycle (detrended Mon −0.08, Thu +0.10, Sun −0.09; ACF lag 7 = 0.92). Lane-normalised ppm follows it: Sun −0.9% to Thu +0.8% (ACF lag 7 = 0.45). Small but real |
| No holiday / month-end effect | confirmed | ±2 days around federal holidays: ratio 0.995 vs 1.000; day ≥ 28: 1.000 vs 1.000 |
| Short trips pricier: $2.74 <250 mi vs $1.91 >2,000 mi | confirmed | median 2.745 vs 1.912 (mean 2.815 vs 1.942) |
| Reefer +13%, Flatbed +8% vs Dry Van | confirmed | raw median +13.0% / +8.5%. Controlled for lane: Reefer +11%, Flatbed +7% |
| Southeast +6–7%, West −8% | **differs (confounded by distance)** | raw ppm by pickup: West −6.3%, SE +3.8%. West pickups have median 1,899 mi vs 953 overall. Controlled for equipment × distance bucket: West 0.0%, Southeast +0.8%, South Central +2.2%, Northeast −2.2% |
| Direction barely matters | confirmed | median \|A→B / B→A − 1\| of lane ppm = 3.0% (1,998 pairs) |
| Weight has a small effect | confirmed | ppm ratio: <20k lb 0.963, 20–30k 0.983, 30–40k 1.006, 40k+ 1.02 (corr 0.08) |
| No duplicates, no pickup == delivery | confirmed | 0 duplicate IDs, 0 exact duplicates excluding the ID. One pair matches on lane+date+equipment+weight (TR-012210 / TR-012304) but has different rates, so both are kept. 0 self-lanes |
| Dates | confirmed | all ISO `YYYY-MM-DD`; train 2025-01-01..10-31, val 11-01..12-31; no unparseable dates |
| Equipment spelling | confirmed clean | exactly {Dry Van, Reefer, Flatbed}; no variants |
| Train vs val covariate shift | confirmed none except dates | equipment shares 56.7/25.1/18.2 vs 56.5/25.4/18.1; distance p10/50/90 identical (328/953/2,216 vs 328/954/2,225); abs-weight quantiles identical. (Signed weight std differs only because of the negative values) |

## 2. OPEN-item resolutions

### a. market_index fill: same-day mean, applied to **every** row
- **Structure:** MI = per-day value + row noise. The day mean explains R² = 0.978. Within-day std is 0.025, while the day-to-day jump of the daily mean is 0.057. The within-day deviation has no structure: by equipment ±0.0000, by longitude band ±0.0002, by city ±0.002. Its corr with lane-normalised ppm is 0.011, so it is pure noise.
- **Masking test:** 5% of known values masked at random, 10 seeds, MAE vs truth:

| Fill | Train MAE | Val MAE |
|---|---|---|
| same-day mean | **0.0200** | **0.0202** |
| ±1-day mean | 0.0254 | 0.0257 |
| ±5-day mean (row- or day-weighted) | 0.0795 | 0.0753 |
| global mean | 0.1441 | 0.0645 |
| floor (true day mean incl. masked rows) | 0.0198 | 0.0197 |

- Same-day hits the noise floor, and ±5-day is 4× worse because it averages over the weekly cycle.
- **Decision:** replace every row's MI with the same-day mean. This removes noise and handles missing values in one step. The raw value is kept as `market_index_raw` for diagnostics.
- Also provided: `market_index_trend`, a centred 28-day rolling mean (a multiple of 7, so the weekly cycle cancels), and `market_index_detrended` = daily − trend.
- The daily series is built from **train + validation inputs only (no labels)**. That way Nov–Dec days exist (December rows take the validation same-day means) and the centred window has data across the Oct/Nov boundary.
- Each December day has at least 163 validation rows.
- MI *level* does not map to price level across months (Apr–Jul MI 1.2–1.3, prices only +1–3%). The detrended version may generalise better; this is tested in the model ablations (reports/experiments.md).

### b. quote_signal for December: not needed
quote_signal is excluded from features (§0). For the record, the Lexington→Fort Wayne Dry Van training median is 2.111 (21 rows; 32 rows for all equipment, median 1.973), and the 6 validation Dry Van rows have a median of 1.962.

### c. Same route, different distances: random per-load noise, so keep the raw distance
- 3,947 routes with ≥2 rows. Within-route range: median 49.5 mi, p90 117.7, max 273.5.
- The relative deviation from the route median is symmetric: std 2.2%, p1/p99 −5.7%/+5.8%. It scales with route length (abs std 8 mi under 200 mi of haversine, 41 mi over 2,000 mi).
- Coordinates are fixed per city, so haversine is constant per route. The variation is not caused by coordinates.
- It is unrelated to equipment (mean dev ≤0.05%), month (≤0.05%), weight (corr 0.004), MI (0.002) and direction (A→B vs B→A median distance differ by 0.7%).
- **Price follows the per-row distance:** within-lane elasticity of rate w.r.t. distance is 0.868, the same as the cross-lane 0.871. Per-row distance is therefore a real input (like a routed mileage), not noise to smooth away.
- **Decision:** keep raw `distance` as a feature. Do not replace it with a route mean.

### d. Outlier rule
- `ratio = (posted_rate / distance) / reference`. The reference is the training median rate-per-mile of **pickup × delivery × equipment** when that lane has ≥ 5 training rows (`LANE_MIN_SUPPORT`). Otherwise it is the training median of **equipment × distance bucket** (`[0,250,500,1000,1500,2000,∞)`).
- Flag if `ratio < 0.5` or `ratio > 2.0`.
- The thresholds sit inside an empty gap. Kept rows have ratio 0.81–1.27 (p0.1–p99.9 0.86–1.16), while outliers have 0.17–0.47 and 2.18–5.41. Thresholds 0.5/2, 0.6/1.67 and support k = 3/5/8 all flag exactly the same rows.
- **Ground truth:** in QS-leak months, rows with ppm/QS outside [0.7, 1.4] are the corrupted ones (336 rows, a clean bimodal split). The rule's precision and recall are both 100% (0 FP / 0 FN). Even the equipment × distance fallback alone scores 100%.
- **Counts:** 677 flagged in Jan–Oct (1.41%), 57–81 per month. With the reference fit on Jan–Aug: Jan–Aug 533, **Sep–Oct 144 (Sep 64, Oct 80)**, the same rows as with the full fit.
- **Handling:** `remove_outliers()` is applied to training rows only. The test fold keeps every row with `is_outlier` set, so metrics can be reported with and without them. Validation and December are never dropped (`is_outlier` = NA there).

## 3. Final cleaning rules

| # | Rule | Rows affected (train / val / Dec) | Why | Fit on | Applied to |
|---|---|---|---|---|---|
| 1 | `weight = abs(weight)` | 292 / 145 / 0 | sign errors with an identical magnitude distribution and normal prices | none | all |
| 2 | Missing weight → training median per equipment (DV 31,444 / FB 31,532 / RF 31,577 on Jan–Oct) | 300 / 165 / 0 | missing at random, normal prices; medians nearly equal | training part | all |
| 3 | Keep 47,500 cap and 5,000 floor | 1,204+32 / 301+7 / 0 | plausible legal and data-entry clip; no price anomaly (ratio 1.02 at cap) | none | all |
| 4 | Missing coordinates → training city table | 0 / 0 / 31 | December file has none; both cities in train; given coordinates never overwritten | training part | all (only fills NaN) |
| 5 | `market_index` → same-day mean for every row, plus 28-day centred trend and detrended | all rows (374 / 249 / 31 were missing) | per-row deviation is noise; same-day hits the masking floor | train+val **inputs** (no labels) | all |
| 6 | `day_of_week` feature | all | weekly cycle of ±1% in price | none | all |
| 7 | Outlier flag (ratio rule above) | 677 / n/a / n/a | label corruption, confirmed against QS ground truth | training part | flag: rows with target; **removal: training rows only** |
| 8 | Drop `quote_signal` from features | all | month-toggled leak, noise in Nov–Dec | none | all |
| 9 | Coordinates kept as-is (not replaced with real ones) | all | `distance` is consistent with them, not with real coordinates | none | all |
| 10 | No missing-value indicators | n/a | missing rows priced normally | none | all |

## 4. Split and leakage
- Model selection: fit on Jan–Aug (38,477 rows; 37,944 after outlier removal) and test on Sep–Oct (9,523, of which 144 are outliers). Final refit: Jan–Oct (47,323 after removal).
- Dates do not overlap, so no lane-date group straddles the split. There are no duplicate loads.
- Sep is a QS-leak month and Oct is anti-correlated. Any model using QS would be scored on a corrupted test, which is another reason it is excluded.
- Transductive use of inputs: the MI daily series and its centred trend use train+validation **inputs**. For Aug rows near the split, the trend window includes Sep MI values. No labels are involved, and at inference all Nov–Dec MI is given, so this is legitimate.
- `LaneRateReference.oof_transform()` gives out-of-fold (5-fold, seed 42) lane references for training rows if they are used as a target-encoding feature.
- A sanity baseline using the lane reference alone (no time adjustment), fit on Jan–Aug without outliers, gets Sep–Oct MAPE **3.29%** on non-outlier rows (lane level 3.04%, city fallback 3.51%; the pure equipment × distance fallback would be 3.87%).

## 5. Addendum: market_index within each month (2026-10-02)
Correlation between the daily mean `market_index` and the daily median lane-normalised price per mile (price per mile ÷ lane median, clipped to [0.5, 2]), computed within each calendar month:

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 0.91 | 0.75 | 0.95 | 0.88 | 0.33 | 0.93 | 0.76 | 0.23 | 0.76 |

In 8 of 10 months the daily index moves with prices (r 0.75–0.95). The monthly *level* of the index does not track the monthly price level (§2a), which is why the model uses the detrended index.
