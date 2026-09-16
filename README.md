# California Housing Price Prediction

A regression project that predicts the final sale price (`ClosePrice`) of single-family
residential properties using monthly CRMLS (California Regional MLS) sold-listing data, with
several models trained, tuned, and compared.

## 1. Dataset

- **Source**: CRMLS (California Regional MLS) monthly sold-listing data (files named
  `CRMLSSold{YYYYMM}.csv`, covering 2025-05 through 2026-04)
- **Filter applied**: `PropertyType == "Residential"`, `PropertySubType == "SingleFamilyResidence"`
  (141,997 rows after filtering)
- **Target**: `ClosePrice` (final sale price)
- **Key raw features**: `LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, `LotSizeArea`,
  `YearBuilt`, `AssociationFee`, `Flooring`, `PostalCode`, etc.
- **Supplementary data**: school district boundaries — 
  [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25)
  (data.ca.gov, GeoJSON)
- Column-level descriptions and drop decisions are documented in `columns.xlsx`.

> The raw CSV files (`CRMLSSold*.csv`) are not included in this repository due to size/licensing
> constraints. To reproduce the pipeline, place CRMLS sold-listing data in the same format at the
> project root.

## 2. Repository Structure

```
├── 01_exploration.ipynb        # Merge monthly CSVs into one file, filter, exploratory analysis (EDA)
├── 02_preprocessing.ipynb      # Missing-value handling, outlier handling, feature engineering, encoding/scaling, train/test split
├── 03_baseline_model.ipynb     # Linear Regression baseline (log-transformed target)
├── 04_model_comparison.ipynb   # Decision Tree / Random Forest comparison and depth tuning
├── 05_advanced_models.ipynb    # XGBoost / LightGBM hyperparameter search
├── 06_evaluation.ipynb         # Walk-forward CV tuning of all 5 models + MAPE/MdAPE evaluation by price band
├── columns.xlsx                # Column descriptions and drop decisions
└── README.md
```

## 3. Preprocessing Summary (`02_preprocessing.ipynb`)

1. **Drop rows with missing/invalid target**: rows with missing `ClosePrice` or `ClosePrice <= 0`
   are removed.
2. **Missing-value handling**: columns with more than 50% missing values are dropped; the rest
   are imputed (median for numeric, most-frequent for categorical), with
   `SimpleImputer(add_indicator=True)` also adding a flag column marking which rows were
   originally missing.
3. **Column-specific fixes**
   - `AssociationFee`: missing values filled with 0 (both NaN and 0 represent "no HOA" in this
     dataset).
   - `Flooring`: the original column is kept and multi-hot encoded (`Flooring_Wood`,
     `Flooring_Carpet`, etc.), while an additional `premium_score` / `grade` summary feature is
     derived from the flooring material mix.
4. **Leakage / unnecessary column removal**: IDs, URLs, photos, agent info, date strings,
   near-unique address-like columns, and very-high-cardinality categorical columns (`City`,
   `PostalCode` as a raw feature, etc.) are excluded.
5. **Time-based train/test split**: the most recent month is fixed as the test set, and the
   preceding N months (N ∈ {3, 6, 9, 12}) are used as the training window — the N with the lowest
   RMSE is selected.
6. **Outlier handling** (train-only, log1p-space IQR with Tukey `k=1.5`):
   - `ClosePrice`: rows outside the train-only IQR bounds ($184,446 – $4,777,794) are **dropped**,
     but only within the training window — test rows are always kept as-is.
   - `LivingArea` / `LotSizeSquareFeet`: **clipped** (winsorized) to the train-only bounds,
     applied to both train and test.
7. **Feature engineering**
   - `BedBathRatio` = bedroom count / bathroom count
   - `PropertyAge` = sale year − `YearBuilt`
   - `SchoolDistrictGIS`: each property's latitude/longitude spatially joined against CA school
     district boundaries to assign a district label. Only the top 30 districts (by frequency in
     the training data) are kept; the rest are bucketed into `"Other"`.
   - **Area-level trailing comps**: for each `PostalCode`, the median/mean price-per-sqft of sales
     over the 3 months immediately *before* that sale's own month (`AreaCompsMedianPricePerSqFt`,
     `AreaCompsMeanPricePerSqFt`, `AreaCompsSalesCount`). Built causally (`shift(1)` + rolling
     window before use) so a row's own sale is never part of its own comp.
   - The raw `Latitude`/`Longitude` coordinates are excluded from the model inputs (to prevent the
     coordinates themselves from dominating the model).
   - With these features added, the final feature set grows from 22 raw columns (14 numeric + 8
     categorical) to 28 raw columns (19 numeric + 9 categorical).
8. **Preprocessing pipeline**: numeric (median impute + StandardScaler), categorical
   (most-frequent impute + OneHotEncoder) — `fit` only on the training set, `transform` only on
   the test set, to avoid data leakage.
9. **Output**: `cleaned_preprocessed_data.csv` (138,859 rows × 120 columns; a `split` column marks
   train/test/unused rows — 126,835 train / 12,024 test after outlier removal).

## 4. Modeling Approach

All models predict `log1p(ClosePrice)` rather than the raw price (predictions are back-transformed
with `expm1` before scoring) — this, combined with the outlier handling and new features in
`02_preprocessing.ipynb`, dramatically improved every model's test-set performance compared to
earlier iterations of this project (where unconstrained trees produced deeply negative test R²).

The modeling work builds up notebook by notebook:

- **`03_baseline_model.ipynb`**: Linear Regression baseline.
- **`04_model_comparison.ipynb`**: Decision Tree and Random Forest (default settings), plus a
  depth-constrained sweep to check for overfitting.
- **`05_advanced_models.ipynb`**: XGBoost and LightGBM, lightly tuned (`max_depth`,
  `learning_rate`, `n_estimators`, plus `num_leaves` for LightGBM) against a single held-out
  validation month.
- **`06_evaluation.ipynb`** (final, most rigorous pass): all 5 models (Linear Regression, Decision
  Tree, Random Forest, XGBoost, LightGBM) are re-tuned using **3-fold walk-forward
  (rolling-origin) cross-validation** within the training window, selecting hyperparameters by
  **validation MdAPE** rather than R²/RMSE. The tuned models are then evaluated once on the true
  test month, overall and broken down by price quintile, using MAPE and MdAPE in addition to
  R²/MAE/RMSE.

## 5. Final Results (`06_evaluation.ipynb`)

Overall test-set performance, retrained on the full training window with the best
walk-forward-CV hyperparameters:

| Model | R² | MAE ($) | RMSE ($) | MAPE (%) | MdAPE (%) |
|---|---|---|---|---|---|
| **XGBoost (tuned)** | 0.405 | 237,625 | 1,294,507 | 16.92 | **9.22** |
| **LightGBM (tuned)** | 0.404 | 237,801 | 1,295,490 | 16.74 | **9.17** |
| Random Forest (tuned) | 0.377 | 249,359 | 1,324,845 | 17.19 | 9.28 |
| Decision Tree (tuned) | 0.368 | 282,699 | 1,334,262 | 19.95 | 11.61 |
| Linear Regression | 0.316 | 306,216 | 1,387,910 | 23.84 | 13.55 |

- **Best hyperparameters**:
  - XGBoost: `{max_depth: 7, learning_rate: 0.1, n_estimators: 400, min_child_weight: 1, subsample: 0.8}`
  - LightGBM: `{max_depth: 7, num_leaves: 63, learning_rate: 0.1, n_estimators: 400, min_child_samples: 10}`
  - Random Forest: `{max_depth: None, n_estimators: 400, min_samples_leaf: 1, max_features: 0.5}`
  - Decision Tree: `{max_depth: None, min_samples_leaf: 10}`
- **XGBoost and LightGBM are effectively tied** for best overall (MdAPE within 0.05 percentage
  points, R² within 0.001) — either is a reasonable choice, with LightGBM's lower fold-to-fold
  RMSE variance during tuning a mild point in its favor for stability.
- **Random Forest closes most of the gap** to the boosted models once `max_features=0.5` is tuned
  in, but still trails by roughly 0.1 percentage points of MdAPE.
- **Decision Tree and Linear Regression remain the weakest** models by every metric — a single
  tree still overfits even after tuning `min_samples_leaf`, and Linear Regression can't capture
  the non-linear pricing patterns the tree-based models pick up.

**Price-band breakdown** (5 quintiles of test-set `ClosePrice`):

- MdAPE for XGBoost/LightGBM is mildly **U-shaped**: ~8.5% in the cheapest quintile, dropping to
  ~7.2% in the second quintile, then climbing back up to ~13% in the most expensive quintile
  (≥ $1.69M). So relative error does concentrate somewhat at the high-price tail, but far less
  dramatically than raw RMSE numbers would suggest.
- R² computed *within* a narrow price band is unreliable and often strongly negative even for the
  best models — restricting the price range shrinks the variance in the denominator, exaggerating
  R² swings from the same absolute errors. MAPE/MdAPE (per-row percentage ratios) don't have this
  problem and are the metrics to trust for band-level comparisons.

**Conclusion: XGBoost and LightGBM are the recommended models from this comparison**, with very
similar accuracy overall and across price bands. Their main remaining weakness is concentrated in
the top ~20% of listings by price, where dollar errors are large in absolute terms even though
relative (percentage) error only degrades modestly.

## 6. How to Reproduce

### 6.1 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl geopandas shapely xgboost lightgbm jupyter
```

- `geopandas` / `shapely` are only needed for the school-district spatial join in
  `02_preprocessing.ipynb` (section 9.1). (In a conda environment,
  `conda install -c conda-forge geopandas` tends to be more reliable.)
- Python 3.9+ recommended.

### 6.2 Run Order

Each notebook reads a CSV produced by the previous one, so they **must be run in order**.

1. **`01_exploration.ipynb`**: place the raw `CRMLSSold*.csv` files in the project root and run →
   produces `merged_sales_data.csv`
2. **`02_preprocessing.ipynb`**: reads `merged_sales_data.csv` and preprocesses it →
   produces `cleaned_preprocessed_data.csv`, `cleaned_preprocessed_data_old.csv`,
   `feature_engineering_comparison.csv`
   - The school-district spatial join (section 9.1) automatically downloads a CA school-district
     GeoJSON. If your environment has no internet access, download the GeoJSON in advance and
     save it as `CA_School_District_Areas_2024-25.geojson` to use the local fallback.
3. **`03_baseline_model.ipynb`**: reads `cleaned_preprocessed_data.csv` and trains Linear
   Regression on `log1p(ClosePrice)` → produces `baseline_model_results.csv`
4. **`04_model_comparison.ipynb`**: compares Decision Tree / Random Forest →
   produces `model_comparison_results.csv`
5. **`05_advanced_models.ipynb`**: lightly tunes XGBoost / LightGBM →
   produces `advanced_model_tuning_results.csv`, `advanced_model_results.csv`
6. **`06_evaluation.ipynb`**: re-tunes all five models with walk-forward CV and evaluates them
   together (MAPE/MdAPE, performance by price band) → produces `metrics_summary.csv`. This is the
   final, most rigorous evaluation and its results table (Section 5 above) is the one to cite.

Each notebook can simply be run top-to-bottom ("Run All") in Jupyter.

```bash
jupyter notebook
# or
jupyter lab
```

## 7. Notes / Limitations

- The test set is always a single "most recent month," so the sample size is limited and model
  performance can be sensitive to that particular month's market conditions.
- `ClosePrice` is heavily right-skewed; predicting `log1p(ClosePrice)` substantially improved
  model stability and test performance versus predicting the raw price directly.
- R² computed within a narrow price band is misleading (see notebook 06, Section 7) because the
  reduced variance in the denominator exaggerates R² swings for the same absolute error — per-band
  comparisons should be read using MAPE/MdAPE rather than R².
- Model accuracy degrades somewhat for the highest-priced listings (top quintile); if this model
  is used for pricing decisions, that weakness is worth keeping in mind specifically for luxury
  properties.

## 8. (Optional, Not Included) Prediction App

A Streamlit app (`app.py`) that takes `LivingArea`, `Beds`, `Baths`, and `LotSize` as input and
outputs a predicted price is not included in this submission. If added later, it can be built by
saving the final tuned model (XGBoost or LightGBM, per Section 5) from `06_evaluation.ipynb` with
`joblib` and loading it in the app.
