# California Home Sale Price Prediction

A regression project that predicts home sale price (`ClosePrice`) using monthly CRMLS (California Regional MLS) sales data. The work was carried out over 10 weeks: data exploration → preprocessing/feature engineering → baseline model → model comparison → advanced models (gradient boosting) → extended evaluation.

> **Note**: In several places, the markdown narrative inside the notebooks disagreed with the actual numbers printed by the cells. This README is written from the **actual cell outputs**, not the markdown prose. (For example, the "Summary" section in `06_evaluation.ipynb` was written as a note before a fourth model (LightGBM) was added, and its numbers are off by roughly an order of magnitude from what the cells actually output.)

---

## 1. Dataset

- **Source**: CRMLS (California Regional MLS) sold-listing data, loaded from monthly files named `CRMLSSold{YYYYMM}.csv` and merged together (`notebooks/01_exploration.ipynb`).
- **Period**: 2025-05 through 2026-05 (13 monthly files)
- **Filtering**: restricted to `PropertyType == "Residential"` & `PropertySubType == "SingleFamilyResidence"`
- **Rows after filtering**: 141,997 rows, 79 columns (141,988 rows after further cleaning, e.g. dropping `ClosePrice <= 0`)
- **Target**: `ClosePrice` (sale price)
- **Raw data files are not included in this repository** (licensed MLS data, cannot be redistributed — see Section 4 for how to reproduce without it)
- Column definitions (name, description, type, whether dropped, etc.) are documented in `data/processed/columns.xlsx`.

---

## 2. Preprocessing (`notebooks/02_preprocessing.ipynb`)

1. **Missing values**
   - Columns with more than 50% missing values (23 columns) are dropped
   - The rest are imputed with `SimpleImputer(add_indicator=True)` (median for numeric, most_frequent for categorical), which also auto-generates a "was this value missing" indicator column
   - `AssociationFee` is filled with `0` rather than the median, since in this dataset both NaN and 0 mean "no HOA"
2. **`Flooring` → `premium_score` / `grade`**: a derived feature that scores (0–1) the mix of comma-separated flooring materials by premium vs. budget material ratio. The original `Flooring` column is kept and separately multi-hot encoded.
3. **Removing leakage / ID columns**: ID, URL, photo, and agent-related columns; date-type columns (auto-detected); columns where >90% of values are unique; columns with too many categories (City, PostalCode, MLSAreaMajor, HighSchoolDistrict, etc., over 50 categories); and constant columns (`PropertyType`, `MlsStatus`, `PropertySubType`) — 35 columns dropped in total.
4. **Feature engineering (added in Week 6)**
   - `BedBathRatio` = bedrooms / bathrooms
   - `PropertyAge` = sale year − `YearBuilt`
   - `SchoolDistrictGIS`: each property's latitude/longitude spatially joined against the [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25) boundaries to get a district label (overlapping matches resolved with priority Unified > Elementary > High). Only the top 30 districts (by frequency in the training window) are kept; everything else is bucketed into `"Other"` (~59% of rows end up in `"Other"`).
   - Raw `Latitude`/`Longitude` are excluded from the model inputs, since the coordinates otherwise dominate the location signal and make it hard to tell whether the engineered geographic features add anything.
5. **Time-based train/test split**
   - Test: the most recent month (2026-05)
   - Train: the preceding X months — candidates [3, 6, 9, 12] were searched by RMSE → **12 months (2025-05 to 2026-04) was best** (129,964 train rows / 12,024 test rows)
6. **Preprocessing pipeline**: numeric (median impute + StandardScaler) / categorical (most_frequent impute + OneHotEncoder), built with `ColumnTransformer` + `Pipeline`. `fit` is applied only to the training set; `transform` only to the test set and the full cleaned CSV.
7. **Old vs. new feature set comparison** (same train/test split, actual reported numbers)

   | Model | Feature set | R2 | MAE | RMSE |
   |---|---|---:|---:|---:|
   | LinearRegression | Original (22 raw columns) | 0.3076 | 572,953 | 1,396,359 |
   | LinearRegression | **+ Week 6 features (25 raw columns)** | **0.3365** | **547,257** | **1,366,886** |
   | RandomForest (unconstrained) | Original | -3.9518 | 543,759 | 3,734,124 |
   | RandomForest (unconstrained) | + Week 6 features | -4.5744 | 531,403 | 3,961,939 |
   | RandomForest (max_depth=5) | Original | -0.0273 | 597,218 | 1,700,795 |
   | RandomForest (max_depth=5) | **+ Week 6 features** | **0.0283** | 593,513 | 1,654,152 |

   → The three Week 6 features improved LinearRegression and the depth-constrained RandomForest (only the unconstrained RandomForest got worse). By feature importance, their contribution ranked `SchoolDistrictGIS` (0.0246, summed across its one-hot columns) > `BedBathRatio` (0.0153, rank 13 of 111) > `PropertyAge` (0.0010, rank 33 of 111).
8. **Outputs** (saved to `data/processed/`): `merged_sales_data.csv`, `cleaned_preprocessed_data.csv` (final, includes Week 6 features, with a `split` column marking train/test/unused), `cleaned_preprocessed_data_old.csv` (pre-Week-6 backup), `feature_engineering_comparison.csv`, `window_selection_results.csv`

---

## 3. Models tested & Results

All later notebooks share the same split defined by the `split` column in `data/processed/cleaned_preprocessed_data.csv` (129,964 train rows / 12,024 test rows, test fixed to 2026-05).

### 3.1 Baseline (`notebooks/03_baseline_model.ipynb`)
- `LinearRegression` (no hyperparameter tuning) → **R2 = 0.3365, MAE = 547,257, RMSE = 1,366,886**

### 3.2 Tree-based model comparison (`notebooks/04_model_comparison.ipynb`)
| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| LinearRegression (baseline) | 0.3365 | 547,257 | 1,366,886 |
| RandomForest (max_depth=5) | 0.0283 | 593,513 | 1,654,152 |
| DecisionTree (max_depth=5) | -0.5626 | 628,362 | 2,097,658 |
| RandomForest (unconstrained) | -4.5744 | 531,403 | 3,961,939 |
| DecisionTree (unconstrained) | -80.3816 | 802,622 | 15,138,073 |

- The unconstrained Decision Tree has train R2 ≈ 1.0 but a test R2 of -80 — severe overfitting, essentially memorizing individual rows in the sparse one-hot feature space. Random Forest overfits far less badly thanks to bagging, but is still worse than the baseline.
- Constraining `max_depth` recovers a large part of the test performance for both models, with RandomForest(max_depth=5) being the most stable.

### 3.3 Gradient boosting (`notebooks/05_advanced_models.ipynb`)
- The last month of the training window (2026-04) is carved out as a validation month for a grid search over `max_depth`/`learning_rate`/`n_estimators` (plus `num_leaves` for LightGBM); the true test month is evaluated exactly once at the end.
- Best hyperparameters: XGBoost `{max_depth:3, learning_rate:0.05, n_estimators:200}`, LightGBM `{max_depth:3, num_leaves:7, learning_rate:0.05, n_estimators:200}`

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| LinearRegression | 0.3365 | 547,257 | 1,366,886 |
| XGBoost (tuned) | 0.2824 | 508,814 | 1,421,509 |
| RandomForest (max_depth=5) | 0.0283 | 593,513 | 1,654,152 |
| LightGBM (tuned) | 0.0062 | 521,955 | 1,672,845 |

### 3.4 Extended evaluation — MAPE / MdAPE / price-band breakdown (`notebooks/06_evaluation.ipynb`)

Since R2/RMSE are heavily influenced by errors on a handful of very expensive properties, percentage-based metrics (MAPE, MdAPE) and a breakdown by price quintile were added.

**Overall metrics**

| Model | R2 | MAE | RMSE | MAPE | MdAPE |
|---|---:|---:|---:|---:|---:|
| LinearRegression | **0.3365** | 547,257 | 1,366,886 | 58.25% | 34.54% |
| XGBoost (tuned) | 0.2824 | **508,814** | 1,421,509 | 57.84% | **30.40%** |
| LightGBM (tuned) | 0.0062 | 521,955 | 1,672,845 | **56.57%** | 30.96% |
| RandomForest (max_depth=5) | 0.0283 | 593,513 | 1,654,152 | 65.67% | 38.45% |

- **The "best" model depends on which metric is used.** By explanatory power (R2), LinearRegression comes out slightly ahead. By average absolute error (MAE) or typical/median error (MdAPE, robust to outliers), XGBoost is lowest. RandomForest(max_depth=5) is the weakest of the four models overall.

**MdAPE (%) by price quintile** — lower is better

| Price band | LinearRegression | RandomForest | XGBoost | LightGBM |
|---|---:|---:|---:|---:|
| $11,899 – $580,000 | 66.01 | 127.11 | 75.86 | 82.12 |
| $580,000 – $807,500 | 42.13 | 49.89 | 32.22 | 32.56 |
| $807,500 – $1,125,000 | 31.44 | 17.28 | 21.94 | 20.98 |
| $1,125,000 – $1,690,000 | 27.74 | 20.27 | 20.99 | 20.50 |
| $1,690,000 – $97,972,500 | 25.51 | 36.73 | 28.13 | 28.67 |

- All four models have their **highest error rate (MdAPE) in the cheapest price quintile** (RandomForest spikes to 127%). A fixed dollar error is proportionally larger relative to a cheaper home's price.
- Accuracy is generally best in the mid-range ($807K–$1.69M) and ticks back up slightly in the top price band (≥$1.69M) — no model is uniformly ("flat") accurate across the entire price range.
- `RMSE >> MAE` holds across every price band for every model, meaning a small number of severely mispriced properties are present across essentially the whole price range, consistent with what was observed in notebooks 04/05.
- R2 computed within a narrow price band is heavily distorted (all negative in the lower bands) because variance shrinks within the band; it was treated as reference-only. MAPE/MdAPE are more suitable for comparing across bands.

**Summary**: for this dataset and time window — (1) LinearRegression has the highest raw explanatory power (R2), (2) XGBoost (tuned) has the lowest absolute/typical error (MAE, MdAPE), and (3) RandomForest(max_depth=5) is the weakest performer overall. Predictions are consistently hardest for the cheapest homes (in percentage terms) and for the most expensive homes (in dollar/RMSE terms).

---

## 4. How to re-run

### 4.1 Required packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm geopandas shapely openpyxl jupyter
```

### 4.2 Getting the raw data

The raw CRMLS sold-listing data is not included in this repository due to licensing restrictions. To run `notebooks/01_exploration.ipynb`:

1. Place monthly files named `CRMLSSold{YYYYMM}.csv` (e.g. `CRMLSSold202505.csv`) in a `data/raw/` folder (create it if it doesn't exist yet), or wherever you keep raw source files.
2. Update the file-loading path/glob in `notebooks/01_exploration.ipynb` (it currently looks for `CRMLSSold*.csv` in the notebook's own working directory) to point at wherever you placed the files, and adjust the `start`/`end` variables (currently `"202505"` to `"202605"`) to match the months you have.

If you don't have the raw data, you can still reproduce everything from `notebooks/03_baseline_model.ipynb` onward using the included `data/processed/cleaned_preprocessed_data.csv`.

### 4.3 Execution order

| # | Notebook | Input | Output |
|---|---|---|---|
| 1 | `notebooks/01_exploration.ipynb` | `CRMLSSold*.csv`  | `data/processed/merged_sales_data.csv` |
| 2 | `notebooks/02_preprocessing.ipynb` | `data/processed/merged_sales_data.csv` | `data/processed/cleaned_preprocessed_data.csv`, `data/processed/cleaned_preprocessed_data_old.csv`, `data/processed/feature_engineering_comparison.csv`, `data/processed/window_selection_results.csv` |
| 3 | `notebooks/03_baseline_model.ipynb` | `data/processed/cleaned_preprocessed_data.csv` | `data/processed/baseline_model_results.csv` |
| 4 | `notebooks/04_model_comparison.ipynb` | `data/processed/cleaned_preprocessed_data.csv`, `data/processed/baseline_model_results.csv` | `data/processed/model_comparison_results.csv` |
| 5 | `notebooks/05_advanced_models.ipynb` | `data/processed/cleaned_preprocessed_data.csv` | `data/processed/advanced_model_tuning_results.csv`, `data/processed/advanced_model_results.csv` |
| 6 | `notebooks/06_evaluation.ipynb` | `data/processed/cleaned_preprocessed_data.csv` | `data/processed/metrics_summary.csv` |

Run the notebooks top-to-bottom in order. Notebooks 3–6 can each be reproduced independently as long as `data/processed/cleaned_preprocessed_data.csv` is present (notebook 4 additionally loads `data/processed/baseline_model_results.csv` produced by notebook 3).

> Each notebook currently reads/writes CSVs by filename only (relative to its own working directory), so if you run them directly from inside `notebooks/`, either move/symlink the relevant CSVs into `notebooks/` first, or update the read/write paths in each notebook to point at `../data/processed/...`.

### 4.4 Note on the school-district spatial join (`notebooks/02_preprocessing.ipynb`, Section 9.1)

Building the `SchoolDistrictGIS` feature automatically downloads the [CA School District Areas 2024-25 GeoJSON](https://data.ca.gov/dataset/california-school-district-areas-2024-25) from a live URL. If that URL is not reachable from your environment, manually download the GeoJSON and save it as `CA_School_District_Areas_2024-25.geojson` in the same working directory the notebook is run from; the notebook will fall back to the local file automatically.

---

## 5. Repository structure

```
.
├── notebooks/
│   ├── 01_exploration.ipynb          # data merging & exploratory analysis (EDA)
│   ├── 02_preprocessing.ipynb        # missing-value/leakage handling, feature engineering, train/test split
│   ├── 03_baseline_model.ipynb       # LinearRegression baseline
│   ├── 04_model_comparison.ipynb     # DecisionTree / RandomForest comparison
│   ├── 05_advanced_models.ipynb      # XGBoost / LightGBM tuning and comparison
│   └── 06_evaluation.ipynb           # MAPE, MdAPE, and price-band evaluation
├── data/
│   └── processed/
│       ├── columns.xlsx                  # data dictionary: column descriptions, types, drop status
│       ├── cleaned_preprocessed_data.csv # final preprocessing output (input to notebooks 03-06)
│       └── advanced_model_results.csv    # final comparison results from notebook 05
└── README.md
```

> Note: raw source data (`CRMLSSold*.csv`) is not part of this repository (see Section 4.2) and is not shown above — if you add it locally, `data/raw/` is a reasonable place for it.
