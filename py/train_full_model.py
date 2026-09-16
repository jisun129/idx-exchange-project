"""
Trains the FULL-feature model (all 28 engineered/raw features used in the main pipeline,
per 02_preprocessing.ipynb Week 6) and saves it - plus a few small lookup files the app
needs for its input widgets - for app_full.py.

Mirrors 02_preprocessing.ipynb's feature engineering (Sections 3.1, 9, 9.1-9.3) and
06_evaluation.ipynb's best XGBoost hyperparameters (Section 6). Unlike the benchmark
notebooks, this trains on the FULL dataset (no held-out test month) since the point here
is to produce the model actually deployed in the app, not to measure test performance.

Run from the same folder as merged_sales_data.csv (i.e. after 01_exploration.ipynb):

    pip install pandas numpy scikit-learn xgboost joblib geopandas shapely
    python train_full_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET = "ClosePrice"

NUMERIC_COLS = [
    "LivingArea", "DaysOnMarket", "ParkingTotal", "LotSizeAcres", "YearBuilt",
    "BathroomsTotalInteger", "BedroomsTotal", "Stories", "LotSizeArea",
    "MainLevelBedrooms", "GarageSpaces", "AssociationFee", "LotSizeSquareFeet",
    "premium_score", "BedBathRatio", "PropertyAge",
    "AreaCompsMedianPricePerSqFt", "AreaCompsMeanPricePerSqFt", "AreaCompsSalesCount",
]
CATEGORICAL_COLS = [
    "ViewYN", "PoolPrivateYN", "AttachedGarageYN", "StateOrProvince", "FireplaceYN",
    "Levels", "NewConstructionYN", "grade", "SchoolDistrictGIS",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Best XGBoost hyperparameters found in 06_evaluation.ipynb (Section 6)
XGB_BEST_PARAMS = dict(max_depth=7, learning_rate=0.1, n_estimators=400,
                        min_child_weight=1, subsample=0.8)

TOP_N_DISTRICTS = 30
N_MONTHS_COMPS = 3
SCHOOL_DISTRICT_GEOJSON_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)
SCHOOL_DISTRICT_LOCAL_PATH = "CA_School_District_Areas_2024-25.geojson"
CA_LAT_RANGE = (32.5, 42.1)
CA_LON_RANGE = (-124.5, -114.0)

PREMIUM_MATERIALS = {"Wood", "Stone", "Bamboo"}
BUDGET_MATERIALS = {"Carpet", "Vinyl", "Laminate", "Concrete"}


def premium_score_fn(floor_str, full_confidence_at=3):
    if pd.isna(floor_str):
        return np.nan
    materials = set(floor_str.split(","))
    premium_count = len(materials & PREMIUM_MATERIALS)
    budget_count = len(materials & BUDGET_MATERIALS)
    judgeable = premium_count + budget_count
    if judgeable == 0:
        return 0.5
    raw_score = premium_count / judgeable
    confidence_weight = min(judgeable / full_confidence_at, 1.0)
    return raw_score * confidence_weight + 0.5 * (1 - confidence_weight)


def grade_fn(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 0.65:
        return "Likely Premium"
    if score <= 0.35:
        return "Likely Budget"
    return "Mixed/Mid-range"


def add_school_district(df):
    """Spatial join against CA School District Areas 2024-25. Falls back to 'Unknown' for
    every row if geopandas/shapely aren't installed, or the boundaries can't be loaded."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("geopandas/shapely not installed - SchoolDistrictGIS will be 'Unknown' for "
              "every row. Install with: pip install geopandas shapely")
        df["SchoolDistrictGIS"] = "Unknown"
        return df

    try:
        districts = gpd.read_file(SCHOOL_DISTRICT_GEOJSON_URL)
    except Exception:
        if os.path.exists(SCHOOL_DISTRICT_LOCAL_PATH):
            districts = gpd.read_file(SCHOOL_DISTRICT_LOCAL_PATH)
        else:
            print("Couldn't reach the CA school-district GeoJSON and no local fallback "
                  f"at {SCHOOL_DISTRICT_LOCAL_PATH} - SchoolDistrictGIS will be 'Unknown'.")
            df["SchoolDistrictGIS"] = "Unknown"
            return df

    name_candidates = ["DistrictName", "District_N", "NAME", "District"]
    district_name_col = next((c for c in name_candidates if c in districts.columns), None)
    if district_name_col is None:
        print(f"No district-name column found among {list(districts.columns)} - "
              "SchoolDistrictGIS will be 'Unknown'.")
        df["SchoolDistrictGIS"] = "Unknown"
        return df

    has_coords = df["Latitude"].between(*CA_LAT_RANGE) & df["Longitude"].between(*CA_LON_RANGE)
    points = gpd.GeoDataFrame(
        df.loc[has_coords, []],
        geometry=[Point(xy) for xy in zip(df.loc[has_coords, "Longitude"],
                                           df.loc[has_coords, "Latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, districts[[district_name_col, "geometry"]],
                        how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    df["SchoolDistrictGIS"] = "Unknown"
    df.loc[has_coords, "SchoolDistrictGIS"] = joined[district_name_col].values
    return df


def add_area_comps(df):
    """Trailing N-month, PostalCode-level $/sqft comps (causal: shift(1) before rolling)."""
    df["_PricePerSqFt"] = df[TARGET] / df["LivingArea"].replace(0, np.nan)
    stats = (
        df.groupby(["PostalCode", "year_month"])["_PricePerSqFt"]
        .agg(median="median", mean="mean", count="count")
        .reset_index()
        .sort_values(["PostalCode", "year_month"])
    )

    def add_trailing(group):
        group = group.set_index("year_month")
        group["AreaCompsMedianPricePerSqFt"] = (
            group["median"].shift(1).rolling(N_MONTHS_COMPS, min_periods=1).mean()
        )
        group["AreaCompsMeanPricePerSqFt"] = (
            group["mean"].shift(1).rolling(N_MONTHS_COMPS, min_periods=1).mean()
        )
        group["AreaCompsSalesCount"] = (
            group["count"].shift(1).rolling(N_MONTHS_COMPS, min_periods=1).sum()
        )
        return group.reset_index()

    stats = stats.groupby("PostalCode", group_keys=False).apply(add_trailing)
    df = df.merge(
        stats[["PostalCode", "year_month", "AreaCompsMedianPricePerSqFt",
               "AreaCompsMeanPricePerSqFt", "AreaCompsSalesCount"]],
        on=["PostalCode", "year_month"], how="left",
    )
    return df.drop(columns=["_PricePerSqFt"])


def compute_iqr_bounds(series, k=1.5):
    log_s = np.log1p(series.dropna())
    q1, q3 = log_s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return np.expm1(q1 - k * iqr), np.expm1(q3 + k * iqr)


def build_pipeline():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])
    model = XGBRegressor(**XGB_BEST_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    df = pd.read_csv("merged_sales_data.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] > 0].copy()

    df["AssociationFee"] = df["AssociationFee"].fillna(0)

    df["premium_score"] = df["Flooring"].apply(premium_score_fn)
    df["grade"] = df["premium_score"].apply(grade_fn)

    # Outlier handling (log1p-space IQR, k=1.5) on the full dataset - this is the final
    # deployed model, not a train/test benchmark
    price_lower, price_upper = compute_iqr_bounds(df[TARGET])
    df = df[(df[TARGET] >= price_lower) & (df[TARGET] <= price_upper)].copy()
    for col in ["LivingArea", "LotSizeSquareFeet"]:
        lower, upper = compute_iqr_bounds(df[col])
        df[col] = df[col].clip(lower=lower, upper=upper)

    df["BedBathRatio"] = df["BedroomsTotal"] / df["BathroomsTotalInteger"].replace(0, np.nan)
    df.loc[(df["BedroomsTotal"] < 0) | (df["BathroomsTotalInteger"] < 0), "BedBathRatio"] = np.nan

    df["PropertyAge"] = df["date"].dt.year - df["YearBuilt"]
    df.loc[(df["PropertyAge"] < 0) | (df["PropertyAge"] > 200), "PropertyAge"] = np.nan

    df = add_school_district(df)
    top_districts = df["SchoolDistrictGIS"].value_counts().head(TOP_N_DISTRICTS).index
    df["SchoolDistrictGIS"] = df["SchoolDistrictGIS"].where(
        df["SchoolDistrictGIS"].isin(top_districts) | df["SchoolDistrictGIS"].isna(), "Other"
    )

    df = add_area_comps(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected column(s): {missing}")

    X = df[FEATURE_COLS]
    y_log = np.log1p(df[TARGET])

    pipeline = build_pipeline()
    pipeline.fit(X, y_log)

    joblib.dump(pipeline, "full_price_model.pkl")
    print("Saved full_price_model.pkl")

    # Lookup files app_full.py uses to prefill/build its 28 input widgets
    defaults = {
        col: (float(X[col].median()) if col in NUMERIC_COLS else str(X[col].mode().iloc[0]))
        for col in FEATURE_COLS
    }
    with open("feature_defaults.json", "w") as f:
        json.dump(defaults, f, indent=2)
    print("Saved feature_defaults.json")

    with open("school_district_options.json", "w") as f:
        json.dump(sorted(X["SchoolDistrictGIS"].dropna().unique().tolist()), f, indent=2)
    print("Saved school_district_options.json")

    with open("levels_options.json", "w") as f:
        json.dump(sorted(X["Levels"].dropna().unique().tolist()), f, indent=2)
    print("Saved levels_options.json")


if __name__ == "__main__":
    main()
