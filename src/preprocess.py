"""
src/preprocess.py

Reproduces 01_exploration.ipynb + 02_preprocessing.ipynb as a single, path-independent
script: merges the raw monthly CRMLS CSVs, filters to single-family residential sales,
cleans missing/invalid target rows, handles outliers, and builds the full Week-6 feature
set (28 raw columns). See notebooks/01_exploration.ipynb and notebooks/02_preprocessing.ipynb
for the narrative, cell-by-cell walkthrough of this same logic.

Output (written to --output-dir, default data/processed/):
  - merged_sales_data.csv : raw rows merged across months, filtered to Residential /
    SingleFamilyResidence (intermediate artifact, kept for transparency/debugging)
  - model_ready_data.csv  : the 28 feature columns + ClosePrice, UNENCODED (median/mode
    imputation, scaling, and one-hot encoding happen later, inside the sklearn Pipeline
    fit by train_model.py - not here)

Paths default relative to this file's location, not the current working directory, so
this can be run from anywhere:

    python src/preprocess.py
    python src/preprocess.py --raw-dir /custom/raw --output-dir /custom/out
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET = "ClosePrice"
TOP_N_DISTRICTS = 30
N_MONTHS_COMPS = 3
SCHOOL_DISTRICT_GEOJSON_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)
CA_LAT_RANGE = (32.5, 42.1)
CA_LON_RANGE = (-124.5, -114.0)

PREMIUM_MATERIALS = {"Wood", "Stone", "Bamboo"}
BUDGET_MATERIALS = {"Carpet", "Vinyl", "Laminate", "Concrete"}

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw",
                    help="Folder containing raw CRMLSSold*.csv files (default: data/raw)")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed",
                    help="Folder to write merged_sales_data.csv and model_ready_data.csv "
                         "(default: data/processed)")
    p.add_argument("--school-district-geojson", type=Path, default=None,
                    help="Local CA school-district GeoJSON to use instead of downloading it")
    return p.parse_args()


def merge_raw_files(raw_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_dir / "CRMLSSold*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CRMLSSold*.csv files found in {raw_dir}. Place the raw CRMLS monthly "
            "sold-listing CSVs there, or pass --raw-dir to point elsewhere."
        )
    frames = []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        month_str = Path(fp).stem.replace("CRMLSSold", "")  # e.g. "202505"
        df["date"] = pd.to_datetime(month_str, format="%Y%m")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged["year_month"] = merged["date"].dt.to_period("M")

    merged = merged[
        (merged["PropertyType"] == "Residential")
        & (merged["PropertySubType"] == "SingleFamilyResidence")
    ].copy()
    return merged


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


def compute_iqr_bounds(series, k=1.5):
    log_s = np.log1p(series.dropna())
    q1, q3 = log_s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return np.expm1(q1 - k * iqr), np.expm1(q3 + k * iqr)


def add_school_district(df, local_geojson_path=None):
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

    districts = None
    if local_geojson_path and Path(local_geojson_path).exists():
        districts = gpd.read_file(local_geojson_path)
    else:
        try:
            districts = gpd.read_file(SCHOOL_DISTRICT_GEOJSON_URL)
        except Exception:
            print("Couldn't reach the CA school-district GeoJSON and no local file given "
                  "(--school-district-geojson) - SchoolDistrictGIS will be 'Unknown'.")
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
        postal_code = group.name  # pandas >=2.2 may exclude this column from `group` itself
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
        result = group.reset_index()
        result["PostalCode"] = postal_code
        return result

    stats = stats.groupby("PostalCode", group_keys=False).apply(add_trailing)
    df = df.merge(
        stats[["PostalCode", "year_month", "AreaCompsMedianPricePerSqFt",
               "AreaCompsMeanPricePerSqFt", "AreaCompsSalesCount"]],
        on=["PostalCode", "year_month"], how="left",
    )
    return df.drop(columns=["_PricePerSqFt"])


def build_features(df: pd.DataFrame, school_district_geojson: Path = None) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] > 0].copy()

    df["AssociationFee"] = df["AssociationFee"].fillna(0)

    df["premium_score"] = df["Flooring"].apply(premium_score_fn)
    df["grade"] = df["premium_score"].apply(grade_fn)

    # Outlier handling (log1p-space IQR, k=1.5) on the full dataset. Note: 02_preprocessing.ipynb
    # computes these bounds on the train window only, to avoid test-set leakage during
    # benchmarking; there's no such split here since this script's output feeds a single
    # production model, not a train/test comparison.
    price_lower, price_upper = compute_iqr_bounds(df[TARGET])
    df = df[(df[TARGET] >= price_lower) & (df[TARGET] <= price_upper)].copy()
    for col in ["LivingArea", "LotSizeSquareFeet"]:
        lower, upper = compute_iqr_bounds(df[col])
        df[col] = df[col].clip(lower=lower, upper=upper)

    df["BedBathRatio"] = df["BedroomsTotal"] / df["BathroomsTotalInteger"].replace(0, np.nan)
    df.loc[(df["BedroomsTotal"] < 0) | (df["BathroomsTotalInteger"] < 0), "BedBathRatio"] = np.nan

    df["PropertyAge"] = df["date"].dt.year - df["YearBuilt"]
    df.loc[(df["PropertyAge"] < 0) | (df["PropertyAge"] > 200), "PropertyAge"] = np.nan

    df = add_school_district(df, local_geojson_path=school_district_geojson)
    top_districts = df["SchoolDistrictGIS"].value_counts().head(TOP_N_DISTRICTS).index
    df["SchoolDistrictGIS"] = df["SchoolDistrictGIS"].where(
        df["SchoolDistrictGIS"].isin(top_districts) | df["SchoolDistrictGIS"].isna(), "Other"
    )

    df = add_area_comps(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected column(s) after feature engineering: {missing}")

    return df[FEATURE_COLS + [TARGET]]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw CSVs from {args.raw_dir} ...")
    merged = merge_raw_files(args.raw_dir)
    merged_path = args.output_dir / "merged_sales_data.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved {merged_path} ({len(merged):,} rows)")

    print("Building the 28-feature model-ready dataset ...")
    model_ready = build_features(merged, school_district_geojson=args.school_district_geojson)
    model_ready_path = args.output_dir / "model_ready_data.csv"
    model_ready.to_csv(model_ready_path, index=False)
    print(f"Saved {model_ready_path} ({len(model_ready):,} rows, "
          f"{len(FEATURE_COLS)} features + target)")


if __name__ == "__main__":
    main()
