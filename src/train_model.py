"""
src/train_model.py

Fits the final LightGBM pipeline - the best hyperparameters found via walk-forward CV in
notebooks/06_evaluation.ipynb - on the full model-ready dataset produced by preprocess.py,
and saves it, plus the lookup files app_full.py needs for its input widgets, to models/.

This is a development/maintenance script, run only when you want to refresh the deployed
model (e.g. after adding a new month of raw data and re-running preprocess.py). The app
itself never calls this - it just loads the saved models/price_model.pkl directly.

The hyperparameters below are fixed constants, not re-tuned here. If you want to re-tune
them (e.g. after a meaningfully larger or different raw dataset), re-run the walk-forward
CV in notebooks/06_evaluation.ipynb and update LGBM_BEST_PARAMS accordingly.

Usage:
    python src/train_model.py
    python src/train_model.py --input data/processed/model_ready_data.csv --model-dir models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `src.transformers` resolves the same way here
                                        # and in app_full.py, regardless of CWD - required
                                        # for joblib to unpickle the saved pipeline correctly
from src.transformers import bool_to_str, sanitize_columns  # noqa: E402

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

# Best LightGBM hyperparameters found in notebooks/06_evaluation.ipynb (walk-forward CV,
# Section 6). XGBoost was statistically tied on the same evaluation; LightGBM was chosen
# for deployment for its slightly lower fold-to-fold RMSE variance during tuning.
LGBM_BEST_PARAMS = dict(
    max_depth=7, num_leaves=63, learning_rate=0.1, n_estimators=400, min_child_samples=10,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                    default=PROJECT_ROOT / "data" / "processed" / "model_ready_data.csv",
                    help="Output of preprocess.py (default: data/processed/model_ready_data.csv)")
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models",
                    help="Where to save price_model.pkl and the app's lookup JSON files "
                         "(default: models/)")
    return p.parse_args()


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("bool_to_str", FunctionTransformer(bool_to_str)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])
    preprocessor.set_output(transform="pandas")  # keep column names through the pipeline

    sanitizer = FunctionTransformer(sanitize_columns)
    model = LGBMRegressor(**LGBM_BEST_PARAMS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("sanitize_names", sanitizer),
        ("model", model),
    ])


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"{args.input} is missing column(s): {missing}. Did you run "
                        "preprocess.py first?")

    X = df[FEATURE_COLS]
    y_log = np.log1p(df[TARGET])

    pipeline = build_pipeline()
    pipeline.fit(X, y_log)

    model_path = args.model_dir / "price_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Saved {model_path}")

    # Lookup files app_full.py uses to prefill/build its 28 input widgets
    defaults = {
        col: (float(X[col].median()) if col in NUMERIC_COLS else str(X[col].mode().iloc[0]))
        for col in FEATURE_COLS
    }
    with open(args.model_dir / "feature_defaults.json", "w") as f:
        json.dump(defaults, f, indent=2)

    with open(args.model_dir / "school_district_options.json", "w") as f:
        json.dump(sorted(X["SchoolDistrictGIS"].dropna().unique().tolist()), f, indent=2)

    with open(args.model_dir / "levels_options.json", "w") as f:
        json.dump(sorted(X["Levels"].dropna().unique().tolist()), f, indent=2)

    print(f"Saved feature_defaults.json, school_district_options.json, "
          f"levels_options.json to {args.model_dir}")


if __name__ == "__main__":
    main()
