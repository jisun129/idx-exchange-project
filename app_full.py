"""
app_full.py

Full-feature Prediction App (Streamlit). Loads the pre-trained pipeline from models/
(built by src/preprocess.py + src/train_model.py) and predicts a sale price from all 28
model features.

This app never retrains anything - it only loads models/price_model.pkl and the lookup
JSON files next to it, which are committed to the repo. Run src/preprocess.py and
src/train_model.py only when you want to refresh those files (e.g. with new raw data).

Run with:
    pip install -r requirements.txt
    streamlit run app_full.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.transformers import bool_to_str, sanitize_columns  # noqa: F401,E402
# ^ unused directly, but must be importable as `src.transformers` for joblib.load() below to
# unpickle the FunctionTransformer steps saved inside models/price_model.pkl by train_model.py

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "price_model.pkl"
DEFAULTS_PATH = MODEL_DIR / "feature_defaults.json"
DISTRICTS_PATH = MODEL_DIR / "school_district_options.json"
LEVELS_PATH = MODEL_DIR / "levels_options.json"

st.set_page_config(page_title="Home Price Predictor (Full)", page_icon="\U0001F3E1", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(DEFAULTS_PATH) as f:
        defaults = json.load(f)
    with open(DISTRICTS_PATH) as f:
        districts = json.load(f)
    with open(LEVELS_PATH) as f:
        levels = json.load(f)
    return model, defaults, districts, levels


st.title("\U0001F3E1 California Home Price Predictor (Full Model)")
st.write(
    "Enter as many details as you know about the property. Fields you leave at their "
    "default (pre-filled from the training data's median/most-common value) won't hurt "
    "accuracy much, but filling in what you actually know will help."
)

try:
    model, defaults, districts, levels_options = load_assets()
except FileNotFoundError as e:
    st.error(
        f"Missing file: `{e.filename}`. Run `python src/preprocess.py` and then "
        "`python src/train_model.py` first to generate the model and its lookup files "
        "(see the project README) - or check that the `models/` folder was included when "
        "this repo was cloned."
    )
    st.stop()

bool_options = {"Yes": True, "No": False}


def bool_default(key):
    return "Yes" if str(defaults.get(key)).lower() in ("true", "1", "1.0") else "No"


st.subheader("Basic Info")
c1, c2 = st.columns(2)
with c1:
    living_area = st.number_input("Living Area (sq ft)", min_value=100, max_value=20000,
                                   value=int(defaults["LivingArea"]), step=50)
    beds = st.number_input("Bedrooms (BedroomsTotal)", min_value=0, max_value=15,
                            value=int(defaults["BedroomsTotal"]), step=1)
    baths = st.number_input("Bathrooms (BathroomsTotalInteger)", min_value=0, max_value=15,
                             value=int(defaults["BathroomsTotalInteger"]), step=1)
    main_level_beds = st.number_input("Main-Level Bedrooms", min_value=0, max_value=15,
                                       value=int(defaults["MainLevelBedrooms"]), step=1)
with c2:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026,
                                  value=int(defaults["YearBuilt"]), step=1)
    stories = st.number_input("Stories", min_value=1, max_value=10,
                               value=int(defaults["Stories"]), step=1)
    days_on_market = st.number_input("Days on Market", min_value=0, max_value=1000,
                                      value=int(defaults["DaysOnMarket"]), step=1)

st.divider()
st.subheader("Lot & Structure")
c1, c2 = st.columns(2)
with c1:
    lot_size_sqft = st.number_input("Lot Size (sq ft)", min_value=0, max_value=500000,
                                     value=int(defaults["LotSizeSquareFeet"]), step=100)
    lot_size_area = st.number_input("Lot Area (LotSizeArea)", min_value=0, max_value=500000,
                                     value=int(defaults["LotSizeArea"]), step=100)
    lot_size_acres = st.number_input("Lot Size (acres)", min_value=0.0, max_value=100.0,
                                      value=float(defaults["LotSizeAcres"]), step=0.01,
                                      format="%.4f")
with c2:
    garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=10,
                                     value=int(defaults["GarageSpaces"]), step=1)
    parking_total = st.number_input("Total Parking Spaces", min_value=0, max_value=20,
                                     value=int(defaults["ParkingTotal"]), step=1)
    levels = st.selectbox("Levels", options=levels_options,
                           index=levels_options.index(defaults["Levels"])
                           if defaults["Levels"] in levels_options else 0)

st.divider()
st.subheader("Amenities")
c1, c2 = st.columns(2)
with c1:
    view = st.selectbox("Has a View?", options=list(bool_options),
                         index=list(bool_options).index(bool_default("ViewYN")))
    pool = st.selectbox("Private Pool?", options=list(bool_options),
                         index=list(bool_options).index(bool_default("PoolPrivateYN")))
    attached_garage = st.selectbox("Attached Garage?", options=list(bool_options),
                                    index=list(bool_options).index(bool_default("AttachedGarageYN")))
with c2:
    fireplace = st.selectbox("Has a Fireplace?", options=list(bool_options),
                              index=list(bool_options).index(bool_default("FireplaceYN")))
    new_construction = st.selectbox("New Construction?", options=list(bool_options),
                                     index=list(bool_options).index(bool_default("NewConstructionYN")))
    association_fee = st.number_input("HOA Fee ($/month, 0 = none)", min_value=0.0,
                                       max_value=5000.0, value=float(defaults["AssociationFee"]),
                                       step=10.0)

st.divider()
st.subheader("Location & Quality")
c1, c2 = st.columns(2)
with c1:
    state = st.text_input("State", value=defaults["StateOrProvince"])
    school_district = st.selectbox(
        "School District (GIS-assigned)", options=districts,
        index=districts.index(defaults["SchoolDistrictGIS"])
        if defaults["SchoolDistrictGIS"] in districts else 0,
        help="Assigned via spatial join against CA School District boundaries during "
             "training. Pick the closest match, or leave the default.",
    )
with c2:
    grade = st.selectbox(
        "Flooring Grade", options=["Likely Premium", "Mixed/Mid-range", "Likely Budget", "Unknown"],
        index=["Likely Premium", "Mixed/Mid-range", "Likely Budget", "Unknown"].index(defaults["grade"])
        if defaults["grade"] in ["Likely Premium", "Mixed/Mid-range", "Likely Budget", "Unknown"] else 1,
    )
    premium_score = st.slider(
        "Flooring Premium Score (0 = budget, 1 = premium)", min_value=0.0, max_value=1.0,
        value=float(defaults["premium_score"]), step=0.05,
        help="Derived from flooring material mix (Wood/Stone/Bamboo = premium; "
             "Carpet/Vinyl/Laminate/Concrete = budget).",
    )

st.divider()
st.subheader("Market Context")
st.caption(
    "These reflect recent nearby ($/sqft) sale activity and are normally computed "
    "automatically from historical sales near the property. Leave at the defaults "
    "(overall training-data medians) unless you have a specific local estimate."
)
c1, c2 = st.columns(2)
with c1:
    area_comps_median = st.number_input(
        "Nearby Median $/sqft (trailing 3 months)", min_value=0.0, max_value=5000.0,
        value=float(defaults["AreaCompsMedianPricePerSqFt"]), step=10.0)
    area_comps_mean = st.number_input(
        "Nearby Mean $/sqft (trailing 3 months)", min_value=0.0, max_value=5000.0,
        value=float(defaults["AreaCompsMeanPricePerSqFt"]), step=10.0)
with c2:
    area_comps_count = st.number_input(
        "Nearby Sales Count (trailing 3 months)", min_value=0, max_value=2000,
        value=int(defaults["AreaCompsSalesCount"]), step=1)

st.divider()

# BedBathRatio / PropertyAge are derived, but are literally two of the 28 model features -
# computed here from the inputs above rather than asked for directly.
bed_bath_ratio = beds / baths if baths else np.nan
property_age = 2026 - year_built

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "LivingArea": living_area,
        "DaysOnMarket": days_on_market,
        "ParkingTotal": parking_total,
        "LotSizeAcres": lot_size_acres,
        "YearBuilt": year_built,
        "BathroomsTotalInteger": baths,
        "BedroomsTotal": beds,
        "Stories": stories,
        "LotSizeArea": lot_size_area,
        "MainLevelBedrooms": main_level_beds,
        "GarageSpaces": garage_spaces,
        "AssociationFee": association_fee,
        "LotSizeSquareFeet": lot_size_sqft,
        "premium_score": premium_score,
        "BedBathRatio": bed_bath_ratio,
        "PropertyAge": property_age,
        "AreaCompsMedianPricePerSqFt": area_comps_median,
        "AreaCompsMeanPricePerSqFt": area_comps_mean,
        "AreaCompsSalesCount": area_comps_count,
        "ViewYN": bool_options[view],
        "PoolPrivateYN": bool_options[pool],
        "AttachedGarageYN": bool_options[attached_garage],
        "StateOrProvince": state,
        "FireplaceYN": bool_options[fireplace],
        "Levels": levels,
        "NewConstructionYN": bool_options[new_construction],
        "grade": grade,
        "SchoolDistrictGIS": school_district,
    }])

    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"### Estimated Price: ${pred_price:,.0f}")
    st.caption(
        "Predicted with the full 28-feature LightGBM pipeline (same hyperparameters found "
        "via walk-forward CV in notebooks/06_evaluation.ipynb). `BedBathRatio` and "
        "`PropertyAge` were computed automatically from your Beds/Baths/Year Built inputs above."
    )
