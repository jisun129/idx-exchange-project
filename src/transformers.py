"""
src/transformers.py

Helper functions used inside the sklearn Pipeline saved to models/price_model.pkl.

These live in their own module - not inline in train_model.py - because joblib/pickle
stores a *reference* to each function (its module path + name), not its code. app_full.py
loads the pickled pipeline from a different file than the one that created it, so whatever
module these functions live in must be importable from both train_model.py (in src/) and
app_full.py (at the project root) using the exact same import path - `src.transformers`.
Both files add the project root to sys.path before importing this module for that reason.
"""

import numpy as np
import pandas as pd


def bool_to_str(X: pd.DataFrame) -> pd.DataFrame:
    """SimpleImputer/OneHotEncoder don't reliably handle pandas bool-dtype columns - and
    pd.read_csv infers bool dtype for YN columns (ViewYN, PoolPrivateYN, ...) whenever they
    have no missing values. Convert any bool values to 'True'/'False' strings first, leaving
    real NaN and any already-string values untouched."""
    def convert(v):
        if isinstance(v, (bool, np.bool_)):
            return "True" if v else "False"
        return v

    X = X.copy()
    for col in X.columns:
        X[col] = X[col].map(convert)
    return X


def sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM rejects feature names containing characters like , : { } [ ] " - which can
    show up in one-hot column names built from category text (e.g. district names). This
    replaces anything that isn't alphanumeric/underscore with '_' before the model sees it."""
    import re
    X = X.copy()
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in X.columns]
    return X
