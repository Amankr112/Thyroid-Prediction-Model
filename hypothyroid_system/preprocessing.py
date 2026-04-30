"""
preprocessing.py
----------------
Handles all data cleaning, encoding, and imputation for the
hypothyroid classification dataset.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Column groups ──────────────────────────────────────────────────────────────
BOOL_COLS = [
    "on thyroxine", "query on thyroxine", "on antithyroid medication",
    "sick", "pregnant", "thyroid surgery", "I131 treatment",
    "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
    "tumor", "hypopituitary", "psych",
    "TSH measured", "T3 measured", "TT4 measured",
    "T4U measured", "FTI measured", "TBG measured",
]
NUMERICAL_COLS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
CATEGORICAL_COLS = ["sex", "referral source"]
DROP_COLS = ["TBG"]   # 100% missing
TARGET_COL = "binaryClass"


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts: dict = None):
    """
    Clean, encode, and impute the raw dataset.

    Parameters
    ----------
    df        : Raw DataFrame ('?' marks missing values).
    fit       : True  → fit encoders/imputers and store in artifacts.
                False → apply pre-fitted artifacts (inference).
    artifacts : Dict that stores/supplies fitted objects.

    Returns
    -------
    X, y, artifacts
    """
    if artifacts is None:
        artifacts = {}

    df = df.copy()
    logger.info("Preprocessing started. Input shape: %s", df.shape)

    # 1. Replace "?" ─────────────────────────────────────────────────────────
    df.replace("?", np.nan, inplace=True)

    # 2. Drop always-missing columns ─────────────────────────────────────────
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped '%s' (100%% missing).", col)

    # 3. Extract target ──────────────────────────────────────────────────────
    y = None
    if TARGET_COL in df.columns:
        # N = Hypothyroid (positive / minority), P = Normal
        y = (df[TARGET_COL].str.strip().str.upper() == "N").astype(int)
        df.drop(columns=[TARGET_COL], inplace=True)
        logger.info("Target: Hypothyroid (N)=1  Normal (P)=0. Counts: %s", y.value_counts().to_dict())

    # 4. Boolean t/f → 1/0 ──────────────────────────────────────────────────
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0})

    # 5. Numerical → float ───────────────────────────────────────────────────
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Categorical: mode-fill → OneHotEncode ───────────────────────────────
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_present:
        if fit:
            artifacts[f"cat_mode_{col}"] = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(artifacts.get(f"cat_mode_{col}", "unknown"))

    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(df[cat_present])
        artifacts["ohe"] = ohe
        artifacts["ohe_cols"] = cat_present
    else:
        ohe = artifacts["ohe"]
        cat_present = artifacts["ohe_cols"]

    ohe_df = pd.DataFrame(
        ohe.transform(df[cat_present]),
        columns=ohe.get_feature_names_out(cat_present),
        index=df.index,
    )
    df.drop(columns=cat_present, inplace=True)
    df = pd.concat([df, ohe_df], axis=1)

    # 7. Age median-fill ─────────────────────────────────────────────────────
    if "age" in df.columns:
        if fit:
            artifacts["age_median"] = df["age"].median()
        df["age"] = df["age"].fillna(artifacts["age_median"])

    # 8. KNN imputation for numerical columns ────────────────────────────────
    num_present = [c for c in NUMERICAL_COLS if c in df.columns]
    if fit:
        knn = KNNImputer(n_neighbors=5)
        df[num_present] = knn.fit_transform(df[num_present])
        artifacts["knn_imputer"] = knn
        artifacts["num_cols"] = num_present
    else:
        df[num_present] = artifacts["knn_imputer"].transform(df[num_present])

    # 9. Column alignment ────────────────────────────────────────────────────
    if fit:
        artifacts["feature_cols"] = df.columns.tolist()
    else:
        for col in artifacts["feature_cols"]:
            if col not in df.columns:
                df[col] = 0
        df = df[artifacts["feature_cols"]]

    logger.info("Preprocessing complete. Output shape: %s", df.shape)
    return df, y, artifacts


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded '%s'. Shape: %s", path, df.shape)
    return df
