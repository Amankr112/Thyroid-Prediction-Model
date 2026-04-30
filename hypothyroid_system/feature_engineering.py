"""
feature_engineering.py
-----------------------
Creates domain-specific features for hypothyroid prediction.
All transformations are deterministic (no fitting required).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add medically meaningful engineered features.

    Parameters
    ----------
    df : Preprocessed feature DataFrame (output of preprocessing.preprocess).

    Returns
    -------
    DataFrame with additional columns.
    """
    df = df.copy()

    # ── 1. Hormone ratios ────────────────────────────────────────────────────
    # T3/TT4 ratio — reflects peripheral conversion of thyroid hormones
    if "T3" in df.columns and "TT4" in df.columns:
        df["ratio_T3_TT4"] = df["T3"] / (df["TT4"] + 1e-6)

    # TSH/TT4 ratio — elevated TSH with low TT4 → hypothyroid signal
    if "TSH" in df.columns and "TT4" in df.columns:
        df["ratio_TSH_TT4"] = df["TSH"] / (df["TT4"] + 1e-6)

    # FTI/T4U ratio — free thyroxine index relative to T4 uptake
    if "FTI" in df.columns and "T4U" in df.columns:
        df["ratio_FTI_T4U"] = df["FTI"] / (df["T4U"] + 1e-6)

    logger.info("Hormone ratios created.")

    # ── 2. Binary interaction features ───────────────────────────────────────
    # on_thyroxine AND sick → already on treatment but still unwell
    if "on thyroxine" in df.columns and "sick" in df.columns:
        df["interact_thyroxine_sick"] = (
            df["on thyroxine"].fillna(0) * df["sick"].fillna(0)
        ).astype(int)

    # pregnant AND thyroid_surgery → elevated risk combination
    if "pregnant" in df.columns and "thyroid surgery" in df.columns:
        df["interact_pregnant_surgery"] = (
            df["pregnant"].fillna(0) * df["thyroid surgery"].fillna(0)
        ).astype(int)

    logger.info("Interaction features created.")

    # ── 3. Age groups ─────────────────────────────────────────────────────────
    if "age" in df.columns:
        bins = [0, 30, 60, np.inf]
        labels = ["young", "adult", "senior"]
        age_group = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        age_dummies = pd.get_dummies(age_group, prefix="age_grp")
        df = pd.concat([df, age_dummies], axis=1)

    logger.info("Age group dummies created.")

    # ── 4. Measurement reliability flags ─────────────────────────────────────
    # If a hormone was NOT measured, zero out its value (unreliable reading)
    measure_pairs = [
        ("TSH measured", "TSH"),
        ("T3 measured", "T3"),
        ("TT4 measured", "TT4"),
        ("T4U measured", "T4U"),
        ("FTI measured", "FTI"),
    ]
    for flag_col, val_col in measure_pairs:
        if flag_col in df.columns and val_col in df.columns:
            df[val_col] = df[val_col] * df[flag_col].fillna(0)

    logger.info("Measurement reliability applied.")
    logger.info("Feature engineering complete. Shape: %s", df.shape)

    return df
