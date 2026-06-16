"""
cleaning.py
===========
Modular preprocessing: null handling, type conversion,
encoding, scaling, outlier removal, duplicate dropping.
Every function is pure (returns a new df) so steps are composable.
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Validation ─────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> dict:
    """Full quality report: nulls, dupes, dtypes, outliers."""
    num = df.select_dtypes(include=np.number)
    report = {
        "rows":          len(df),
        "cols":          len(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_by_col": df.isnull().sum().to_dict(),
        "missing_pct":   (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes":        {c: str(t) for c, t in df.dtypes.items()},
        "outliers_iqr":  {},
        "constant_cols": [c for c in df.columns if df[c].nunique() <= 1],
        "high_cardinality": [c for c in df.select_dtypes("object").columns
                             if df[c].nunique() / max(len(df), 1) > 0.9],
    }
    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((num[col] < q1 - 1.5 * iqr) | (num[col] > q3 + 1.5 * iqr)).sum())
        if n > 0:
            report["outliers_iqr"][col] = n

    logger.info(f"Validation: {report['missing_total']} nulls, "
                f"{report['duplicate_rows']} dupes, "
                f"{len(report['outliers_iqr'])} cols with outliers")
    return report


# ── Null handling ───────────────────────────────────────────────────────────
def drop_high_null_cols(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Drop columns where null fraction > threshold."""
    bad = [c for c in df.columns if df[c].isnull().mean() > threshold]
    if bad:
        logger.info(f"Dropping {len(bad)} high-null columns: {bad}")
    return df.drop(columns=bad)


def impute_nulls(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    strategy: 'auto' | 'median' | 'mean' | 'mode' | 'zero' | 'ffill'
    'auto' → median for numeric, mode for categorical
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if strategy in ("auto", "median"):
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == "mean":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == "zero":
        df[num_cols] = df[num_cols].fillna(0)
    elif strategy == "ffill":
        df = df.ffill().bfill()

    # Categorical always filled with mode
    for c in cat_cols:
        mode = df[c].mode()
        if not mode.empty:
            df[c] = df[c].fillna(mode[0])

    remaining = int(df.isnull().sum().sum())
    logger.info(f"Imputation ({strategy}): {remaining} nulls remaining")
    return df


# ── Duplicates ──────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    logger.info(f"Duplicates removed: {before - len(df):,} rows dropped")
    return df


# ── Type conversion ─────────────────────────────────────────────────────────
def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-convert obvious numerics stored as strings, parse date cols."""
    df = df.copy()
    for col in df.select_dtypes("object").columns:
        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.8:
            df[col] = converted
            logger.info(f"  '{col}' → numeric")
            continue
        # Try datetime
        if any(kw in col.lower() for kw in ("date", "time", "dt", "year", "month")):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"  '{col}' → datetime")
            except Exception:
                pass
    # Downcast to save memory
    for col in df.select_dtypes("int").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case, lowercase, strip special chars."""
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
                  for c in df.columns]
    return df


# ── Encoding ────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame,
                         max_cardinality: int = 10,
                         drop_first: bool = True,
                         exclude: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode low-cardinality object/category columns.
    `exclude` (e.g. a modeling target) is never encoded or dropped, so the
    caller can still find that column in the returned DataFrame.
    Returns (encoded_df, list_of_encoded_cols).
    """
    exclude = set(exclude or [])
    cat_cols = [c for c in df.select_dtypes(["object", "category"]).columns
                if c not in exclude and df[c].nunique() <= max_cardinality]
    if not cat_cols:
        return df, []
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first, dtype=int)
    logger.info(f"One-hot encoded {len(cat_cols)} columns: {cat_cols}")
    return df, cat_cols


# ── Scaling ─────────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   method: str = "standard",
                   exclude: List[str] = None) -> Tuple[pd.DataFrame, object]:
    """
    method: 'standard' (z-score) | 'minmax' | 'robust'
    Returns (scaled_df, fitted_scaler).
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    scaler_cls = scalers.get(method, StandardScaler)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if exclude:
        num_cols = [c for c in num_cols if c not in exclude]
    scaler = scaler_cls()
    df = df.copy()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled {len(num_cols)} numeric cols using {method}")
    return df, scaler


# ── Outlier handling ────────────────────────────────────────────────────────
def cap_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Winsorize outliers to IQR fences (cap, not drop).
    Skips binary/flag columns (nunique <= 2) and columns with IQR == 0,
    since capping those would collapse all values to a single number
    (e.g. a 0/1 'returned' flag where >75% of rows are 0).
    """
    df = df.copy()
    skipped = []
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique(dropna=True) <= 2:
            skipped.append(col)
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            skipped.append(col)
            continue
        df[col] = df[col].clip(q1 - factor * iqr, q3 + factor * iqr)
    if skipped:
        logger.info(f"Outlier capping skipped for binary/zero-IQR columns: {skipped}")
    return df


# ── Full pipeline ───────────────────────────────────────────────────────────
def auto_clean(df: pd.DataFrame,
               null_strategy: str = "auto",
               null_threshold: float = 0.6,
               cap_outliers: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    End-to-end cleaning in one call.
    Returns (clean_df, cleaning_report).
    """
    report = {"steps": []}
    original_shape = df.shape

    df = standardize_column_names(df);       report["steps"].append("column_names_standardized")
    df = drop_high_null_cols(df, null_threshold); report["steps"].append(f"dropped_cols_null>{null_threshold}")
    df = remove_duplicates(df);              report["steps"].append("duplicates_removed")
    df = infer_and_convert_types(df);        report["steps"].append("types_inferred")
    df = impute_nulls(df, null_strategy);    report["steps"].append(f"nulls_imputed_{null_strategy}")
    if cap_outliers:
        df = cap_outliers_iqr(df);           report["steps"].append("outliers_capped_iqr")

    report["shape_before"] = original_shape
    report["shape_after"]  = df.shape
    report["rows_removed"] = original_shape[0] - df.shape[0]
    report["cols_removed"] = original_shape[1] - df.shape[1]
    logger.info(f"auto_clean complete: {original_shape} → {df.shape}")
    return df, report
