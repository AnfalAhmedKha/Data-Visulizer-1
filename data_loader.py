"""
data_loader.py
==============
Handles all dataset ingestion: CSV, Excel, JSON, SQL.
Returns a validated pandas DataFrame with load metadata.
"""

import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    logger.info(f"CSV loaded: {path}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_excel(path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    logger.info(f"Excel loaded: {path} (sheet={sheet_name})  →  {df.shape}")
    return df


def load_json(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_json(path, **kwargs)
    logger.info(f"JSON loaded: {path}  →  {df.shape}")
    return df


def load_sql(query: str, connection_string: str) -> pd.DataFrame:
    from sqlalchemy import create_engine
    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine)
    logger.info(f"SQL loaded: {len(df):,} rows")
    return df


def load_file(path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Auto-detect file type and load into DataFrame.
    Returns (df, metadata_dict).
    """
    ext = Path(path).suffix.lower()
    loaders = {
        ".csv":     load_csv,
        ".xlsx":    load_excel,
        ".xls":     load_excel,
        ".json":    load_json,
        ".parquet": lambda p: pd.read_parquet(p),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

    df = loaders[ext](path)
    meta = {
        "file":      os.path.basename(path),
        "ext":       ext,
        "rows":      df.shape[0],
        "cols":      df.shape[1],
        "columns":   df.columns.tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "loaded_at": datetime.now().isoformat(),
    }
    return df, meta
