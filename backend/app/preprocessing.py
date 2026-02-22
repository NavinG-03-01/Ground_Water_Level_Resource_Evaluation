"""
preprocessing.py — Adapted for CGWB dataset
Target column : currentlevel  (depth to water in metres)
Extra feature : level_diff    (change from previous reading)
Date format   : DD-MM-YYYY    (parsed to datetime before this stage)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
from loguru import logger


# ── Column names matching your CGWB dataset ───────────────────────────────────
TARGET_COL   = "depth_to_water_m"   # stored as this in DB (= currentlevel)
DATE_COL     = "recorded_at"        # stored as this in DB (parsed datetime)
DIFF_COL     = "level_diff"         # optional extra feature


# ── Load DB rows → DataFrame ──────────────────────────────────────────────────
def load_readings_to_df(readings: list) -> pd.DataFrame:
    """
    Convert list of dicts (from DB query) to DataFrame.
    Handles both 'recorded_at' (DB) and 'parsed_date' (raw CSV) keys.
    """
    df = pd.DataFrame(readings)

    # Normalise date column name
    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    elif "parsed_date" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["parsed_date"])

    # Normalise level column name
    if "currentlevel" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["currentlevel"]

    df = df.sort_values("recorded_at").reset_index(drop=True)
    return df


# ── Validation ────────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with:
    - missing date or water level
    - physically impossible levels (your data has range ~-5 to 300)
    """
    before = len(df)
    df = df.dropna(subset=["recorded_at", TARGET_COL])

    # CGWB currentlevel can be slightly negative (artesian wells) down to -5
    df = df[(df[TARGET_COL] >= -5) & (df[TARGET_COL] <= 300)]

    logger.info(f"Validation: {before - len(df)} invalid rows dropped, {len(df):,} remain.")
    return df.copy()


# ── Outlier removal ───────────────────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Uses 3.0× IQR (looser than default 2.5) because groundwater levels
    across India have genuinely high variance between seasons and states.
    Apply this AFTER grouping by well, not on the full 555k dataset.
    """
    Q1  = df[TARGET_COL].quantile(0.25)
    Q3  = df[TARGET_COL].quantile(0.75)
    IQR = Q3 - Q1
    mask    = df[TARGET_COL].between(Q1 - multiplier * IQR, Q3 + multiplier * IQR)
    removed = (~mask).sum()
    if removed:
        logger.info(f"Outliers removed: {removed} rows beyond {multiplier}×IQR.")
    return df[mask].copy()


# ── Resample irregular CGWB data to monthly ───────────────────────────────────
def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    CGWB data is typically measured twice a year (pre/post monsoon).
    Resample to MONTHLY ('MS') with linear interpolation for gaps ≤ 3 months.
    This gives enough points for ARIMA & LSTM without fabricating too much data.
    """
    df = df.set_index("recorded_at")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_monthly = df[numeric_cols].resample("MS").mean()           # month start
    df_monthly = df_monthly.interpolate(method="time", limit=3)   # fill ≤3 month gaps
    df_monthly = df_monthly.ffill().bfill()                        # fill edges

    df_monthly = df_monthly.reset_index()
    df_monthly = df_monthly.rename(columns={"recorded_at": "date"})
    logger.info(f"Resampled to {len(df_monthly)} monthly records.")
    return df_monthly


# ── Feature engineering ───────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar, cyclical, rolling, and lag features.
    Uses monthly cadence (matches resampled CGWB data).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["month"]      = df[date_col].dt.month
    df["quarter"]    = df[date_col].dt.quarter
    df["year"]       = df[date_col].dt.year
    df["month_idx"]  = (df["year"] - df["year"].min()) * 12 + df["month"]

    # Cyclical month encoding (captures Jan↔Dec adjacency)
    df["sin_month"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Monsoon flag (June–September)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Rolling statistics (3-month and 6-month windows)
    df["rolling_3m_mean"] = df[TARGET_COL].rolling(3,  min_periods=1).mean()
    df["rolling_6m_mean"] = df[TARGET_COL].rolling(6,  min_periods=1).mean()
    df["rolling_3m_std"]  = df[TARGET_COL].rolling(3,  min_periods=1).std().fillna(0)

    # Lag features — 1 month, 3 months, 6 months, 12 months
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}m"] = df[TARGET_COL].shift(lag)

    # level_diff as a feature if available
    if DIFF_COL in df.columns:
        df["rolling_diff_3m"] = df[DIFF_COL].rolling(3, min_periods=1).mean().fillna(0)

    df = df.dropna()
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────
def scale_series(series: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return scaled, scaler


# ── LSTM sequence builder ─────────────────────────────────────────────────────
def create_sequences(
    data: np.ndarray,
    seq_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window sequences for LSTM.
    With monthly data and seq_length=12: uses 12 months to predict next month.
    """
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length + horizon - 1])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


# ── Full pipeline ─────────────────────────────────────────────────────────────
def full_pipeline(
    readings: list,
    seq_length: int = 12,          # 12 months of history (monthly data)
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    End-to-end: raw DB readings → LSTM-ready tensors.
    Returns (enriched_df, X_train, y_train, scaler)
    """
    df = load_readings_to_df(readings)
    df = validate_dataframe(df)
    df = remove_outliers_iqr(df)
    df = resample_to_monthly(df)
    df = add_temporal_features(df, date_col="date")

    series        = df[TARGET_COL].values
    scaled, scaler = scale_series(series)
    X, y          = create_sequences(scaled, seq_length)

    split   = int(0.8 * len(X))
    X_train = X[:split]
    y_train = y[:split]

    return df, X_train, y_train, scaler


# ── Per-state / per-basin summary ─────────────────────────────────────────────
def summarise_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate mean water levels by state or basin.
    Useful for the analytics/regional endpoint.
    df must have columns: [group_col, 'depth_to_water_m', 'recorded_at']
    """
    summary = (
        df.groupby(group_col)[TARGET_COL]
        .agg(mean_level="mean", min_level="min", max_level="max",
             std_level="std", count="count")
        .reset_index()
        .sort_values("mean_level", ascending=False)
    )
    return summary