"""
preprocessing.py — Data Cleaning, Interpolation & Feature Engineering
Groundwater Level Prediction System
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
from loguru import logger


# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = ["recorded_at", "depth_to_water_m"]


# ── Loading & Validation ──────────────────────────────────────────────────────
def load_readings_to_df(readings: list[dict]) -> pd.DataFrame:
    """Convert list of reading dicts (from DB) to a clean DataFrame."""
    df = pd.DataFrame(readings)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Assert required columns exist and remove invalid rows."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[df["depth_to_water_m"] >= 0]          # Negative depth is invalid
    df = df[df["depth_to_water_m"] <= 200]         # Physical upper bound (metres)
    after = len(df)
    logger.info(f"Validation: dropped {before - after} invalid rows.")
    return df


# ── Outlier Removal ───────────────────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, col: str = "depth_to_water_m",
                         multiplier: float = 2.5) -> pd.DataFrame:
    """Remove outliers beyond multiplier × IQR."""
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[col] >= Q1 - multiplier * IQR) & (df[col] <= Q3 + multiplier * IQR)
    removed = (~mask).sum()
    logger.info(f"Outlier removal: {removed} rows flagged beyond {multiplier}×IQR.")
    return df[mask].copy()


# ── Resampling & Interpolation ────────────────────────────────────────────────
def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample irregular time-series to daily frequency.
    Gaps ≤ 7 days → linear interpolation.
    Gaps > 7 days → forward-fill (then flagged as interpolated).
    """
    df = df.set_index("recorded_at")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_daily = df[numeric_cols].resample("D").mean()

    # Linear interpolation for small gaps
    df_daily = df_daily.interpolate(method="time", limit=7)
    # Forward-fill remaining
    df_daily = df_daily.ffill()

    df_daily = df_daily.reset_index()
    df_daily.rename(columns={"recorded_at": "date"}, inplace=True)
    logger.info(f"Resampled to {len(df_daily)} daily records.")
    return df_daily


# ── Feature Engineering ───────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar and lag features used by ML models."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_year"] = df[date_col].dt.dayofyear
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year

    # Cyclical encoding (sine/cosine) to preserve periodicity
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Rolling statistics
    target = "depth_to_water_m"
    df["rolling_7d_mean"] = df[target].rolling(7, min_periods=1).mean()
    df["rolling_30d_mean"] = df[target].rolling(30, min_periods=1).mean()
    df["rolling_7d_std"] = df[target].rolling(7, min_periods=1).std().fillna(0)

    # Lag features
    for lag in [1, 7, 30]:
        df[f"lag_{lag}d"] = df[target].shift(lag)

    df = df.dropna()
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────
def scale_series(
    series: np.ndarray,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale a 1D numpy array to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return scaled, scaler


# ── LSTM Sequence Builder ─────────────────────────────────────────────────────
def create_sequences(
    data: np.ndarray,
    seq_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) pairs for LSTM training.
    X.shape = (n_samples, seq_length, 1)
    y.shape = (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length + horizon - 1])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def full_pipeline(
    readings: list[dict],
    seq_length: int = 30,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    End-to-end preprocessing: raw readings → LSTM-ready tensors.
    Returns (enriched_df, X_train, y_train, scaler)
    """
    df = load_readings_to_df(readings)
    df = validate_dataframe(df)
    df = remove_outliers_iqr(df)
    df = resample_to_daily(df)
    df = add_temporal_features(df, date_col="date")

    series = df["depth_to_water_m"].values
    scaled, scaler = scale_series(series)
    X, y = create_sequences(scaled, seq_length)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    return df, X_train, y_train, scaler
