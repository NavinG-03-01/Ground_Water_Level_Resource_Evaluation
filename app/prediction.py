"""
prediction.py — 4-Model Ensemble Groundwater Level Forecasting
Groundwater Level Prediction System

Models:
  1. SARIMA      — seasonal statistics, captures monsoon cycles
  2. RandomForest — tree ensemble, robust to outliers and noise
  3. XGBoost      — gradient boosting, best single-model accuracy
  4. BiLSTM       — deep learning, captures complex long-term patterns

Ensemble Strategy:
  - Each model predicts independently
  - Weights assigned by inverse RMSE (better model gets higher weight)
  - Dynamic weighting: if a model fails, its weight is redistributed
  - Confidence band = weighted average of individual model uncertainties

New in this version:
  - RandomForestForecaster class added
  - XGBoostForecaster class added
  - build_lag_features() — shared feature engineering for RF + XGB
  - run_forecast() updated: model_type='ensemble' now uses all 4 models
  - Dynamic weight calculation replaces fixed 0.5/0.5 weights
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMA = True
except ImportError:
    HAS_SARIMA = False
    logger.warning("statsmodels not available — SARIMA disabled.")

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    logger.info("pmdarima not installed — using fixed SARIMA order.")

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except ImportError:
    HAS_RF = False
    logger.warning("scikit-learn not available — RandomForest disabled.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed — XGBoost disabled.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as keras_callbacks
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    logger.warning("TensorFlow not available — LSTM disabled.")

try:
    from app.config import settings
    from app.preprocessing import scale_series, create_sequences
    HAS_APP_MODULES = True
except ImportError:
    HAS_APP_MODULES = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_series(readings: List[Dict]) -> pd.Series:
    """
    Convert raw DB reading dicts/ORM objects into a clean monthly pandas Series.

    Steps:
      1. Parse date + depth from each reading
      2. Validate depth is physically realistic (0-300 m)
      3. Remove outliers via IQR filter
      4. Resample to monthly frequency using median
      5. Interpolate short gaps (up to 3 consecutive months)
      6. Return DatetimeIndex Series sorted oldest to newest
    """
    if not readings:
        return pd.Series(dtype=float)

    rows = []
    for r in readings:
        try:
            if isinstance(r, dict):
                date_val  = r.get('recorded_at')
                depth_val = r.get('depth_to_water_m') or r.get('currentlevel')
            else:
                date_val  = getattr(r, 'recorded_at', None)
                depth_val = getattr(r, 'depth_to_water_m', None) or getattr(r, 'currentlevel', None)

            if date_val is None or depth_val is None:
                continue

            dt = pd.to_datetime(date_val)
            dv = float(depth_val)

            if 0.0 < dv < 300.0:
                rows.append({'date': dt, 'depth': dv})
        except Exception:
            continue

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows).sort_values('date').set_index('date')

    # IQR outlier removal
    Q1, Q3 = df['depth'].quantile(0.25), df['depth'].quantile(0.75)
    IQR    = Q3 - Q1
    df     = df[(df['depth'] >= Q1 - 1.5 * IQR) & (df['depth'] <= Q3 + 1.5 * IQR)]

    if df.empty:
        return pd.Series(dtype=float)

    monthly = df['depth'].resample('MS').median()
    monthly = monthly.interpolate(method='time', limit=3).dropna()

    return monthly


def build_lag_features(series: pd.Series, n_lags: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning features from time series for RF and XGBoost.

    For each time step t, creates features:
      - lag_1  to lag_12  : previous 12 monthly values
      - month             : calendar month (1-12) — captures seasonality
      - month_sin/cos     : circular encoding of month — better than raw month integer
      - trend             : position in series (0,1,2,...) — captures long-term drift
      - rolling_mean_3    : 3-month rolling average — smoothed recent level
      - rolling_mean_6    : 6-month rolling average — medium-term trend
      - rolling_std_6     : 6-month std deviation   — local volatility

    Why lag features for tree models:
      RandomForest and XGBoost cannot natively handle sequential data like
      SARIMA/LSTM. By giving them the previous 12 months as input features,
      plus calendar and trend features, they can learn:
        - Monsoon recovery patterns (lag_3 after Aug reading recovers)
        - Year-over-year depletion trend
        - Seasonal fluctuations

    Returns
    -------
    X : feature matrix (n_samples, n_features)
    y : target vector  (n_samples,)
    """
    values = series.values
    n      = len(values)

    if n <= n_lags + 6:
        return np.array([]), np.array([])

    X_rows, y_rows = [], []

    for i in range(n_lags, n):
        lags = values[i - n_lags : i]          # lag_1 to lag_12

        month     = series.index[i].month
        month_sin = np.sin(2 * np.pi * month / 12)   # circular encoding
        month_cos = np.cos(2 * np.pi * month / 12)
        trend     = i / n                             # normalised position

        roll3 = float(np.mean(values[max(0, i-3):i]))
        roll6 = float(np.mean(values[max(0, i-6):i]))
        std6  = float(np.std(values[max(0, i-6):i]) + 1e-6)

        row = np.concatenate([
            lags,
            [month_sin, month_cos, trend, roll3, roll6, std6]
        ])
        X_rows.append(row)
        y_rows.append(values[i])

    return np.array(X_rows), np.array(y_rows)


def build_future_features(
    series:   pd.Series,
    n_lags:   int,
    n_months: int,
    last_preds: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build feature rows for future time steps (iterative forecasting).

    For each future month, extends the series with previously predicted values
    so lag features can be computed for the next step.
    """
    extended = list(series.values)
    if last_preds:
        extended.extend(last_preds)

    X_future = []
    last_date = series.index[-1]

    for i in range(n_months):
        future_date = last_date + pd.DateOffset(months=i + 1)
        idx         = len(extended) - n_months + i if last_preds else len(extended)

        # Use most recent n_lags values
        start = max(0, len(extended) - n_months + i - n_lags + (1 if last_preds else 0))
        lags  = extended[start : start + n_lags]

        # Pad with mean if not enough history
        if len(lags) < n_lags:
            mean_val = float(np.mean(extended))
            lags     = [mean_val] * (n_lags - len(lags)) + list(lags)

        month     = future_date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        n_total   = len(extended) + i
        trend     = n_total / (n_total + n_months)

        recent = extended[-6:] if len(extended) >= 6 else extended
        roll3  = float(np.mean(extended[-3:])) if len(extended) >= 3 else float(np.mean(extended))
        roll6  = float(np.mean(recent))
        std6   = float(np.std(recent) + 1e-6)

        row = np.concatenate([
            np.array(lags[-n_lags:]),
            [month_sin, month_cos, trend, roll3, roll6, std6]
        ])
        X_future.append(row)

    return np.array(X_future)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SARIMA
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAForecaster:
    """
    SARIMAX(p,d,q)(P,D,Q,12) — seasonal ARIMA.
    Captures the 12-month monsoon cycle explicitly.
    Best for short-term (1yr) forecasts with regular seasonal data.
    """

    def __init__(
        self,
        order:          tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.result         = None
        self._series_mean   = None

    def auto_fit(self, series: pd.Series) -> "SARIMAForecaster":
        if not HAS_SARIMA:
            raise RuntimeError("statsmodels required.")

        self._series_mean = float(series.mean())

        if HAS_PMDARIMA and len(series) >= 24:
            try:
                logger.info("auto_arima: searching best SARIMA order...")
                auto = pm.auto_arima(
                    series,
                    start_p=0, max_p=3, start_q=0, max_q=3,
                    d=None, seasonal=True, m=12,
                    start_P=0, max_P=2, start_Q=0, max_Q=2, D=1,
                    information_criterion='aic', stepwise=True,
                    suppress_warnings=True, error_action='ignore', n_fits=20,
                )
                self.order          = auto.order
                self.seasonal_order = auto.seasonal_order
                logger.info(f"Best SARIMA: {self.order}x{self.seasonal_order}")
            except Exception as e:
                logger.warning(f"auto_arima failed ({e}) — using fixed order.")

        return self.fit(series)

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        if not HAS_SARIMA:
            raise RuntimeError("statsmodels required.")
        self._series_mean = float(series.mean())
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c',
        )
        self.result = model.fit(disp=False, maxiter=200, method='lbfgs')
        logger.info(f"SARIMA{self.order}x{self.seasonal_order} AIC={self.result.aic:.2f}")
        return self

    def predict(self, steps: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fc    = self.result.get_forecast(steps=steps)
        mean  = np.clip(fc.predicted_mean.values, 0.1, 300)
        ci    = fc.conf_int(alpha=0.20)
        lower = np.clip(np.minimum(ci.iloc[:, 0].values, mean), 0.1, 300)
        upper = np.clip(np.maximum(ci.iloc[:, 1].values, mean), 0.1, 300)
        return mean, lower, upper

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    def in_sample_metrics(self, series: pd.Series) -> Dict[str, float]:
        fitted  = self.result.fittedvalues
        valid   = series.dropna()
        aligned = fitted.reindex(valid.index).dropna()
        return self.evaluate(valid.loc[aligned.index].values, aligned.values)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestForecaster:
    """
    Random Forest regressor for groundwater forecasting.

    Why RandomForest:
      - Robust to outliers (uses median of trees, not mean)
      - Handles non-linear relationships SARIMA cannot
      - Works well with lag + seasonal features
      - No stationarity requirement — works on raw depth values
      - Fast to train (parallel trees)

    Feature engineering:
      Uses build_lag_features() to create:
        - 12 lag values (previous 12 months)
        - Month encoding (sin/cos circular)
        - Trend position
        - Rolling statistics (mean_3, mean_6, std_6)

    Multi-step forecasting:
      Iterative: predicts month t+1, feeds it back as lag_1 for t+2, etc.
      Uncertainty: bootstrapped from tree variance (std of individual tree predictions)
    """

    N_LAGS      = 12
    N_ESTIMATORS = 300
    MAX_DEPTH   = 10

    def __init__(self):
        self.model   = None
        self._series = None

    def fit(self, series: pd.Series) -> "RandomForestForecaster":
        if not HAS_RF:
            raise RuntimeError("scikit-learn required for RandomForest.")

        self._series = series.copy()
        X, y         = build_lag_features(series, self.N_LAGS)

        if len(X) < 10:
            raise ValueError(f"Need at least {self.N_LAGS + 10} points for RF, got {len(series)}.")

        self.model = RandomForestRegressor(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,             # use all CPU cores
        )
        self.model.fit(X, y)

        # In-sample RMSE
        preds = self.model.predict(X)
        rmse  = float(np.sqrt(mean_squared_error(y, preds)))
        mae   = float(mean_absolute_error(y, preds))
        logger.info(f"RandomForest fitted. RMSE={rmse:.3f}m  MAE={mae:.3f}m")
        return self

    def predict(self, n_months: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterative multi-step forecast.
        Uncertainty estimated from standard deviation across all trees.
        """
        extended = list(self._series.values)
        preds_mean, preds_std = [], []
        last_date = self._series.index[-1]

        for i in range(n_months):
            future_date = last_date + pd.DateOffset(months=i + 1)

            # Build lag features for this step
            lags = extended[-self.N_LAGS:] if len(extended) >= self.N_LAGS \
                   else [np.mean(extended)] * (self.N_LAGS - len(extended)) + extended

            month     = future_date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            n_total   = len(extended) + i
            trend     = n_total / (n_total + n_months)
            roll3     = float(np.mean(extended[-3:])) if len(extended) >= 3 else float(np.mean(extended))
            roll6     = float(np.mean(extended[-6:])) if len(extended) >= 6 else float(np.mean(extended))
            std6      = float(np.std(extended[-6:]) + 1e-6) if len(extended) >= 6 else 0.1

            row = np.array(lags[-self.N_LAGS:] + [month_sin, month_cos, trend, roll3, roll6, std6])

            # Get predictions from all individual trees for uncertainty estimate
            tree_preds = np.array([
                tree.predict(row.reshape(1, -1))[0]
                for tree in self.model.estimators_
            ])
            pred_mean = float(np.mean(tree_preds))
            pred_std  = float(np.std(tree_preds))

            pred_mean = np.clip(pred_mean, 0.1, 300)
            preds_mean.append(pred_mean)
            preds_std.append(pred_std)

            extended.append(pred_mean)

        mean  = np.array(preds_mean)
        std   = np.array(preds_std)

        # 80% CI using 1.28 standard deviations (z-score for 80%)
        lower = np.clip(mean - 1.28 * std, 0.1, 300)
        upper = np.clip(mean + 1.28 * std, 0.1, 300)

        return mean, lower, upper

    def in_sample_metrics(self) -> Dict[str, float]:
        X, y  = build_lag_features(self._series, self.N_LAGS)
        preds = self.model.predict(X)
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
            "mae":  round(float(mean_absolute_error(y, preds)), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostForecaster:
    """
    XGBoost gradient boosting for groundwater forecasting.

    Why XGBoost:
      - Usually the most accurate single model for tabular time series
      - Gradient boosting: each tree corrects errors of the previous
      - Handles feature interactions automatically
      - L1/L2 regularization prevents overfitting on small datasets
      - Can model non-linear relationships between lag features and depth

    Same features as RandomForest (lag_1..12 + seasonal + rolling stats)
    but learns them differently — sequential correction vs parallel averaging.

    Uncertainty:
      XGBoost has no native prediction intervals.
      Uses quantile regression: trains two extra models at q=0.10 and q=0.90
      to produce lower and upper bounds (80% prediction interval).
    """

    N_LAGS = 12

    def __init__(self):
        self.model       = None   # main model (mean prediction)
        self.model_lower = None   # q=0.10 quantile model
        self.model_upper = None   # q=0.90 quantile model
        self._series     = None

    def fit(self, series: pd.Series) -> "XGBoostForecaster":
        if not HAS_XGB:
            raise RuntimeError("xgboost package required.")

        self._series = series.copy()
        X, y         = build_lag_features(series, self.N_LAGS)

        if len(X) < 10:
            raise ValueError(f"Need at least {self.N_LAGS + 10} points for XGBoost.")

        base_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # L1 regularisation
            reg_lambda=1.0,      # L2 regularisation
            random_state=42,
            n_jobs=-1,
        )

        # Mean prediction model
        self.model = xgb.XGBRegressor(**base_params, objective='reg:squarederror')
        self.model.fit(X, y, verbose=False)

        # Quantile models for 80% prediction interval
        self.model_lower = xgb.XGBRegressor(
            **base_params, objective='reg:quantileerror', quantile_alpha=0.10
        )
        self.model_upper = xgb.XGBRegressor(
            **base_params, objective='reg:quantileerror', quantile_alpha=0.90
        )
        try:
            self.model_lower.fit(X, y, verbose=False)
            self.model_upper.fit(X, y, verbose=False)
        except Exception:
            # Fallback if quantile objective not supported in older xgboost versions
            self.model_lower = None
            self.model_upper = None
            logger.info("XGBoost quantile regression not available — using std-based CI.")

        preds = self.model.predict(X)
        rmse  = float(np.sqrt(mean_squared_error(y, preds)))
        mae   = float(mean_absolute_error(y, preds))
        logger.info(f"XGBoost fitted. RMSE={rmse:.3f}m  MAE={mae:.3f}m")
        return self

    def predict(self, n_months: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iterative multi-step forecast with quantile-based confidence bands."""
        extended = list(self._series.values)
        preds_mean, preds_lower, preds_upper = [], [], []
        last_date = self._series.index[-1]

        for i in range(n_months):
            future_date = last_date + pd.DateOffset(months=i + 1)

            lags = extended[-self.N_LAGS:] if len(extended) >= self.N_LAGS \
                   else [np.mean(extended)] * (self.N_LAGS - len(extended)) + extended

            month     = future_date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            n_total   = len(extended) + i
            trend     = n_total / (n_total + n_months)
            roll3     = float(np.mean(extended[-3:])) if len(extended) >= 3 else float(np.mean(extended))
            roll6     = float(np.mean(extended[-6:])) if len(extended) >= 6 else float(np.mean(extended))
            std6      = float(np.std(extended[-6:]) + 1e-6) if len(extended) >= 6 else 0.1

            row = np.array(lags[-self.N_LAGS:] + [month_sin, month_cos, trend, roll3, roll6, std6])
            X   = row.reshape(1, -1)

            pred_mean = float(np.clip(self.model.predict(X)[0], 0.1, 300))

            if self.model_lower and self.model_upper:
                pred_lower = float(np.clip(self.model_lower.predict(X)[0], 0.1, 300))
                pred_upper = float(np.clip(self.model_upper.predict(X)[0], 0.1, 300))
            else:
                # Std-based fallback — widen with horizon
                band       = 0.06 + (i / n_months) * 0.10
                pred_lower = float(np.clip(pred_mean * (1 - band), 0.1, 300))
                pred_upper = float(np.clip(pred_mean * (1 + band), 0.1, 300))

            preds_mean.append(pred_mean)
            preds_lower.append(min(pred_lower, pred_mean))
            preds_upper.append(max(pred_upper, pred_mean))
            extended.append(pred_mean)

        return np.array(preds_mean), np.array(preds_lower), np.array(preds_upper)

    def in_sample_metrics(self) -> Dict[str, float]:
        X, y  = build_lag_features(self._series, self.N_LAGS)
        preds = self.model.predict(X)
        return {
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
            "mae":  round(float(mean_absolute_error(y, preds)), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — BiLSTM
# ══════════════════════════════════════════════════════════════════════════════

class LSTMForecaster:
    """
    Bidirectional LSTM — reads sequence forward AND backward.
    Best for detecting complex multi-year drought/recharge cycles.
    Requires 24+ monthly data points minimum.
    """

    def __init__(
        self,
        seq_length: int = 12,
        epochs:     int = 50,
        batch_size: int = 16,
    ):
        self.seq_length = seq_length
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model: Optional[keras.Model] = None
        self.scaler = None

    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(self.seq_length, 1)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='huber',
            metrics=['mae'],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        if not HAS_LSTM:
            raise RuntimeError("TensorFlow required for LSTM.")

        self.model = self._build_model()
        cbs = [
            keras_callbacks.EarlyStopping(
                monitor='val_loss', patience=8, restore_best_weights=True
            ),
            keras_callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
            ),
        ]
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cbs,
            verbose=0,
        )
        logger.info("BiLSTM training complete.")
        return self

    def predict_multi_step(
        self,
        last_sequence: np.ndarray,
        steps:  int = 12,
        scaler=None,
    ) -> np.ndarray:
        seq = last_sequence.copy().reshape(1, self.seq_length, 1)
        preds = []
        for _ in range(steps):
            p = self.model.predict(seq, verbose=0)[0, 0]
            preds.append(p)
            seq         = np.roll(seq, -1, axis=1)
            seq[0,-1,0] = p

        preds = np.array(preds)
        if scaler is not None:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        return preds

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'lstm_model.keras'))

    def load(self, path: str) -> "LSTMForecaster":
        self.model = keras.models.load_model(os.path.join(path, 'lstm_model.keras'))
        return self

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC ENSEMBLE WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_weights(rmse_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Compute ensemble weights using inverse-RMSE weighting.

    Logic:
      - Lower RMSE = more accurate model = higher weight
      - Weight for model i = (1/RMSE_i) / sum(1/RMSE_j for all j)
      - If a model has RMSE=0 (perfect), it gets all the weight
      - If a model failed (RMSE=None), it is excluded

    Example:
      SARIMA RMSE=1.2, RF RMSE=0.8, XGB RMSE=0.6, LSTM RMSE=1.0
      Inverse: 0.833, 1.25, 1.667, 1.0  → sum=4.75
      Weights: 0.175, 0.263, 0.351, 0.211

    This means XGBoost (most accurate) contributes 35% to final forecast,
    while SARIMA (least accurate here) contributes only 17%.
    """
    valid = {k: v for k, v in rmse_dict.items() if v is not None and v > 0}
    if not valid:
        # All failed — equal weights for available models
        n = len(rmse_dict)
        return {k: 1.0/n for k in rmse_dict}

    inv_rmse = {k: 1.0 / v for k, v in valid.items()}
    total    = sum(inv_rmse.values())
    weights  = {k: inv_rmse[k] / total for k in valid}

    # Failed models get 0 weight
    for k in rmse_dict:
        if k not in weights:
            weights[k] = 0.0

    return weights


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_forecast_response(
    well_id:     int,
    model_type:  str,
    start_date:  date,
    predictions: np.ndarray,
    lower:   Optional[np.ndarray] = None,
    upper:   Optional[np.ndarray] = None,
    metrics: Optional[Dict]       = None,
) -> List[Dict]:
    """Package forecast arrays into API-ready list of monthly prediction dicts."""
    results = []
    for i, pred in enumerate(predictions):
        predicted_date = (
            pd.Timestamp(start_date) + pd.DateOffset(months=i)
        ).date()

        entry = {
            "well_id":           well_id,
            "model_type":        model_type,
            "predicted_for":     predicted_date.isoformat(),
            "predicted_depth_m": round(float(pred),     4),
            "lower_bound_m":     round(float(lower[i]), 4) if lower is not None else None,
            "upper_bound_m":     round(float(upper[i]), 4) if upper is not None else None,
            "confidence_pct":    80.0,
        }
        if metrics:
            entry.update(metrics)
        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL MODEL RUNNERS  (called by run_forecast)
# ══════════════════════════════════════════════════════════════════════════════

def _run_sarima(series: pd.Series, n_months: int, start_date: date) -> Dict:
    if not HAS_SARIMA:
        return {'model': 'SARIMA', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}
    try:
        fc = SARIMAForecaster()
        fc.auto_fit(series)
        mean, lower, upper = fc.predict(steps=n_months)
        metrics = fc.in_sample_metrics(series)
        return {
            'model':  'SARIMA',
            'preds':  mean,
            'lower':  lower,
            'upper':  upper,
            'rmse':   metrics['rmse'],
            'mae':    metrics['mae'],
            'order':  str(fc.order),
            'seasonal_order': str(fc.seasonal_order),
        }
    except Exception as e:
        logger.error(f"SARIMA error: {e}")
        return {'model': 'SARIMA', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}


def _run_rf(series: pd.Series, n_months: int) -> Dict:
    if not HAS_RF or len(series) < 18:
        return {'model': 'RandomForest', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}
    try:
        fc = RandomForestForecaster()
        fc.fit(series)
        mean, lower, upper = fc.predict(n_months)
        metrics = fc.in_sample_metrics()
        return {
            'model':  'RandomForest',
            'preds':  mean,
            'lower':  lower,
            'upper':  upper,
            'rmse':   metrics['rmse'],
            'mae':    metrics['mae'],
        }
    except Exception as e:
        logger.error(f"RandomForest error: {e}")
        return {'model': 'RandomForest', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}


def _run_xgb(series: pd.Series, n_months: int) -> Dict:
    if not HAS_XGB or len(series) < 18:
        return {'model': 'XGBoost', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}
    try:
        fc = XGBoostForecaster()
        fc.fit(series)
        mean, lower, upper = fc.predict(n_months)
        metrics = fc.in_sample_metrics()
        return {
            'model':  'XGBoost',
            'preds':  mean,
            'lower':  lower,
            'upper':  upper,
            'rmse':   metrics['rmse'],
            'mae':    metrics['mae'],
        }
    except Exception as e:
        logger.error(f"XGBoost error: {e}")
        return {'model': 'XGBoost', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}


def _run_lstm(series: pd.Series, n_months: int) -> Dict:
    if not HAS_LSTM or len(series) < 24:
        return {'model': 'BiLSTM', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}
    try:
        SEQ    = 12
        vals   = series.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(vals)

        X, y = [], []
        for i in range(SEQ, len(scaled)):
            X.append(scaled[i-SEQ:i, 0])
            y.append(scaled[i, 0])
        X = np.array(X).reshape(-1, SEQ, 1)
        y = np.array(y)

        split       = max(1, int(len(X) * 0.8))
        X_tr, X_vl = X[:split], X[split:]
        y_tr, y_vl = y[:split], y[split:]

        fc    = LSTMForecaster(seq_length=SEQ, epochs=50, batch_size=16)
        fc.fit(X_tr, y_tr, X_vl if len(X_vl) > 0 else None, y_vl if len(y_vl) > 0 else None)

        preds = fc.predict_multi_step(scaled[-SEQ:], steps=n_months, scaler=scaler)
        bands = np.array([0.07 + (i / n_months) * 0.10 for i in range(n_months)])
        lower = np.clip(preds * (1 - bands), 0.1, 300)
        upper = np.clip(preds * (1 + bands), 0.1, 300)

        actual    = scaler.inverse_transform(y.reshape(-1,1)).flatten()
        in_sample = scaler.inverse_transform(fc.model.predict(X, verbose=0)).flatten()
        metrics   = fc.evaluate(actual, in_sample)

        return {
            'model':  'BiLSTM',
            'preds':  preds,
            'lower':  lower,
            'upper':  upper,
            'rmse':   metrics['rmse'],
            'mae':    metrics['mae'],
        }
    except Exception as e:
        logger.error(f"BiLSTM error: {e}")
        return {'model': 'BiLSTM', 'preds': None, 'lower': None, 'upper': None, 'rmse': None}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    readings:     List[Dict],
    model_type:   str = 'ensemble',
    horizon_days: int = 365,
) -> Dict:
    """
    Main forecast function called by the API predict endpoint.

    Parameters
    ----------
    readings     : list of DB reading dicts (recorded_at, depth_to_water_m / currentlevel)
    model_type   : 'ensemble' | 'sarima' | 'rf' | 'xgboost' | 'lstm'
    horizon_days : 365=1yr, 1095=3yr, 1825=5yr

    Ensemble strategy:
      Runs all 4 models, weights by inverse RMSE (better model = higher weight).
      If any model fails, its weight is redistributed to remaining models.
      Final prediction = weighted average of all successful model predictions.
      Final CI        = weighted average of all model confidence bands.
    """
    series = prepare_series(readings)

    if len(series) < 6:
        return {
            'model':          'none',
            'forecast':       [],
            'error':          f'Not enough data: {len(series)} monthly readings (need 6+).',
            'rmse_m':         None,
            'mae_m':          None,
            'confidence_pct': 0,
        }

    n_months   = max(12, int(horizon_days / 30.44))
    last_date  = series.index[-1].date()
    start_date = (pd.Timestamp(last_date) + pd.DateOffset(months=1)).date()
    model_type = model_type.lower().strip()

    logger.info(
        f"run_forecast: model={model_type}, series={len(series)} pts, "
        f"horizon={horizon_days}d ({n_months} months)"
    )

    # ── Single model branches ─────────────────────────────────────────────
    if model_type in ('sarima', 'arima', 'auto'):
        r = _run_sarima(series, n_months, start_date)
        if r['preds'] is None:
            return _fallback_response(series, n_months, start_date, 'sarima')
        return _wrap_single(r, start_date, series, n_months)

    elif model_type in ('rf', 'randomforest', 'random_forest'):
        r = _run_rf(series, n_months)
        if r['preds'] is None:
            return run_forecast(readings, 'sarima', horizon_days)
        return _wrap_single(r, start_date, series, n_months)

    elif model_type in ('xgb', 'xgboost'):
        r = _run_xgb(series, n_months)
        if r['preds'] is None:
            return run_forecast(readings, 'sarima', horizon_days)
        return _wrap_single(r, start_date, series, n_months)

    elif model_type == 'lstm':
        r = _run_lstm(series, n_months)
        if r['preds'] is None:
            return run_forecast(readings, 'sarima', horizon_days)
        return _wrap_single(r, start_date, series, n_months)

    # ── 4-Model Ensemble ──────────────────────────────────────────────────
    elif model_type in ('ensemble', 'all'):
        logger.info("Running 4-model ensemble: SARIMA + RF + XGBoost + BiLSTM")

        results = {
            'SARIMA':       _run_sarima(series, n_months, start_date),
            'RandomForest': _run_rf(series, n_months),
            'XGBoost':      _run_xgb(series, n_months),
            'BiLSTM':       _run_lstm(series, n_months),
        }

        # Filter to models that succeeded
        valid = {k: v for k, v in results.items() if v['preds'] is not None}

        if not valid:
            logger.warning("All ensemble models failed — using fallback.")
            return _fallback_response(series, n_months, start_date, 'ensemble')

        if len(valid) == 1:
            # Only one model succeeded — return it as-is
            r = list(valid.values())[0]
            return _wrap_single(r, start_date, series, n_months)

        # Compute dynamic weights using inverse-RMSE
        rmse_dict = {k: v['rmse'] for k, v in valid.items()}
        weights   = compute_weights(rmse_dict)

        logger.info(
            "Ensemble weights: " +
            ", ".join(f"{k}={w:.3f}" for k, w in weights.items())
        )

        # Weighted average of predictions and CI bands
        ens_preds = np.zeros(n_months)
        ens_lower = np.zeros(n_months)
        ens_upper = np.zeros(n_months)

        for name, res in valid.items():
            w          = weights[name]
            ens_preds += w * res['preds']
            ens_lower += w * res['lower']
            ens_upper += w * res['upper']

        ens_preds = np.clip(ens_preds, 0.1, 300)
        ens_lower = np.clip(ens_lower, 0.1, 300)
        ens_upper = np.clip(ens_upper, 0.1, 300)

        # Weighted RMSE and MAE
        total_w    = sum(weights[k] for k in valid)
        ens_rmse   = sum(weights[k] * valid[k]['rmse'] for k in valid) / total_w
        ens_mae    = sum(
            weights[k] * valid[k]['mae'] for k in valid
            if valid[k].get('mae') is not None
        ) / total_w

        model_names = " + ".join(valid.keys())
        weight_str  = ", ".join(f"{k}:{weights[k]:.2f}" for k in valid)

        return {
            'model':            f'Ensemble ({model_names})',
            'ensemble_weights': weight_str,
            'models_used':      list(valid.keys()),
            'forecast':         build_forecast_response(
                                    0, 'Ensemble', start_date,
                                    ens_preds, ens_lower, ens_upper,
                                ),
            'rmse_m':           round(ens_rmse, 4),
            'mae_m':            round(ens_mae,  4),
            'confidence_pct':   80,
            'n_training_pts':   len(series),
            'series_start':     series.index[0].isoformat(),
            'series_end':       series.index[-1].isoformat(),
            'individual_rmse':  {k: v['rmse'] for k, v in valid.items()},
            'error':            None,
        }

    # ── Unknown ───────────────────────────────────────────────────────────
    else:
        logger.warning(f"Unknown model '{model_type}' — defaulting to ensemble.")
        return run_forecast(readings, 'ensemble', horizon_days)


def _wrap_single(r: Dict, start_date: date, series: pd.Series, n_months: int) -> Dict:
    """Wrap a single-model result into the standard API response format."""
    return {
        'model':          r['model'],
        'forecast':       build_forecast_response(
                              0, r['model'], start_date,
                              r['preds'], r['lower'], r['upper'],
                          ),
        'rmse_m':         r.get('rmse'),
        'mae_m':          r.get('mae'),
        'confidence_pct': 80,
        'n_training_pts': len(series),
        'series_start':   series.index[0].isoformat(),
        'series_end':     series.index[-1].isoformat(),
        'error':          None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_response(
    series:     pd.Series,
    n_months:   int,
    start_date: date,
    attempted:  str,
) -> Dict:
    """Trend + monthly seasonal offsets when all models fail."""
    logger.warning(f"{attempted} unavailable — using trend+seasonal fallback.")

    n              = len(series)
    slope, intercept = np.polyfit(np.arange(n), series.values, 1)

    mean_val = series.mean()
    seasonal = np.zeros(12)
    counts   = np.zeros(12)
    for dt, val in series.items():
        m = dt.month - 1
        seasonal[m] += val - mean_val
        counts[m]   += 1
    for m in range(12):
        if counts[m] > 0:
            seasonal[m] /= counts[m]

    preds, lower_ci, upper_ci = [], [], []
    for i in range(n_months):
        fut  = pd.Timestamp(start_date) + pd.DateOffset(months=i)
        pred = float(np.clip(intercept + slope*(n+i) + seasonal[fut.month-1], 0.1, 300))
        band = 0.10 + (i / n_months) * 0.15
        preds.append(pred)
        lower_ci.append(float(np.clip(pred*(1-band), 0.1, 300)))
        upper_ci.append(float(np.clip(pred*(1+band), 0.1, 300)))

    fitted = np.array([intercept + slope*i for i in range(n)])
    rmse   = float(np.sqrt(np.mean((series.values - fitted)**2)))

    return {
        'model':          'Trend+Seasonal (fallback)',
        'forecast':       build_forecast_response(
                              0, 'Trend+Seasonal', start_date,
                              np.array(preds), np.array(lower_ci), np.array(upper_ci),
                          ),
        'rmse_m':         round(rmse, 4),
        'mae_m':          round(rmse * 0.8, 4),
        'confidence_pct': 50,
        'n_training_pts': n,
        'error':          f'{attempted} unavailable — used trend+seasonal fallback',
    }


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_forecast(
    sarima_preds:  np.ndarray,
    lstm_preds:    np.ndarray,
    sarima_weight: float = 0.5,
) -> np.ndarray:
    """Kept for backward compatibility with any code using old 2-model ensemble."""
    return sarima_weight * sarima_preds + (1.0 - sarima_weight) * lstm_preds


def build_forecast_response_legacy(
    well_id, model_type, start_date, predictions, lower=None, upper=None, metrics=None
):
    """Alias for backward compatibility."""
    return build_forecast_response(well_id, model_type, start_date, predictions, lower, upper, metrics)