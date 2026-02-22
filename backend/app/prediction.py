"""
prediction.py — ARIMA & LSTM Groundwater Level Forecasting
Groundwater Level Prediction System

Supports:
  - ARIMA  (statsmodels SARIMAX)
  - LSTM   (Keras/TensorFlow Bidirectional LSTM)
  - Ensemble (weighted average of both)
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Optional heavy imports (graceful degradation) ─────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    logger.warning("statsmodels not available — ARIMA disabled.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    logger.warning("TensorFlow not available — LSTM disabled.")

from app.config import settings
from app.preprocessing import scale_series, create_sequences


# ── ARIMA Forecaster ──────────────────────────────────────────────────────────
class ARIMAForecaster:
    def __init__(self, order: tuple = (2, 1, 2)):
        self.order = order
        self.model = None
        self.result = None

    def fit(self, series: np.ndarray) -> "ARIMAForecaster":
        if not HAS_ARIMA:
            raise RuntimeError("statsmodels is required for ARIMA.")
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=(1, 0, 1, 12),   # Annual seasonality
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False)
        logger.info(f"ARIMA{self.order} fitted. AIC={self.result.aic:.2f}")
        return self

    def predict(self, steps: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (forecast, lower_ci, upper_ci)."""
        fc = self.result.get_forecast(steps=steps)
        mean = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.10)       # 90% confidence interval
        return mean, ci.iloc[:, 0].values, ci.iloc[:, 1].values

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


# ── LSTM Forecaster ───────────────────────────────────────────────────────────
class LSTMForecaster:
    def __init__(
        self,
        seq_length: int = 30,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[keras.Model] = None
        self.scaler = None

    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(self.seq_length, 1)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMForecaster":
        if not HAS_LSTM:
            raise RuntimeError("TensorFlow is required for LSTM.")
        self.model = self._build_model()

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ]
        validation_data = (X_val, y_val) if X_val is not None else None

        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cbs,
            verbose=0,
        )
        logger.info("LSTM training complete.")
        return self

    def predict_multi_step(
        self,
        last_sequence: np.ndarray,
        steps: int = 30,
        scaler=None,
    ) -> np.ndarray:
        """
        Iterative multi-step forecast:
        Feeds each predicted value back as input for the next step.
        """
        seq = last_sequence.copy().reshape(1, self.seq_length, 1)
        predictions = []
        for _ in range(steps):
            pred = self.model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = pred

        preds = np.array(predictions)
        if scaler is not None:
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        return preds

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "lstm_model.keras"))
        logger.info(f"LSTM saved → {path}")

    def load(self, path: str) -> "LSTMForecaster":
        self.model = keras.models.load_model(os.path.join(path, "lstm_model.keras"))
        return self

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


# ── Ensemble Forecast ─────────────────────────────────────────────────────────
def ensemble_forecast(
    arima_preds: np.ndarray,
    lstm_preds: np.ndarray,
    arima_weight: float = 0.4,
) -> np.ndarray:
    """
    Weighted average ensemble.
    ARIMA captures linear trends; LSTM captures non-linear patterns.
    """
    lstm_weight = 1.0 - arima_weight
    return arima_weight * arima_preds + lstm_weight * lstm_preds


# ── Forecast Orchestrator ─────────────────────────────────────────────────────
def build_forecast_response(
    well_id: int,
    model_type: str,
    start_date: date,
    predictions: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    metrics: Optional[Dict] = None,
) -> List[Dict]:
    """Package forecast arrays into API-ready dicts."""
    results = []
    for i, pred in enumerate(predictions):
        entry = {
            "well_id": well_id,
            "model_type": model_type,
            "predicted_for": (start_date + timedelta(days=i)).isoformat(),
            "predicted_depth_m": round(float(pred), 4),
            "lower_bound_m": round(float(lower[i]), 4) if lower is not None else None,
            "upper_bound_m": round(float(upper[i]), 4) if upper is not None else None,
            "confidence_pct": 90.0,
        }
        if metrics:
            entry.update(metrics)
        results.append(entry)
    return results
