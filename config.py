"""
config.py — Application & Database Configuration
Groundwater Level Prediction System
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────
    APP_NAME: str = "Groundwater Level Prediction System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    API_PREFIX: str = "/api/v1"

    # ── Database ──────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/groundwater_db"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    # ── Security ──────────────────────────────────────────
    SECRET_KEY: str = Field(default="CHANGE_ME_IN_PRODUCTION")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── ML Model ──────────────────────────────────────────
    MODEL_DIR: str = Field(default="models/saved")
    LSTM_SEQ_LENGTH: int = 30
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    ARIMA_ORDER: tuple = (2, 1, 2)

    # ── Data Ingestion ────────────────────────────────────
    CGWB_API_URL: Optional[str] = Field(default=None)
    DATA_REFRESH_INTERVAL_MINUTES: int = 60
    SIMULATED_STREAM_INTERVAL_SECONDS: int = 5

    # ── Alert Thresholds ──────────────────────────────────
    ALERT_CRITICAL_BELOW_M: float = 20.0
    ALERT_WARNING_BELOW_M: float = 10.0

    # ── CORS ──────────────────────────────────────────────
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]

    model_config = {"env_file": ".env", "case_sensitive": True}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
