from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    APP_NAME:    str = "Groundwater Level Prediction System"
    APP_VERSION: str = "1.0.0"
    DEBUG:       bool = Field(default=False)
    API_PREFIX:  str = "/api/v1"

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://gwuser:gwpassword@localhost:5432/groundwater_db"
    )
    DB_POOL_SIZE:    int  = 10
    DB_MAX_OVERFLOW: int  = 20
    DB_ECHO:         bool = False

    SECRET_KEY:                  str = Field(default="CHANGE_ME_IN_PRODUCTION")
    ALGORITHM:                   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    MODEL_DIR:       str   = Field(default="models/saved")
    LSTM_SEQ_LENGTH: int   = 12          # 12 months (monthly resampled CGWB data)
    LSTM_EPOCHS:     int   = 50
    LSTM_BATCH_SIZE: int   = 32
    ARIMA_ORDER:     tuple = (2, 1, 2)

    CGWB_API_URL:                       Optional[str] = Field(default=None)
    DATA_REFRESH_INTERVAL_MINUTES:      int = 60
    SIMULATED_STREAM_INTERVAL_SECONDS:  int = 5

    ALERT_CRITICAL_BELOW_M: float = 20.0
    ALERT_WARNING_BELOW_M:  float = 10.0

    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]

    model_config = {"env_file": ".env", "case_sensitive": True}

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()