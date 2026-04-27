"""
utils.py — Updated Pydantic schemas for CGWB dataset
"""

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator
from fastapi import HTTPException, status
from app.config import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


# ── Schemas ───────────────────────────────────────────────────────────────────
class WellCreate(BaseModel):
    station_code:  str
    station_name:  str
    state:         str
    district:      str
    latitude:      float
    longitude:     float
    well_depth_m:  Optional[float] = None
    aquifer_type:  Optional[str]   = None
    basin:         Optional[str]   = None   # ← new, from your dataset
    sub_basin:     Optional[str]   = None   # ← new

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and 90")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_lon(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v


class ReadingCreate(BaseModel):
    well_id:          int
    recorded_at:      datetime
    depth_to_water_m: float         # maps to currentlevel
    rainfall_mm:      Optional[float] = None
    temperature_c:    Optional[float] = None

    @field_validator("depth_to_water_m")
    @classmethod
    def validate_depth(cls, v):
        # CGWB currentlevel ranges from slightly negative (artesian) to ~300m
        if v < -5 or v > 300:
            raise ValueError("currentlevel must be between -5 and 300 metres")
        return v


class ForecastRequest(BaseModel):
    well_id:      int
    model_type:   str = "arima"       # arima | lstm | ensemble
    horizon_days: int = 30

    @field_validator("model_type")
    @classmethod
    def validate_model(cls, v):
        if v not in {"arima", "lstm", "ensemble"}:
            raise ValueError("model_type must be arima, lstm, or ensemble")
        return v

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v):
        if not (1 <= v <= 365):
            raise ValueError("horizon_days must be between 1 and 365")
        return v


class UserCreate(BaseModel):
    username:  str
    email:     str
    password:  str
    full_name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


def paginate(items: list, page: int = 1, page_size: int = 50) -> dict:
    total = len(items)
    start = (page - 1) * page_size
    end   = start + page_size
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
        "data":      items[start:end],
    }