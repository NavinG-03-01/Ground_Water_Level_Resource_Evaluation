from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime,
    Boolean, ForeignKey, Text, Enum as SAEnum,
    UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from app.database import Base
import enum

class AlertSeverity(str, enum.Enum):
    NORMAL   = "normal"
    WARNING  = "warning"
    CRITICAL = "critical"

class ModelType(str, enum.Enum):
    ARIMA    = "arima"
    LSTM     = "lstm"
    ENSEMBLE = "ensemble"

class Well(Base):
    __tablename__ = "wells"
    id            = Column(Integer, primary_key=True, index=True)
    station_code  = Column(String(100), unique=True, nullable=False, index=True)
    station_name  = Column(String(300))
    state         = Column(String(100))
    state_code    = Column(Integer)
    district      = Column(String(100))
    district_code = Column(Integer)
    basin         = Column(String(200))
    sub_basin     = Column(String(200))
    latitude      = Column(Float)
    longitude     = Column(Float)
    well_depth_m  = Column(Float)
    aquifer_type  = Column(String(50))
    created_at    = Column(DateTime, default=datetime.utcnow)

    readings    = relationship("WaterLevelReading", back_populates="well", lazy="dynamic")
    predictions = relationship("Prediction",        back_populates="well", lazy="dynamic")
    alerts      = relationship("Alert",             back_populates="well", lazy="dynamic")

class WaterLevelReading(Base):
    __tablename__ = "water_level_readings"
    id               = Column(Integer, primary_key=True, index=True)
    well_id          = Column(Integer, ForeignKey("wells.id"), nullable=False)
    recorded_at      = Column(DateTime, nullable=False, index=True)
    depth_to_water_m = Column(Float, nullable=False)
    level_diff       = Column(Float)
    rainfall_mm      = Column(Float)
    temperature_c    = Column(Float)
    source           = Column(String(50), default="CGWB")
    is_interpolated  = Column(Boolean, default=False)
    well = relationship("Well", back_populates="readings")
    __table_args__ = (
        UniqueConstraint("well_id", "recorded_at", name="uq_reading_well_time"),
    )

class Prediction(Base):
    __tablename__ = "predictions"
    id                = Column(Integer, primary_key=True, index=True)
    well_id           = Column(Integer, ForeignKey("wells.id"), nullable=False)
    model_type        = Column(String(20), nullable=False)
    predicted_for     = Column(DateTime, nullable=False)
    predicted_depth_m = Column(Float, nullable=False)
    lower_bound_m     = Column(Float)
    upper_bound_m     = Column(Float)
    confidence_pct    = Column(Float)
    generated_at      = Column(DateTime, default=datetime.utcnow)
    rmse              = Column(Float)
    mae               = Column(Float)
    well = relationship("Well", back_populates="predictions")

class Alert(Base):
    __tablename__ = "alerts"
    id              = Column(Integer, primary_key=True, index=True)
    well_id         = Column(Integer, ForeignKey("wells.id"), nullable=False)
    severity        = Column(String(20), nullable=False)
    message         = Column(Text)
    triggered_at    = Column(DateTime, default=datetime.utcnow)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, nullable=True)
    well = relationship("Well", back_populates="alerts")

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(50), unique=True, nullable=False, index=True)
    email           = Column(String(200), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name       = Column(String(200))
    role            = Column(String(20), default="viewer")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_login      = Column(DateTime, nullable=True)
