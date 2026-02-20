"""
db_models.py — SQLAlchemy ORM Models (PostGIS-enabled)
Groundwater Level Prediction System
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Date, Boolean,
    ForeignKey, Text, Enum as SAEnum, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from app.database import Base
import enum


# ── Enums ─────────────────────────────────────────────────────────────────────
class AlertSeverity(str, enum.Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelType(str, enum.Enum):
    ARIMA = "arima"
    LSTM = "lstm"


# ── Well / Station ────────────────────────────────────────────────────────────
class Well(Base):
    """Monitoring well / DWLR station metadata."""
    __tablename__ = "wells"

    id = Column(Integer, primary_key=True, index=True)
    station_code = Column(String(50), unique=True, nullable=False, index=True)
    station_name = Column(String(200))
    state = Column(String(100))
    district = Column(String(100))
    block = Column(String(100))
    aquifer_type = Column(String(50))          # confined / unconfined
    well_depth_m = Column(Float)
    geom = Column(Geometry("POINT", srid=4326))  # PostGIS spatial column
    created_at = Column(DateTime, default=datetime.utcnow)

    readings = relationship("WaterLevelReading", back_populates="well", lazy="dynamic")
    predictions = relationship("Prediction", back_populates="well", lazy="dynamic")
    alerts = relationship("Alert", back_populates="well", lazy="dynamic")

    __table_args__ = (
        Index("idx_wells_state_district", "state", "district"),
    )


# ── Water Level Readings ──────────────────────────────────────────────────────
class WaterLevelReading(Base):
    """Raw / ingested groundwater level observations."""
    __tablename__ = "water_level_readings"

    id = Column(Integer, primary_key=True, index=True)
    well_id = Column(Integer, ForeignKey("wells.id"), nullable=False)
    recorded_at = Column(DateTime, nullable=False, index=True)
    depth_to_water_m = Column(Float, nullable=False)   # metres below ground level
    water_table_elevation_m = Column(Float)            # amsl
    rainfall_mm = Column(Float)
    temperature_c = Column(Float)
    source = Column(String(50), default="CGWB")        # CGWB | simulated
    is_interpolated = Column(Boolean, default=False)

    well = relationship("Well", back_populates="readings")

    __table_args__ = (
        UniqueConstraint("well_id", "recorded_at", name="uq_reading_well_time"),
        Index("idx_readings_well_time", "well_id", "recorded_at"),
    )


# ── Predictions ───────────────────────────────────────────────────────────────
class Prediction(Base):
    """Model forecast output."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    well_id = Column(Integer, ForeignKey("wells.id"), nullable=False)
    model_type = Column(SAEnum(ModelType), nullable=False)
    predicted_for = Column(Date, nullable=False)
    predicted_depth_m = Column(Float, nullable=False)
    lower_bound_m = Column(Float)
    upper_bound_m = Column(Float)
    confidence_pct = Column(Float)
    generated_at = Column(DateTime, default=datetime.utcnow)
    rmse = Column(Float)
    mae = Column(Float)

    well = relationship("Well", back_populates="predictions")

    __table_args__ = (
        Index("idx_predictions_well_date", "well_id", "predicted_for"),
    )


# ── Alerts ────────────────────────────────────────────────────────────────────
class Alert(Base):
    """Risk classification & notification log."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    well_id = Column(Integer, ForeignKey("wells.id"), nullable=False)
    severity = Column(SAEnum(AlertSeverity), nullable=False)
    message = Column(Text)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    well = relationship("Well", back_populates="alerts")
