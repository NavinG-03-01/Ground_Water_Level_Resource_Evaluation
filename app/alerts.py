"""
alerts.py — Risk Classification & Alert Management
Groundwater Level Prediction System
"""

from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from app.config import settings
from app.models.db_models import Alert, AlertSeverity, Well


# ── Risk Classification ────────────────────────────────────────────────────────
def classify_level(depth_m: float) -> AlertSeverity:
    """
    Classify groundwater depth into risk tiers.
    depth_m = metres below ground level (higher = more depleted)
    """
    if depth_m >= settings.ALERT_CRITICAL_BELOW_M:
        return AlertSeverity.CRITICAL
    elif depth_m >= settings.ALERT_WARNING_BELOW_M:
        return AlertSeverity.WARNING
    return AlertSeverity.NORMAL


def classify_trend(slope_m_per_year: float) -> Dict:
    """Classify rate-of-change severity."""
    if slope_m_per_year > 2.0:
        return {"trend_risk": "critical", "message": f"Rapid depletion: {slope_m_per_year:.2f}m/yr decline"}
    elif slope_m_per_year > 0.5:
        return {"trend_risk": "warning", "message": f"Moderate depletion: {slope_m_per_year:.2f}m/yr decline"}
    elif slope_m_per_year < -0.5:
        return {"trend_risk": "normal", "message": f"Recovery detected: {abs(slope_m_per_year):.2f}m/yr rise"}
    return {"trend_risk": "normal", "message": "Stable trend"}


# ── Alert Generation ──────────────────────────────────────────────────────────
def generate_alert_message(
    well_name: str,
    depth_m: float,
    severity: AlertSeverity,
    forecast_depth_m: Optional[float] = None,
) -> str:
    """Build human-readable alert message."""
    base = (
        f"⚠️ [{severity.upper()}] Well '{well_name}': "
        f"Current depth {depth_m:.2f}m below ground level."
    )
    if forecast_depth_m:
        base += f" 30-day forecast: {forecast_depth_m:.2f}m."
    if severity == AlertSeverity.CRITICAL:
        base += " Immediate intervention recommended."
    elif severity == AlertSeverity.WARNING:
        base += " Increased monitoring advised."
    return base


async def check_and_create_alerts(
    well_id: int,
    depth_m: float,
    db: AsyncSession,
    forecast_depth_m: Optional[float] = None,
) -> Optional[Alert]:
    """
    Evaluate current depth against thresholds.
    Creates DB alert record if severity is WARNING or CRITICAL.
    Skips if identical unacknowledged alert already exists.
    """
    severity = classify_level(depth_m)
    if severity == AlertSeverity.NORMAL:
        return None

    # Fetch well name
    result = await db.execute(select(Well).where(Well.id == well_id))
    well = result.scalar_one_or_none()
    well_name = well.station_name if well else f"Well#{well_id}"

    message = generate_alert_message(well_name, depth_m, severity, forecast_depth_m)

    alert = Alert(
        well_id=well_id,
        severity=severity,
        message=message,
        triggered_at=datetime.utcnow(),
    )
    db.add(alert)
    await db.commit()
    logger.warning(f"Alert created: {message}")
    return alert


# ── Alert Summary ─────────────────────────────────────────────────────────────
def build_alert_summary(alerts: List[Alert]) -> Dict:
    """Aggregate alert counts for dashboard consumption."""
    counts = {s.value: 0 for s in AlertSeverity}
    for a in alerts:
        counts[a.severity.value] += 1
    return {
        "total": len(alerts),
        "critical": counts["critical"],
        "warning": counts["warning"],
        "normal": counts["normal"],
        "unacknowledged": sum(1 for a in alerts if not a.is_acknowledged),
    }
