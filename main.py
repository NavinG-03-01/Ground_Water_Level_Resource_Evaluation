"""
main.py — FastAPI Application Entry Point
Groundwater Level Prediction System
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from datetime import date, datetime
from typing import List, Optional
import asyncio

from app.config import settings
from app.database import get_db, init_db, close_db
from app.models.db_models import Well, WaterLevelReading, Prediction, Alert
from app.models.user_model import User
from app.data_ingestion import refresh_all_wells, simulated_stream, ingest_from_csv
from app.preprocessing import full_pipeline, load_readings_to_df, resample_to_daily
from app.prediction import ARIMAForecaster, LSTMForecaster, ensemble_forecast, build_forecast_response
from app.analytics import (
    compute_linear_trend, seasonal_decomposition,
    monsoon_recharge_analysis, detect_anomalies_zscore,
    compute_summary_statistics,
)
from app.alerts import check_and_create_alerts, build_alert_summary
from app.utils import (
    WellCreate, ReadingCreate, ForecastRequest, UserCreate, TokenResponse,
    hash_password, verify_password, create_access_token, decode_token, paginate,
)

import pandas as pd
import numpy as np


# ── Scheduler ─────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()

    # Scheduled data refresh
    async def _refresh():
        async with get_db() as db:
            await refresh_all_wells(db)

    scheduler.add_job(
        _refresh,
        trigger=IntervalTrigger(minutes=settings.DATA_REFRESH_INTERVAL_MINUTES),
        id="data_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Scheduler started (interval: {settings.DATA_REFRESH_INTERVAL_MINUTES}min).")
    yield
    scheduler.shutdown(wait=False)
    await close_db()
    logger.info("Application shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "End-to-end Groundwater Level Prediction System using ARIMA + LSTM. "
        "Supports near-real-time simulation, PostGIS spatial storage, and trend analytics."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/auth/register", tags=["Auth"], status_code=201)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.username == payload.username))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Username already exists.")
    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    return {"message": "User created", "username": user.username}


@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(username: str, password: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")
    token = create_access_token({"sub": user.username, "role": user.role.value})
    return TokenResponse(access_token=token)


# ═══════════════════════════════════════════════════════════════════════════════
# WELLS
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/wells", tags=["Wells"], status_code=201)
async def create_well(payload: WellCreate, db: AsyncSession = Depends(get_db)):
    from geoalchemy2.elements import WKTElement
    geom = WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326)
    well = Well(
        station_code=payload.station_code,
        station_name=payload.station_name,
        state=payload.state,
        district=payload.district,
        well_depth_m=payload.well_depth_m,
        aquifer_type=payload.aquifer_type,
        geom=geom,
    )
    db.add(well)
    await db.commit()
    await db.refresh(well)
    return {"id": well.id, "station_code": well.station_code, "message": "Well created."}


@app.get("/api/v1/wells", tags=["Wells"])
async def list_wells(
    state: Optional[str] = None,
    district: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    q = select(Well)
    if state:
        q = q.where(Well.state.ilike(f"%{state}%"))
    if district:
        q = q.where(Well.district.ilike(f"%{district}%"))
    result = await db.execute(q)
    wells = result.scalars().all()
    data = [
        {"id": w.id, "station_code": w.station_code, "station_name": w.station_name,
         "state": w.state, "district": w.district}
        for w in wells
    ]
    return paginate(data, page, page_size)


@app.get("/api/v1/wells/{well_id}", tags=["Wells"])
async def get_well(well_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Well).where(Well.id == well_id))
    well = result.scalar_one_or_none()
    if not well:
        raise HTTPException(404, "Well not found.")
    return {
        "id": well.id, "station_code": well.station_code,
        "station_name": well.station_name, "state": well.state,
        "district": well.district, "well_depth_m": well.well_depth_m,
        "aquifer_type": well.aquifer_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/ingestion/reading", tags=["Ingestion"], status_code=201)
async def add_reading(payload: ReadingCreate, db: AsyncSession = Depends(get_db)):
    reading = WaterLevelReading(**payload.model_dump())
    db.add(reading)
    await db.commit()
    return {"message": "Reading added."}


@app.post("/api/v1/ingestion/csv", tags=["Ingestion"])
async def ingest_csv(filepath: str, background_tasks: BackgroundTasks,
                      db: AsyncSession = Depends(get_db)):
    background_tasks.add_task(ingest_from_csv, filepath, db)
    return {"message": f"CSV ingestion started for {filepath}."}


@app.get("/api/v1/ingestion/stream/{well_id}", tags=["Ingestion"])
async def stream_readings(well_id: int, duration_seconds: int = Query(30, le=300)):
    """Simulate a real-time stream of readings for a well (Server-Sent Events style)."""
    from fastapi.responses import StreamingResponse
    import json

    async def event_generator():
        count = 0
        max_readings = duration_seconds // settings.SIMULATED_STREAM_INTERVAL_SECONDS
        async for reading in simulated_stream(well_id):
            yield f"data: {json.dumps(reading)}\n\n"
            count += 1
            if count >= max_readings:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════════
# READINGS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/wells/{well_id}/readings", tags=["Readings"])
async def get_readings(
    well_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    q = select(WaterLevelReading).where(WaterLevelReading.well_id == well_id)
    if start_date:
        q = q.where(WaterLevelReading.recorded_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        q = q.where(WaterLevelReading.recorded_at <= datetime.combine(end_date, datetime.max.time()))
    q = q.order_by(WaterLevelReading.recorded_at.desc())
    result = await db.execute(q)
    readings = result.scalars().all()
    data = [
        {"recorded_at": r.recorded_at.isoformat(), "depth_to_water_m": r.depth_to_water_m,
         "rainfall_mm": r.rainfall_mm, "temperature_c": r.temperature_c, "source": r.source}
        for r in readings
    ]
    return paginate(data, page, page_size)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/predict", tags=["Prediction"])
async def predict(payload: ForecastRequest, db: AsyncSession = Depends(get_db)):
    """
    Run ARIMA / LSTM / Ensemble forecast for a well.
    Uses the last 2 years of historical data for training.
    """
    # Fetch readings
    cutoff = datetime(datetime.utcnow().year - 2, 1, 1)
    q = (
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == payload.well_id)
        .where(WaterLevelReading.recorded_at >= cutoff)
        .order_by(WaterLevelReading.recorded_at)
    )
    result = await db.execute(q)
    readings = [
        {"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m}
        for r in result.scalars().all()
    ]

    if len(readings) < settings.LSTM_SEQ_LENGTH + 10:
        raise HTTPException(422, f"Insufficient data: need at least {settings.LSTM_SEQ_LENGTH + 10} readings.")

    # Preprocess
    df, X_train, y_train, scaler = full_pipeline(readings, settings.LSTM_SEQ_LENGTH)
    series = df["depth_to_water_m"].values
    start_date = date.today()

    model_type = payload.model_type
    horizon = payload.horizon_days

    try:
        if model_type == "arima":
            fc = ARIMAForecaster(settings.ARIMA_ORDER)
            fc.fit(series)
            preds, lower, upper = fc.predict(horizon)
            return build_forecast_response(payload.well_id, "arima", start_date, preds, lower, upper)

        elif model_type == "lstm":
            fc = LSTMForecaster(settings.LSTM_SEQ_LENGTH, settings.LSTM_EPOCHS, settings.LSTM_BATCH_SIZE)
            split = int(0.8 * len(X_train))
            fc.fit(X_train[:split], y_train[:split], X_train[split:], y_train[split:])
            last_seq = X_train[-1]
            preds = fc.predict_multi_step(last_seq, horizon, scaler)
            return build_forecast_response(payload.well_id, "lstm", start_date, preds)

        elif model_type == "ensemble":
            arima = ARIMAForecaster(settings.ARIMA_ORDER).fit(series)
            arima_preds, lower, upper = arima.predict(horizon)
            lstm = LSTMForecaster(settings.LSTM_SEQ_LENGTH, settings.LSTM_EPOCHS, settings.LSTM_BATCH_SIZE)
            lstm.fit(X_train, y_train)
            lstm_preds = lstm.predict_multi_step(X_train[-1], horizon, scaler)
            ensemble = ensemble_forecast(arima_preds, lstm_preds)
            return build_forecast_response(payload.well_id, "ensemble", start_date, ensemble, lower, upper)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/wells/{well_id}/analytics/trend", tags=["Analytics"])
async def trend_analysis(well_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == well_id)
        .order_by(WaterLevelReading.recorded_at)
    )
    readings = [{"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m}
                for r in result.scalars().all()]
    if not readings:
        raise HTTPException(404, "No readings found.")
    df = load_readings_to_df(readings)
    df = resample_to_daily(df)
    series = df["depth_to_water_m"]
    return {
        "well_id": well_id,
        "trend": compute_linear_trend(series),
        "summary": compute_summary_statistics(series),
        "monsoon": monsoon_recharge_analysis(df),
        "anomalies": detect_anomalies_zscore(series),
    }


@app.get("/api/v1/wells/{well_id}/analytics/seasonal", tags=["Analytics"])
async def seasonal_analysis(well_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == well_id)
        .order_by(WaterLevelReading.recorded_at)
    )
    readings = [{"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m}
                for r in result.scalars().all()]
    if len(readings) < 730:
        raise HTTPException(422, "Need at least 2 years of data for seasonal decomposition.")
    df = load_readings_to_df(readings)
    df = resample_to_daily(df)
    decomp = seasonal_decomposition(df["depth_to_water_m"])
    return {"well_id": well_id, "decomposition": decomp}


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/alerts", tags=["Alerts"])
async def list_alerts(
    well_id: Optional[int] = None,
    unacknowledged_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    q = select(Alert)
    if well_id:
        q = q.where(Alert.well_id == well_id)
    if unacknowledged_only:
        q = q.where(Alert.is_acknowledged == False)
    q = q.order_by(Alert.triggered_at.desc())
    result = await db.execute(q)
    alerts = result.scalars().all()
    return {
        "summary": build_alert_summary(alerts),
        "alerts": [
            {"id": a.id, "well_id": a.well_id, "severity": a.severity,
             "message": a.message, "triggered_at": a.triggered_at.isoformat(),
             "is_acknowledged": a.is_acknowledged}
            for a in alerts
        ],
    }


@app.patch("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.is_acknowledged = True
    await db.commit()
    return {"message": "Alert acknowledged.", "alert_id": alert_id}
