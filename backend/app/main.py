from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
"""
main.py — FastAPI Application (Updated for CGWB 555k dataset)
New endpoints:
  POST /api/v1/ingestion/cgwb-csv   → load your full dataset
  GET  /api/v1/wells/search         → search by state / district / basin
  GET  /api/v1/analytics/regional   → state/basin-level summary
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from datetime import date, datetime, timedelta
from typing import List, Optional
import asyncio, json, os, shutil, tempfile

from app.config import settings
from app.database import get_db, init_db, close_db
from app.models.db_models import Well, WaterLevelReading, Prediction, Alert
from app.models.user_model import User

from app.data_ingestion import (
    ingest_cgwb_dataset, simulated_stream, refresh_all_wells,
)
from app.preprocessing import (
    full_pipeline, load_readings_to_df, resample_to_monthly,
    summarise_by_group,
)
from app.prediction import (
    ARIMAForecaster, LSTMForecaster, ensemble_forecast, build_forecast_response,
)
from app.analytics import (
    compute_linear_trend, seasonal_decomposition,
    monsoon_recharge_analysis, detect_anomalies_zscore,
    compute_summary_statistics,
)
from app.alerts import check_and_create_alerts, build_alert_summary
from app.utils import (
    WellCreate, ReadingCreate, ForecastRequest,
    UserCreate, TokenResponse,
    hash_password, verify_password, create_access_token, paginate,
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

    async def _scheduled_refresh():
        async for session in get_db():
            await refresh_all_wells(session)

    scheduler.add_job(
        _scheduled_refresh,
        trigger=IntervalTrigger(minutes=settings.DATA_REFRESH_INTERVAL_MINUTES),
        id="data_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started.")
    yield
    scheduler.shutdown(wait=False)
    await close_db()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Groundwater Level Prediction System — "
        "CGWB 555k dataset · ARIMA + BiLSTM · PostGIS · Near Real-Time"
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
    well_count    = await db.scalar(select(func.count(Well.id)))
    reading_count = await db.scalar(select(func.count(WaterLevelReading.id)))
    return {
        "status":         "healthy",
        "timestamp":      datetime.utcnow().isoformat(),
        "version":        settings.APP_VERSION,
        "wells_in_db":    well_count,
        "readings_in_db": reading_count,
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
        username=payload.username, email=payload.email,
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
# CGWB DATASET INGESTION  ← KEY NEW ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/ingestion/cgwb-csv", tags=["Ingestion"])
async def ingest_cgwb_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload your CGWB CSV file (555k rows) and ingest it into PostgreSQL.
    Runs as a background task — returns immediately with a job confirmation.

    Columns expected:
      id, date (DD-MM-YYYY), state_name, state_code, district_name,
      district_code, station_name, latitude, longitude, basin,
      sub_basin, source, currentlevel, level_diff
    """
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    tmp_path = tmp.name

    async def _run():
        try:
            result = await ingest_cgwb_dataset(tmp_path, db)
            logger.info(f"CGWB ingestion finished: {result}")
        finally:
            os.unlink(tmp_path)

    background_tasks.add_task(_run)
    return {
        "message":  "CGWB CSV ingestion started in background.",
        "filename": file.filename,
        "note":     "Check /health to monitor wells_in_db and readings_in_db counts.",
    }


@app.post("/api/v1/ingestion/cgwb-csv-path", tags=["Ingestion"])
async def ingest_cgwb_csv_by_path(
    filepath: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    If your CSV is already on the server (e.g. uploaded to Codespaces),
    pass the file path directly — no upload needed.
    Example: filepath=/home/user/groundwater/cgwb_data.csv
    """
    if not os.path.exists(filepath):
        raise HTTPException(404, f"File not found: {filepath}")

    async def _run():
        result = await ingest_cgwb_dataset(filepath, db)
        logger.info(f"CGWB ingestion finished: {result}")

    background_tasks.add_task(_run)
    return {
        "message":  "CGWB CSV ingestion started in background.",
        "filepath": filepath,
        "note":     "Monitor progress at /health endpoint.",
    }


@app.post("/api/v1/ingestion/reading", tags=["Ingestion"], status_code=201)
async def add_single_reading(
    payload: ReadingCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a single reading manually (useful for testing)."""
    reading = WaterLevelReading(
        well_id=payload.well_id,
        recorded_at=payload.recorded_at,
        depth_to_water_m=payload.depth_to_water_m,
        rainfall_mm=payload.rainfall_mm,
        temperature_c=payload.temperature_c,
        source="manual",
    )
    db.add(reading)
    await db.commit()
    return {"message": "Reading added."}


# ── Real-time simulation stream ───────────────────────────────────────────────
@app.get("/api/v1/ingestion/stream/{well_id}", tags=["Ingestion"])
async def stream_readings(
    well_id: int,
    duration_seconds: int = Query(30, le=300),
):
    """
    Server-Sent Events stream simulating real-time CGWB sensor readings.
    Returns readings in the same schema as your CGWB dataset.
    """
    async def event_generator():
        count = 0
        max_r = duration_seconds // settings.SIMULATED_STREAM_INTERVAL_SECONDS
        async for reading in simulated_stream(well_id):
            yield f"data: {json.dumps(reading)}\n\n"
            count += 1
            if count >= max_r:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    return {"id": well.id, "station_code": well.station_code}


@app.get("/api/v1/wells", tags=["Wells"])
async def list_wells(
    state:    Optional[str] = None,
    district: Optional[str] = None,
    basin:    Optional[str] = None,      # ← new filter using your basin column
    page:     int = Query(1,   ge=1),
    page_size:int = Query(50,  ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    q = select(Well)
    if state:
        q = q.where(Well.state.ilike(f"%{state}%"))
    if district:
        q = q.where(Well.district.ilike(f"%{district}%"))
    if basin:
        q = q.where(Well.basin.ilike(f"%{basin}%"))
    result = await db.execute(q)
    wells  = result.scalars().all()
    data   = [
        {
            "id":           w.id,
            "station_name": w.station_name,
            "station_code": w.station_code,
            "state":        w.state,
            "district":     w.district,
            "basin":        w.basin,
            "sub_basin":    w.sub_basin,
        }
        for w in wells
    ]
    return paginate(data, page, page_size)


@app.get("/api/v1/wells/search", tags=["Wells"])
async def search_wells(
    q:        str = Query(..., min_length=2, description="Search station name / state / district / basin"),
    page:     int = Query(1,  ge=1),
    page_size:int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Full-text search across station_name, state, district, basin."""
    query = select(Well).where(
        Well.station_name.ilike(f"%{q}%") |
        Well.state.ilike(f"%{q}%")        |
        Well.district.ilike(f"%{q}%")     |
        Well.basin.ilike(f"%{q}%")
    )
    result = await db.execute(query)
    wells  = result.scalars().all()
    data   = [{"id": w.id, "station_name": w.station_name,
               "state": w.state, "district": w.district, "basin": w.basin}
              for w in wells]
    return paginate(data, page, page_size)


@app.get("/api/v1/wells/{well_id}", tags=["Wells"])
async def get_well(well_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Well).where(Well.id == well_id))
    well   = result.scalar_one_or_none()
    if not well:
        raise HTTPException(404, "Well not found.")
    return {
        "id":           well.id,
        "station_code": well.station_code,
        "station_name": well.station_name,
        "state":        well.state,
        "state_code":   well.state_code,
        "district":     well.district,
        "district_code":well.district_code,
        "basin":        well.basin,
        "sub_basin":    well.sub_basin,
        "well_depth_m": well.well_depth_m,
        "aquifer_type": well.aquifer_type,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# READINGS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/wells/{well_id}/readings", tags=["Readings"])
async def get_readings(
    well_id:    int,
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
    page:       int = Query(1,   ge=1),
    page_size:  int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    q = select(WaterLevelReading).where(WaterLevelReading.well_id == well_id)
    if start_date:
        q = q.where(WaterLevelReading.recorded_at >= datetime.combine(start_date, datetime.min.time()))
    if end_date:
        q = q.where(WaterLevelReading.recorded_at <= datetime.combine(end_date,   datetime.max.time()))
    q      = q.order_by(WaterLevelReading.recorded_at.asc())
    result = await db.execute(q)
    rds    = result.scalars().all()
    data   = [
        {
            "date":             r.recorded_at.strftime("%d-%m-%Y"),   # match CGWB format
            "recorded_at":      r.recorded_at.isoformat(),
            "currentlevel":     r.depth_to_water_m,                  # use CGWB name in response
            "level_diff":       r.level_diff,
            "source":           r.source,
        }
        for r in rds
    ]
    return paginate(data, page, page_size)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/api/v1/predict", tags=["Prediction"])
async def predict(payload: ForecastRequest, db: AsyncSession = Depends(get_db)):
    """
    Run ARIMA / LSTM / Ensemble forecast for a well.
    Uses up to 5 years of historical CGWB data for training.
    With bi-annual CGWB data → resampled to monthly → ~60 data points per year.
    """
    # Fetch up to 5 years of readings
    cutoff = datetime.utcnow() - timedelta(days=5 * 365)
    q = (
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == payload.well_id)
        .where(WaterLevelReading.recorded_at >= cutoff)
        .order_by(WaterLevelReading.recorded_at.asc())
    )
    result   = await db.execute(q)
    readings = [
        {
            "recorded_at":     r.recorded_at,
            "depth_to_water_m":r.depth_to_water_m,
            "level_diff":      r.level_diff,
        }
        for r in result.scalars().all()
    ]

    min_required = settings.LSTM_SEQ_LENGTH + 5
    if len(readings) < min_required:
        raise HTTPException(
            422,
            f"Not enough data for well {payload.well_id}. "
            f"Found {len(readings)} readings, need at least {min_required}. "
            "Try ingesting the CGWB CSV first via POST /api/v1/ingestion/cgwb-csv-path"
        )

    # Preprocess
    df, X_train, y_train, scaler = full_pipeline(readings, settings.LSTM_SEQ_LENGTH)
    series     = df["depth_to_water_m"].values
    start_date = date.today()
    horizon    = payload.horizon_days
    model_type = payload.model_type

    try:
        if model_type == "arima":
            fc = ARIMAForecaster(settings.ARIMA_ORDER)
            fc.fit(series)
            preds, lower, upper = fc.predict(horizon)
            return build_forecast_response(payload.well_id, "arima", start_date, preds, lower, upper)

        elif model_type == "lstm":
            fc = LSTMForecaster(settings.LSTM_SEQ_LENGTH, settings.LSTM_EPOCHS, settings.LSTM_BATCH_SIZE)
            split = int(0.8 * len(X_train))
            fc.fit(X_train[:split], y_train[:split],
                   X_train[split:],  y_train[split:])
            preds = fc.predict_multi_step(X_train[-1], horizon, scaler)
            return build_forecast_response(payload.well_id, "lstm", start_date, preds)

        elif model_type == "ensemble":
            arima = ARIMAForecaster(settings.ARIMA_ORDER).fit(series)
            arima_preds, lower, upper = arima.predict(horizon)
            lstm  = LSTMForecaster(settings.LSTM_SEQ_LENGTH, settings.LSTM_EPOCHS, settings.LSTM_BATCH_SIZE)
            lstm.fit(X_train, y_train)
            lstm_preds = lstm.predict_multi_step(X_train[-1], horizon, scaler)
            combined   = ensemble_forecast(arima_preds, lstm_preds)
            return build_forecast_response(payload.well_id, "ensemble", start_date, combined, lower, upper)

    except Exception as e:
        logger.error(f"Prediction error for well {payload.well_id}: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/wells/{well_id}/analytics/trend", tags=["Analytics"])
async def trend_analysis(well_id: int, db: AsyncSession = Depends(get_db)):
    result   = await db.execute(
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == well_id)
        .order_by(WaterLevelReading.recorded_at.asc())
    )
    readings = [{"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m,
                 "level_diff": r.level_diff}
                for r in result.scalars().all()]
    if not readings:
        raise HTTPException(404, "No readings found for this well.")

    df     = load_readings_to_df(readings)
    df     = resample_to_monthly(df)
    series = df["depth_to_water_m"]

    return {
        "well_id":  well_id,
        "trend":    compute_linear_trend(series),
        "summary":  compute_summary_statistics(series),
        "monsoon":  monsoon_recharge_analysis(df),
        "anomalies":detect_anomalies_zscore(series),
    }


@app.get("/api/v1/analytics/regional", tags=["Analytics"])
async def regional_analytics(
    group_by: str = Query("state", enum=["state", "district", "basin", "sub_basin"]),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate mean/min/max groundwater levels across all wells grouped by
    state, district, basin, or sub_basin.
    Gives a national/regional picture of your 555k dataset.
    """
    # Join wells + latest reading per well
    q = text("""
        SELECT
            w.state       AS state,
            w.district    AS district,
            w.basin       AS basin,
            w.sub_basin   AS sub_basin,
            AVG(r.depth_to_water_m) AS mean_level,
            MIN(r.depth_to_water_m) AS min_level,
            MAX(r.depth_to_water_m) AS max_level,
            COUNT(r.id)             AS reading_count
        FROM wells w
        JOIN water_level_readings r ON r.well_id = w.id
        GROUP BY w.state, w.district, w.basin, w.sub_basin
        ORDER BY mean_level DESC
    """)
    result = await db.execute(q)
    rows   = result.mappings().all()

    # Group by requested column
    grouped = {}
    for row in rows:
        key = row.get(group_by) or "Unknown"
        if key not in grouped:
            grouped[key] = {"mean_levels": [], "reading_count": 0}
        grouped[key]["mean_levels"].append(float(row["mean_level"] or 0))
        grouped[key]["reading_count"] += int(row["reading_count"] or 0)

    summary = [
        {
            group_by:        k,
            "avg_level_m":   round(np.mean(v["mean_levels"]), 3),
            "reading_count": v["reading_count"],
        }
        for k, v in sorted(grouped.items(), key=lambda x: -x[1]["reading_count"])
    ]
    return {"group_by": group_by, "data": summary}


@app.get("/api/v1/analytics/state-summary", tags=["Analytics"])
async def state_summary(
    state_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Deep summary for a specific state — all wells, trend, avg level.
    Maps directly to your state_name column.
    """
    q = (
        select(Well)
        .where(Well.state.ilike(f"%{state_name}%"))
    )
    result = await db.execute(q)
    wells  = result.scalars().all()
    if not wells:
        raise HTTPException(404, f"No wells found for state: {state_name}")

    well_ids = [w.id for w in wells]
    rq = (
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id.in_(well_ids))
        .order_by(WaterLevelReading.recorded_at.asc())
    )
    rresult  = await db.execute(rq)
    readings = [{"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m}
                for r in rresult.scalars().all()]

    if not readings:
        raise HTTPException(404, f"No readings found for state: {state_name}")

    df     = load_readings_to_df(readings)
    df     = resample_to_monthly(df)
    series = df["depth_to_water_m"]

    return {
        "state":        state_name,
        "well_count":   len(wells),
        "reading_count":len(readings),
        "summary":      compute_summary_statistics(series),
        "trend":        compute_linear_trend(series),
        "monsoon":      monsoon_recharge_analysis(df),
    }


@app.get("/api/v1/wells/{well_id}/analytics/seasonal", tags=["Analytics"])
async def seasonal_analysis(well_id: int, db: AsyncSession = Depends(get_db)):
    result   = await db.execute(
        select(WaterLevelReading)
        .where(WaterLevelReading.well_id == well_id)
        .order_by(WaterLevelReading.recorded_at.asc())
    )
    readings = [{"recorded_at": r.recorded_at, "depth_to_water_m": r.depth_to_water_m}
                for r in result.scalars().all()]

    if len(readings) < 24:
        raise HTTPException(422, "Need at least 2 years of data for seasonal decomposition.")

    df     = load_readings_to_df(readings)
    df     = resample_to_monthly(df)
    decomp = seasonal_decomposition(df["depth_to_water_m"], period=12)
    return {"well_id": well_id, "decomposition": decomp}


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/v1/alerts", tags=["Alerts"])
async def list_alerts(
    well_id:            Optional[int]  = None,
    unacknowledged_only:bool           = False,
    db: AsyncSession = Depends(get_db),
):
    q = select(Alert)
    if well_id:
        q = q.where(Alert.well_id == well_id)
    if unacknowledged_only:
        q = q.where(Alert.is_acknowledged == False)
    q      = q.order_by(Alert.triggered_at.desc())
    result = await db.execute(q)
    alerts = result.scalars().all()
    return {
        "summary": build_alert_summary(alerts),
        "alerts": [
            {
                "id":              a.id,
                "well_id":         a.well_id,
                "severity":        a.severity,
                "message":         a.message,
                "triggered_at":    a.triggered_at.isoformat(),
                "is_acknowledged": a.is_acknowledged,
            }
            for a in alerts
        ],
    }


@app.patch("/api/v1/alerts/{alert_id}/acknowledge", tags=["Alerts"])
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert  = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert.is_acknowledged = True
    await db.commit()
    return {"message": "Alert acknowledged.", "alert_id": alert_id}
@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse("app/static/index.html")
