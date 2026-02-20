"""
data_ingestion.py — CGWB/DWLR Data Loading & Real-Time Simulation
Groundwater Level Prediction System

Supports:
  1. Batch loading from CSV / API
  2. Simulated streaming (periodic synthetic readings)
  3. Scheduled background refresh via APScheduler
"""

import asyncio
import random
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from app.config import settings
from app.models.db_models import Well, WaterLevelReading


# ── CSV / File Ingestion ──────────────────────────────────────────────────────
async def ingest_from_csv(filepath: str, db: AsyncSession) -> int:
    """
    Load historical readings from a CGWB-formatted CSV file.

    Expected CSV columns:
        station_code, recorded_at, depth_to_water_m, rainfall_mm, temperature_c
    """
    df = pd.read_csv(filepath, parse_dates=["recorded_at"])
    df = df.dropna(subset=["station_code", "recorded_at", "depth_to_water_m"])

    inserted = 0
    for _, row in df.iterrows():
        # Look up well
        result = await db.execute(
            select(Well).where(Well.station_code == str(row["station_code"]))
        )
        well = result.scalar_one_or_none()
        if not well:
            logger.warning(f"Station {row['station_code']} not found — skipping.")
            continue

        reading = WaterLevelReading(
            well_id=well.id,
            recorded_at=row["recorded_at"],
            depth_to_water_m=float(row["depth_to_water_m"]),
            rainfall_mm=row.get("rainfall_mm", None),
            temperature_c=row.get("temperature_c", None),
            source="CGWB",
        )
        db.add(reading)
        inserted += 1

    await db.commit()
    logger.info(f"CSV ingestion complete: {inserted} records inserted.")
    return inserted


# ── CGWB REST API Ingestion ───────────────────────────────────────────────────
async def ingest_from_api(well_id: int, station_code: str, db: AsyncSession) -> int:
    """
    Fetch latest readings from CGWB/DWLR API endpoint (if configured).
    Falls back to simulated data when CGWB_API_URL is not set.
    """
    if not settings.CGWB_API_URL:
        logger.info("CGWB_API_URL not configured → using simulated data.")
        return await _simulate_api_response(well_id, db)

    url = f"{settings.CGWB_API_URL}/readings/{station_code}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            records = resp.json().get("data", [])

        inserted = 0
        for rec in records:
            reading = WaterLevelReading(
                well_id=well_id,
                recorded_at=datetime.fromisoformat(rec["timestamp"]),
                depth_to_water_m=rec["level_m"],
                rainfall_mm=rec.get("rainfall_mm"),
                source="CGWB",
            )
            db.add(reading)
            inserted += 1
        await db.commit()
        return inserted

    except httpx.HTTPError as e:
        logger.error(f"API fetch failed for {station_code}: {e}")
        return 0


# ── Simulated Real-Time Streaming ─────────────────────────────────────────────
async def simulated_stream(
    well_id: int,
    base_level: float = 8.0,
    noise_std: float = 0.3,
    seasonal_amplitude: float = 2.0,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields synthetic groundwater readings every
    SIMULATED_STREAM_INTERVAL_SECONDS seconds.

    Simulates:
        - Seasonal trend (annual sinusoidal cycle)
        - Random Gaussian noise
        - Occasional anomalous spikes (drought simulation)
    """
    interval = settings.SIMULATED_STREAM_INTERVAL_SECONDS
    day_counter = 0

    while True:
        # Seasonal component (annual sine wave)
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * day_counter / 365)
        # Random noise
        noise = random.gauss(0, noise_std)
        # Occasional drought spike
        drought_factor = random.choices([0, random.uniform(2, 5)], weights=[0.97, 0.03])[0]

        level = base_level + seasonal + noise + drought_factor
        level = max(0.5, level)  # Physical floor

        reading = {
            "well_id": well_id,
            "recorded_at": datetime.utcnow().isoformat(),
            "depth_to_water_m": round(level, 3),
            "rainfall_mm": round(max(0, random.gauss(2, 3)), 2),
            "temperature_c": round(28 + 5 * np.sin(2 * np.pi * day_counter / 365), 1),
            "source": "simulated",
        }

        yield reading
        day_counter += 1
        await asyncio.sleep(interval)


async def _simulate_api_response(well_id: int, db: AsyncSession, n: int = 10) -> int:
    """Insert n synthetic readings for a well (fallback when API unavailable)."""
    base = 8.0
    inserted = 0
    for i in range(n):
        ts = datetime.utcnow() - timedelta(hours=n - i)
        level = base + random.gauss(0, 0.5)
        db.add(WaterLevelReading(
            well_id=well_id,
            recorded_at=ts,
            depth_to_water_m=round(max(0.5, level), 3),
            source="simulated",
        ))
        inserted += 1
    await db.commit()
    return inserted


# ── Scheduled Batch Refresh ───────────────────────────────────────────────────
async def refresh_all_wells(db: AsyncSession) -> None:
    """
    Called by APScheduler every DATA_REFRESH_INTERVAL_MINUTES.
    Iterates all active wells and triggers API / simulated ingestion.
    """
    result = await db.execute(select(Well))
    wells = result.scalars().all()
    logger.info(f"Scheduled refresh: processing {len(wells)} wells …")

    for well in wells:
        await ingest_from_api(well.id, well.station_code, db)

    logger.info("Scheduled refresh complete.")
