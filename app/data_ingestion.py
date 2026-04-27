"""
data_ingestion.py — CGWB Dataset Ingestion (SQLite compatible)
Columns: id, date, state_name, state_code, district_name, district_code,
         station_name, latitude, longitude, basin, sub_basin, source,
         currentlevel, level_diff
"""

import asyncio
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from loguru import logger

from app.config import settings
from app.models.db_models import Well, WaterLevelReading


def parse_cgwb_date(date_str: str) -> Optional[datetime]:
    for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    logger.warning(f"Could not parse date: {date_str}")
    return None


def load_cgwb_csv(filepath: str, chunksize: int = 50_000) -> pd.DataFrame:
    logger.info(f"Loading CGWB CSV: {filepath}")
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
        chunk["parsed_date"] = chunk["date"].apply(parse_cgwb_date)
        chunk = chunk.dropna(subset=["parsed_date", "currentlevel"])
        chunk = chunk[chunk["currentlevel"].between(-5, 300)]
        chunk["currentlevel"] = pd.to_numeric(chunk["currentlevel"], errors="coerce")
        chunk["level_diff"]   = pd.to_numeric(chunk["level_diff"],   errors="coerce")
        chunk["latitude"]     = pd.to_numeric(chunk["latitude"],     errors="coerce")
        chunk["longitude"]    = pd.to_numeric(chunk["longitude"],    errors="coerce")
        for col in ["state_name", "district_name", "station_name", "basin", "sub_basin", "source"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip()
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("parsed_date").reset_index(drop=True)
    logger.info(f"CSV loaded: {len(df):,} valid rows")
    return df


async def register_wells_from_df(df: pd.DataFrame, db: AsyncSession) -> dict:
    stations = (
        df.groupby("station_name")
        .agg(
            state_name    = ("state_name",    "first"),
            state_code    = ("state_code",    "first"),
            district_name = ("district_name", "first"),
            district_code = ("district_code", "first"),
            latitude      = ("latitude",      "first"),
            longitude     = ("longitude",     "first"),
            basin         = ("basin",         "first"),
            sub_basin     = ("sub_basin",     "first"),
        )
        .reset_index()
    )

    station_id_map = {}
    new_count = 0

    for _, row in stations.iterrows():
        name   = str(row["station_name"])
        result = await db.execute(select(Well).where(Well.station_name == name))
        well   = result.scalar_one_or_none()

        if not well:
            well = Well(
                station_code  = name[:100],
                station_name  = name,
                state         = str(row["state_name"]),
                state_code    = int(row["state_code"])    if pd.notna(row["state_code"])    else None,
                district      = str(row["district_name"]),
                district_code = int(row["district_code"]) if pd.notna(row["district_code"]) else None,
                basin         = str(row["basin"])         if pd.notna(row["basin"])         else None,
                sub_basin     = str(row["sub_basin"])     if pd.notna(row["sub_basin"])     else None,
                latitude      = float(row["latitude"])    if pd.notna(row["latitude"])      else None,
                longitude     = float(row["longitude"])   if pd.notna(row["longitude"])     else None,
            )
            db.add(well)
            await db.flush()
            new_count += 1

        station_id_map[name] = well.id

    await db.commit()
    logger.info(f"Wells registered: {new_count} new, {len(station_id_map)} total.")
    return station_id_map


async def bulk_insert_readings(
    df: pd.DataFrame,
    station_id_map: dict,
    db: AsyncSession,
    batch_size: int = 1_000,
) -> int:
    total_inserted = 0
    rows_buffer    = []

    for _, row in df.iterrows():
        station = str(row["station_name"])
        well_id = station_id_map.get(station)
        if not well_id:
            continue

        rows_buffer.append(WaterLevelReading(
            well_id          = well_id,
            recorded_at      = row["parsed_date"],
            depth_to_water_m = float(row["currentlevel"]),
            level_diff       = float(row["level_diff"]) if pd.notna(row.get("level_diff")) else None,
            source           = str(row.get("source", "CGWB")),
        ))

        if len(rows_buffer) >= batch_size:
            for r in rows_buffer:
                try:
                    db.add(r)
                    await db.flush()
                except Exception:
                    await db.rollback()
            await db.commit()
            total_inserted += len(rows_buffer)
            logger.info(f"Inserted {total_inserted:,} readings so far...")
            rows_buffer.clear()

    if rows_buffer:
        for r in rows_buffer:
            try:
                db.add(r)
                await db.flush()
            except Exception:
                await db.rollback()
        await db.commit()
        total_inserted += len(rows_buffer)

    logger.info(f"Bulk insert complete: {total_inserted:,} total readings inserted.")
    return total_inserted


async def ingest_cgwb_dataset(filepath: str, db: AsyncSession) -> dict:
    df             = load_cgwb_csv(filepath)
    station_id_map = await register_wells_from_df(df, db)
    readings_count = await bulk_insert_readings(df, station_id_map, db)
    return {
        "status":            "success",
        "total_rows_loaded": len(df),
        "wells_registered":  len(station_id_map),
        "readings_inserted": readings_count,
    }


async def simulated_stream(
    well_id: int,
    base_level: float = 8.0,
    noise_std: float = 0.3,
    seasonal_amplitude: float = 2.0,
) -> AsyncGenerator[dict, None]:
    import json
    day_counter = 0
    while True:
        seasonal   = seasonal_amplitude * np.sin(2 * np.pi * day_counter / 365)
        noise      = random.gauss(0, noise_std)
        drought    = random.choices([0, random.uniform(2, 5)], weights=[0.97, 0.03])[0]
        level      = max(0.1, base_level + seasonal + noise + drought)
        yield {
            "well_id":      well_id,
            "date":         datetime.utcnow().strftime("%d-%m-%Y"),
            "recorded_at":  datetime.utcnow().isoformat(),
            "currentlevel": round(level, 3),
            "level_diff":   round(random.gauss(0, 0.4), 3),
            "source":       "simulated",
        }
        day_counter += 1
        await asyncio.sleep(settings.SIMULATED_STREAM_INTERVAL_SECONDS)


async def refresh_all_wells(db: AsyncSession) -> None:
    logger.info("Scheduled refresh triggered.")
