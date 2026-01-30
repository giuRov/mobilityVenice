#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import pandas as pd
from .stops import build_unified_stops, apply_stop_mapping_and_add_stop_info


def load_dataset(
    path: Path,
    *,
    stops_water_path: Path,
    stops_land_mapped_path: Path,
    land_key_areas_path: Path,
) -> pd.DataFrame:
    """
    Load validations dataset and enrich it with stop metadata using external stop tables.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load validations
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["validation_datetime"], low_memory=False)
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")

    # Only require what the validations file should contain
    required_base = {"validation_datetime", "ticket_class", "stop"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(
            "Input dataset is missing required column(s): "
            + ", ".join(sorted(missing))
            + f"\nFound columns: {df.columns.tolist()}"
        )

    # Load stop reference tables from user-provided paths
    for p in [stops_water_path, stops_land_mapped_path, land_key_areas_path]:
        if not p.exists():
            raise FileNotFoundError(f"Stop reference file not found: {p}")

    stops_water = pd.read_csv(stops_water_path)
    stops_land_mapped = pd.read_csv(stops_land_mapped_path)
    land_key_areas = pd.read_csv(land_key_areas_path)

    # Build unified stops + mapping
    stops_unified, stop_id_map = build_unified_stops(
        stops_water=stops_water,
        land_key_areas=land_key_areas,
        stops_land_mapped=stops_land_mapped,
    )

    # Apply mapping and join metadata
    df = apply_stop_mapping_and_add_stop_info(
        validation_data=df,
        stops_unified=stops_unified,
        stop_id_map=stop_id_map,
    )

    # Final strict check (after enrichment)
    required_final = {
        "validation_datetime",
        "ticket_class",
        "stop",
        "stop_name",
        "stop_latitude",
        "stop_longitude",
    }
    missing_final = required_final - set(df.columns)
    if missing_final:
        raise ValueError(
            "Dataset enrichment failed. Missing required column(s): "
            + ", ".join(sorted(missing_final))
            + f"\nFound columns: {df.columns.tolist()}"
        )

    # Clean / types
    df["validation_datetime"] = pd.to_datetime(df["validation_datetime"], errors="coerce")
    df = df.dropna(subset=["validation_datetime"]).copy()
    df["ticket_class"] = df["ticket_class"].astype("category")
    df["date"] = df["validation_datetime"].dt.normalize()

    return df


def build_disabled_days(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    """
    Build DatePickerRange bounds and disabled days list.

    Returns:
      - min_day_str: "YYYY-MM-DD"
      - max_day_str: "YYYY-MM-DD"
      - disabled_days_str: ["YYYY-MM-DD", ...]
    """
    min_day = df["date"].min()
    max_day = df["date"].max()

    all_days = pd.date_range(min_day, max_day, freq="D")
    available = set(df["date"].unique())
    disabled = sorted(set(all_days) - available)

    min_day_str = min_day.date().isoformat()
    max_day_str = max_day.date().isoformat()
    disabled_days_str = [d.date().isoformat() for d in disabled]

    return min_day_str, max_day_str, disabled_days_str


def filter_df(df: pd.DataFrame, start_date: str, end_date: str, ticket_values: list[str]) -> pd.DataFrame:
    """
    Filter dataframe by ticket_class and date interval.

    start_date/end_date come from Dash DatePickerRange and are ISO strings.
    """
    if not ticket_values:
        return df.iloc[0:0]

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    # Guard: avoid empty/invalid ranges
    if start >= end:
        return df.iloc[0:0]

    out = df[df["ticket_class"].isin(ticket_values)]
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out
