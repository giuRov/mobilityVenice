#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def _coerce_key(s: pd.Series) -> pd.Series:
    """
    Coerce join keys to a consistent dtype.
    We use pandas nullable Int64 if possible, otherwise fall back to string.
    """
    # Try numeric
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().any() and numeric.isna().sum() < len(s):
        # Mixed but mostly numeric -> use Int64 (keeps NaNs)
        return numeric.astype("Int64")
    # Otherwise, keep as string
    return s.astype("string")


def build_unified_stops(
    stops_water: pd.DataFrame,
    land_key_areas: pd.DataFrame,
    stops_land_mapped: pd.DataFrame,
    *,
    stop_col: str = "stop",
    name_col: str = "stop_name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a unified stop table where:
    - water stops remain unchanged
    - land stops are replaced by key areas

    Returns:
    - stops_unified: [stop, stop_name, latitude, longitude]
    - stop_id_map: dict {old_land_stop -> key_area_stop}
    """
    # Build mapping old_stop -> stop_mapping
    mapping_df = stops_land_mapped.dropna(subset=[stop_col, "stop_mapping"]).copy()
    mapping_df[stop_col] = _coerce_key(mapping_df[stop_col])
    mapping_df["stop_mapping"] = _coerce_key(mapping_df["stop_mapping"])

    stop_id_map = mapping_df.set_index(stop_col)["stop_mapping"].to_dict()

    # Prepare stop tables (coerce keys)
    water = stops_water[[stop_col, name_col, lat_col, lon_col]].copy()
    keyareas = land_key_areas[[stop_col, name_col, lat_col, lon_col]].copy()

    water[stop_col] = _coerce_key(water[stop_col])
    keyareas[stop_col] = _coerce_key(keyareas[stop_col])

    # Unified table = water + key areas
    stops_unified = (
        pd.concat([water, keyareas], ignore_index=True)
        .drop_duplicates(subset=[stop_col], keep="first")
        .reset_index(drop=True)
    )

    return stops_unified, stop_id_map


def apply_stop_mapping_and_add_stop_info(
    validation_data: pd.DataFrame,
    stops_unified: pd.DataFrame,
    stop_id_map: Dict,
    *,
    stop_col: str = "stop",
    name_col: str = "stop_name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    out_lat_col: str = "stop_latitude",
    out_lon_col: str = "stop_longitude",
    return_report: bool = False,
):
    """
    Apply land->key-area stop mapping and enrich with stop metadata.
    """
    out = validation_data.copy()

    # Coerce stop key for reliable mapping/merge
    out[stop_col] = _coerce_key(out[stop_col])

    # Replace land stop IDs; water stops remain unchanged
    mapped = out[stop_col].map(stop_id_map)
    out[stop_col] = mapped.fillna(out[stop_col])

    # Prepare metadata table (ensure unique keys)
    stop_meta = stops_unified[[stop_col, name_col, lat_col, lon_col]].copy()
    stop_meta[stop_col] = _coerce_key(stop_meta[stop_col])
    stop_meta = stop_meta.drop_duplicates(subset=[stop_col], keep="first")

    stop_meta = stop_meta.rename(columns={lat_col: out_lat_col, lon_col: out_lon_col})

    out = out.merge(stop_meta, on=stop_col, how="left", validate="m:1")

    if not return_report:
        return out

    report = {
        "n_rows": len(out),
        "n_unique_stops_after_mapping": out[stop_col].nunique(dropna=True),
        "n_missing_stop_name": int(out[name_col].isna().sum()) if name_col in out.columns else None,
        "n_missing_stop_lat": int(out[out_lat_col].isna().sum()),
        "n_missing_stop_lon": int(out[out_lon_col].isna().sum()),
    }
    return out, report
