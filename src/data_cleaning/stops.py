from typing import Dict, Tuple
import pandas as pd

def build_unified_stops(
    stops_water: pd.DataFrame,
    stops_land: pd.DataFrame,  # kept for QA only
    land_key_areas: pd.DataFrame,
    stops_land_mapped: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a unified stop table where:
    - water stops remain unchanged
    - land stops are replaced by key areas

    Expected columns:
    - stops_water:       ['stop_id', 'stop_name', 'stop_lat', 'stop_long']
    - stops_land:        ['stop_id', 'stop_name', 'stop_lat', 'stop_long']  (not strictly required here)
    - land_key_areas:    ['area_id', 'area_name', 'area_lat', 'area_long']  (stop_id = key area id)
    - stops_land_mapped: ['stop_id', 'area_id']                             (original land stop -> key area stop)

    Returns:
    - stops_unified: ['stop_id ', 'stop_name', 'stop_lat', 'stop_long']
    - stop_id_map: dict {old_land_stop -> key_area_stop}
    """

    # Mapping: original land stop -> key area
    stop_id_map = (
        stops_land_mapped.dropna(subset=["stop_id", "area_id"])
        .set_index("stop_id")["area_id"]
        .to_dict()
    )

    # Convert key areas to "stop schema"
    key_areas_as_stops = land_key_areas.rename(columns={
        "area_id": "stop_id",
        "area_name": "stop_name",
        "area_lat": "stop_lat",
        "area_long": "stop_long",
    })[["stop_id", "stop_name", "stop_lat", "stop_long"]]

    # Water stops already in correct schema
    water_stops = stops_water[["stop_id", "stop_name", "stop_lat", "stop_long"]]

    stops_unified = (
        pd.concat([water_stops, key_areas_as_stops], ignore_index=True)
        .drop_duplicates(subset=["stop_id"], keep="first")
        .reset_index(drop=True)
    )

    return stops_unified, stop_id_map

from typing import Dict
import pandas as pd

def apply_stop_mapping_and_add_stop_info(
    validation_data: pd.DataFrame,
    stops_unified: pd.DataFrame,
    stop_id_map: Dict,
    *,
    stop_col: str = "stop_id",
    name_col: str = "stop_name",
    out_lat_col: str = "stop_lat",
    out_lon_col: str = "stop_long",
) -> pd.DataFrame:
    """
    Land stops:
      - add area_id (mapped)
      - replace stop_id with area_id
      - set stop_name/lat/long to the area's metadata
    Water stops:
      - keep stop_id
      - keep water metadata via stops_unified
    """

    out = validation_data.copy()

    # Keep original stop id for QA
    out["stop_id_original"] = out[stop_col]

    # area_id only for land stops (NaN for water)
    out["area_id"] = out[stop_col].map(stop_id_map)
    out["area_id"] = pd.to_numeric(out["area_id"], errors="coerce").astype("Int64")

    # Unified stop id (land -> area_id, water unchanged)
    out["stop_id_unified"] = out["area_id"].fillna(out[stop_col])
    out["stop_id_unified"] = pd.to_numeric(out["stop_id_unified"], errors="coerce").astype("Int64")

    # Join metadata using unified stop id
    stop_meta = stops_unified[["stop_id", "stop_name", "stop_lat", "stop_long"]].copy()
    stop_meta = stop_meta.rename(columns={"stop_lat": out_lat_col, "stop_long": out_lon_col})

    out = out.merge(
        stop_meta,
        left_on="stop_id_unified",
        right_on="stop_id",
        how="left",
        validate="m:1",
        suffixes=("", "_meta"),
    )

    # Final columns: stop_id becomes unified; name/coords become unified metadata
    out[stop_col] = out["stop_id_unified"]
    out[name_col] = out["stop_name"]
    out[out_lat_col] = out[out_lat_col]
    out[out_lon_col] = out[out_lon_col]

    # Cleanup helper columns from merge
    out = out.drop(columns=["stop_id_unified", "stop_id_meta"], errors="ignore")

    return out
