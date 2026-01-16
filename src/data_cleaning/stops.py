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
    - stops_water:       ['stop', 'stop_name', 'latitude', 'longitude']
    - stops_land:        ['stop', 'stop_name', 'latitude', 'longitude']  (not strictly required here)
    - land_key_areas:    ['stop', 'stop_name', 'latitude', 'longitude']  (stop = key area id)
    - stops_land_mapped: ['stop', 'stop_mapping']  (original land stop -> key area stop)

    Returns:
    - stops_unified: ['stop', 'stop_name', 'latitude', 'longitude']
    - stop_id_map: dict {old_land_stop -> key_area_stop}
    """

    stop_id_map = (
        stops_land_mapped.dropna(subset=["stop", "stop_mapping"])
        .set_index("stop")["stop_mapping"]
        .to_dict()
    )

    stops_unified = (
        pd.concat(
            [
                stops_water[["stop", "stop_name", "latitude", "longitude"]],
                land_key_areas[["stop", "stop_name", "latitude", "longitude"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["stop"], keep="first")
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
) -> pd.DataFrame:
    """
    Apply the land-to-key-area stop mapping and enrich the dataset with stop metadata.
    """

    out = validation_data.copy()

    # Replace land stop IDs (vectorised). Water stops remain unchanged.
    mapped = out[stop_col].map(stop_id_map)
    out[stop_col] = mapped.fillna(out[stop_col])

    # Ensure stop is an integer dtype (nullable Int64).
    out[stop_col] = pd.to_numeric(out[stop_col], errors="coerce").astype("Int64")

    # Join stop metadata (name + coordinates).
    stop_meta = stops_unified[[stop_col, name_col, lat_col, lon_col]].copy()
    stop_meta = stop_meta.rename(columns={lat_col: out_lat_col, lon_col: out_lon_col})

    out = out.merge(stop_meta, on=stop_col, how="left", validate="m:1")

    return out
