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
    stop_col: str = "stop_id",
    name_col: str = "stop_name",
    lat_col: str = "stop_lat",
    lon_col: str = "stop_long",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Construct the unified stop reference table used across the analysis.

    The goal is to create a single, harmonised spatial reference system where:
    - Water stops are preserved as originally defined.
    - Land stops are replaced by their corresponding key areas.
      Key areas therefore act as aggregated land-stop representations.

    Parameters
    ----------
    stops_water : pd.DataFrame
        Table of water stops already expressed in stop schema:
        ['stop_id', 'stop_name', 'stop_lat', 'stop_long'].

    land_key_areas : pd.DataFrame
        Table of key areas representing aggregated land locations:
        ['area_id', 'area_name', 'area_lat', 'area_long'].

    stops_land_mapped : pd.DataFrame
        Mapping table linking original land stops to their key area:
        ['stop_id', 'area_id'].

    Returns
    -------
    stops_unified : pd.DataFrame
        Unified stop table with schema:
        ['stop_id', 'stop_name', 'stop_lat', 'stop_long'].

        This contains:
        - original water stops
        - key areas promoted to stop-level entities (area_id becomes stop_id)

    stop_id_map : Dict
        Dictionary mapping each original land stop to its key area:
        {original_stop_id -> area_id}.

    Notes
    -----
    This function defines the spatial harmonisation layer used downstream.
    After this step, analyses no longer distinguish between:
    - individual land stops
    - aggregated land areas

    Both are treated uniformly as "stops" within a consistent spatial model.
    """

    # ------------------------------------------------------------------
    # Build mapping: original land stop -> key area.
    # This will later be used to replace land stop IDs in validation data.
    # ------------------------------------------------------------------
    mapping_df = stops_land_mapped.dropna(subset=["stop_id", "area_id"]).copy()
    mapping_df["stop_id"] = _coerce_key(mapping_df["stop_id"])
    mapping_df["area_id"] = _coerce_key(mapping_df["area_id"])
    stop_id_map = mapping_df.set_index("stop_id")["area_id"].to_dict()

    # ------------------------------------------------------------------
    # Water stops already follow the target schema.
    # We only enforce dtype consistency for safe joins later on.
    # ------------------------------------------------------------------
    water = stops_water[[stop_col, name_col, lat_col, lon_col]].copy()
    water[stop_col] = _coerce_key(water[stop_col])

    # ------------------------------------------------------------------
    # Promote key areas to stop-level entities by reshaping their schema.
    # Here, the area identifier becomes the new stop identifier.
    # This is a semantic transformation, not just a rename:
    # key areas are now treated as the canonical representation of land stops.
    # ------------------------------------------------------------------
    keyareas = land_key_areas.rename(columns={
        "area_id": stop_col,
        "area_name": name_col,
        "area_lat": lat_col,
        "area_long": lon_col,
    })[[stop_col, name_col, lat_col, lon_col]].copy()
    keyareas[stop_col] = _coerce_key(keyareas[stop_col])

    # ------------------------------------------------------------------
    # Combine water stops and promoted key areas into a single reference set.
    # Duplicate IDs are removed defensively to guarantee uniqueness.
    # ------------------------------------------------------------------
    stops_unified = (
        pd.concat([water, keyareas], ignore_index=True)
        .drop_duplicates(subset=[stop_col], keep="first")
        .reset_index(drop=True)
    )

    return stops_unified, stop_id_map


from typing import Dict
import pandas as pd

def apply_area_id_and_add_stop_info(
    validation_data: pd.DataFrame,
    stops_unified: pd.DataFrame,
    stop_id_map: Dict,
    *,
    loc_col: str = "loc_id",
    name_col: str = "stop_name",
    lat_col: str = "stop_lat",
    lon_col: str = "stop_long",
    return_report: bool = False,
):
    """
    Apply the land-stop to key-area mapping and enrich validation records with
    unified stop metadata.

    Conceptually:
    - Water stops remain unchanged.
    - Land stops are replaced by their corresponding key area.
    - The final location identifier (loc_id) represents a unified stop system
      where key areas act as aggregated land-stop representations.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Validation dataset containing a location identifier (loc_id). This may
        refer either to a water stop or to an original land stop.

    stops_unified : pd.DataFrame
        Master table of unified stops containing both:
        - original water stops
        - key areas promoted to stop-level entities

        Expected columns:
        ['stop_id', 'stop_name', 'stop_lat', 'stop_long'].

    stop_id_map : Dict
        Mapping {original_land_stop_id -> area_id}. Only land stops appear in
        this dictionary.

    loc_col : str, default="loc_id"
        Column in `validation_data` representing the location identifier to be
        mapped to the unified stop system.

    return_report : bool, default=False
        If True, also return a dictionary summarising mapping diagnostics.

    Returns
    -------
    pd.DataFrame
        Validation dataset enriched with:
        - `area_id` (only for land stops; NaN for water stops)
        - unified `loc_id`
        - stop_name, stop_lat, stop_long corresponding to the unified stop

    Notes
    -----
    After this operation:
    - All land validations are spatially referenced to their key area.
    - Water validations retain their original stop identity.
    - The dataset can be analysed consistently without mixing raw land stops
      and aggregated land areas.
    """

    out = validation_data.copy()

    # ------------------------------------------------------------------
    # Ensure the join key has a consistent dtype to avoid merge mismatches.
    # This is especially important when IDs originate from heterogeneous
    # sources (CSV files, database extracts, Excel exports, etc.).
    # ------------------------------------------------------------------
    out[loc_col] = _coerce_key(out[loc_col])

    # ------------------------------------------------------------------
    # Derive the key-area identifier. For water stops this remains missing,
    # since only land stops are present in the mapping dictionary.
    # ------------------------------------------------------------------
    out["area_id"] = out[loc_col].map(stop_id_map)
    out["area_id"] = pd.to_numeric(out["area_id"], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # Build the unified location identifier in a dtype-safe way.
    # Using Series.fillna() here can raise dtype errors when combining
    # pandas StringArray with nullable IntegerArray. We therefore:
    # 1) work in a temporary string representation, and
    # 2) normalise back using _coerce_key().
    # ------------------------------------------------------------------
    unified = out[loc_col].astype("string")
    unified = unified.where(out["area_id"].isna(), out["area_id"].astype("string"))
    out[loc_col] = _coerce_key(unified)

    # ------------------------------------------------------------------
    # Join unified stop metadata (name and coordinates). Metadata now reflects:
    # - water stop geometry for water validations
    # - key-area geometry for land validations
    # ------------------------------------------------------------------
    stop_meta = stops_unified[["stop_id", name_col, lat_col, lon_col]].copy()
    stop_meta["stop_id"] = _coerce_key(stop_meta["stop_id"])
    stop_meta = stop_meta.drop_duplicates(subset=["stop_id"], keep="first")

    out = out.merge(
        stop_meta,
        left_on=loc_col,
        right_on="stop_id",
        how="left",
        validate="m:1",
    )

    # The right-side stop_id is redundant after the merge
    out = out.drop(columns=["stop_id"], errors="ignore")

    if not return_report:
        return out

    # ------------------------------------------------------------------
    # Diagnostic report 
    # ------------------------------------------------------------------
    report = {
        "n_rows": int(len(out)),
        "n_unique_locations_after_mapping": int(out[loc_col].nunique(dropna=True)),
        "n_rows_mapped_to_area": int(out["area_id"].notna().sum()),
        "n_missing_stop_name": int(out[name_col].isna().sum()),
        "n_missing_stop_lat": int(out[lat_col].isna().sum()),
        "n_missing_stop_long": int(out[lon_col].isna().sum()),
    }

    return out, report
