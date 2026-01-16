import time
import pandas as pd


from .stops import build_unified_stops, apply_stop_mapping_and_add_stop_info
from .dedup import deduplicate_within_minutes
from .tickets import add_ticket_code, drop_nan_ticket_code
from .utils import now_string, elapsed_timedelta


def process_validation_data(
    validation_data: pd.DataFrame,
    *,
    stops_water: pd.DataFrame,
    stops_land: pd.DataFrame,
    land_key_areas: pd.DataFrame,
    stops_land_mapped: pd.DataFrame,
    datetime_col: str = "validation_datetime",
    serial_col: str = "serial",
    stop_col: str = "stop",
    title_col: str = "title_description",
    dedup_minutes: int = 5,
    verbose: bool = True,
):
    """
    Run the full pipeline:
    1) Stop harmonisation + add stop_name and coordinates
    2) 5-minute de-duplication by (serial, stop)
    3) Ticket categorisation (ticket_code)
    4) Removal of records with missing ticket_code

    Returns:
    - processed_data
    - stops_unified
    - stop_id_map
    - dedup_stats
    - nan_ticket_stats
    """

    t_start = time.time()

    if verbose:
        print("=== ACTV validation data processing ===")
        print(f"Start time: {now_string()}")

    # 1) Stops
    stops_unified, stop_id_map = build_unified_stops(
        stops_water=stops_water,
        stops_land=stops_land,
        land_key_areas=land_key_areas,
        stops_land_mapped=stops_land_mapped,
    )

    data1 = apply_stop_mapping_and_add_stop_info(
        validation_data=validation_data,
        stops_unified=stops_unified,
        stop_id_map=stop_id_map,
        stop_col=stop_col,
        name_col="stop_name",
        lat_col="latitude",
        lon_col="longitude",
        out_lat_col="stop_latitude",
        out_lon_col="stop_longitude",
    )

    # 2) De-duplication
    data2, dedup_stats = deduplicate_within_minutes(
        validation_data=data1,
        datetime_col=datetime_col,
        serial_col=serial_col,
        stop_col=stop_col,
        minutes=dedup_minutes,
        return_stats=True,
    )

    # 3) Ticket categorisation
    data3 = add_ticket_code(
        validation_data=data2,
        title_col=title_col,
        out_col="ticket_code",
    )

    # 4) Drop NaN ticket_code
    data4, nan_ticket_stats = drop_nan_ticket_code(
        data3,
        ticket_col="ticket_code",
        title_col=title_col,
        return_stats=True,
        with_counts=True,
    )

    # Reorder columns (final dataset)
    preferred_order = [
        datetime_col,
        serial_col,
        stop_col,
        "stop_name",
        "stop_latitude",
        "stop_longitude",
        "title",
        title_col,
        "ticket_code",
    ]
    preferred_order = [c for c in preferred_order if c in data4.columns]
    remaining_cols = [c for c in data4.columns if c not in preferred_order]
    data4 = data4[preferred_order + remaining_cols]

    if verbose:
        print(f"End time  : {now_string()}")
        print(f"Elapsed   : {elapsed_timedelta(t_start)}")
        print("======================================")

    return data4, stops_unified, stop_id_map, dedup_stats, nan_ticket_stats
