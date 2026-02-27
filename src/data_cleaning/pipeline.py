import time
import pandas as pd

from .stops import build_unified_stops, apply_stop_mapping_and_add_stop_info
from .dedup import deduplicate_within_minutes
from .tickets import add_ticket_class, drop_nan_ticket_class
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
    Run the processing pipeline on ACTV validation data.

    Steps
    -----
    1) Temporal de-duplication of repeated validations within a fixed time window.
    2) Spatial aggregation of land stops into land key areas and enrichment of
       validation records with stop metadata.
    3) Ticket-based categorisation (ticket_class and user_category).
    4) Filtering: drop records with missing or invalid ticket_class.
    5) Reorder columns in the final dataset and rename `stop` to `loc_id`.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Raw validation records.

    stops_water, stops_land, land_key_areas, stops_land_mapped : pd.DataFrame
        Reference tables used to construct the stop mapping and stop metadata.

    datetime_col, serial_col, stop_col, title_col : str
        Column names in `validation_data`.

    dedup_minutes : int
        Length of the temporal window (in minutes) used to remove repeated
        validations associated with the same (serial, stop).

    verbose : bool
        If True, prints timing and progress information.

    Returns
    -------
    processed_data : pd.DataFrame
        Processed validation records with reordered columns. The stop identifier
        column is renamed from `stop` to `loc_id`.

    stop_id_map : dict
        Mapping from raw stop identifiers to standardised stop identifiers,
        including aggregation of land stops into key areas.

    dedup_stats : dict
        Summary statistics produced by the de-duplication step.

    nan_ticket_stats : dict
        Summary statistics for records removed due to missing or invalid
        ticket_class values (including counts per title when available).
    """

    # Record starting time for performance reporting
    t_start = time.time()

    # Print header and starting time if verbose mode is enabled
    if verbose:
        print("=== ACTV validation data processing ===")
        print(f"Start time: {now_string()}")

    # ------------------------------------------------------------------
    # 1) Temporal de-duplication
    #
    # Remove repeated validations occurring within a fixed time window
    # for the same (serial, stop) pair. Statistics on removed records
    # are also returned.
    # ------------------------------------------------------------------
    data1, dedup_stats = deduplicate_within_minutes(
        validation_data=validation_data,
        datetime_col=datetime_col,
        serial_col=serial_col,
        stop_col=stop_col,
        minutes=dedup_minutes,
        return_stats=True,
    )

    # ------------------------------------------------------------------
    # 2) Stop processing
    #
    # Build the stop mapping that standardises raw stop identifiers and
    # aggregates land stops into predefined land key areas. The mapping
    # is then applied to the validation records, and stop metadata
    # (name and coordinates) are added.
    # ------------------------------------------------------------------
    stops_unified, stop_id_map = build_unified_stops(
        stops_water=stops_water,
        stops_land=stops_land,
        land_key_areas=land_key_areas,
        stops_land_mapped=stops_land_mapped,
    )
    data2 = apply_stop_mapping_and_add_stop_info(
        validation_data=data1,
        stops_unified=stops_unified,
        stop_id_map=stop_id_map,
        stop_col=stop_col,
        name_col="stop_name",
        out_lat_col="stop_lat",
        out_lon_col="stop_long",
    )

    # ------------------------------------------------------------------
    # 3) Ticket-based categorisation
    #
    # Assign a ticket_class and a user_category to each validation
    # record based on the title_description field.
    # ------------------------------------------------------------------
    data3 = add_ticket_class(
        validation_data=data2,
        title_col=title_col,
        out_col="ticket_class",
        user_category_col="user_category",
    )

    # ------------------------------------------------------------------
    # 4) Ticket filtering
    #
    # Remove records with missing or invalid ticket_class values.
    # Statistics describing the removed records are returned.
    # ------------------------------------------------------------------
    data4, nan_ticket_stats = drop_nan_ticket_class(
        data3,
        ticket_col="ticket_class",
        title_col=title_col,
        return_stats=True,
        with_counts=True,
    )

    # ------------------------------------------------------------------
    # 5) Final column selection and ordering
    #
    # Keep only the relevant columns.
    # ------------------------------------------------------------------
    preferred_order = [
        datetime_col,
        serial_col,
        stop_col,
        "ticket_class",
        "user_category",
    ]

    preferred_order = [c for c in preferred_order if c in data4.columns]
    data4 = data4[preferred_order]

    # Rename stop identifier column to loc_id in the final dataset
    data4 = data4.rename(columns={'stop': 'loc_id'})

    # Print ending time and total elapsed time if verbose mode is enabled
    if verbose:
        print(f"End time  : {now_string()}")
        print(f"Elapsed   : {elapsed_timedelta(t_start)}")
        print("======================================")

    # Return processed dataset and processing statistics
    return data4, stop_id_map, dedup_stats, nan_ticket_stats
