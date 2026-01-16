import pandas as pd

def deduplicate_within_minutes(
    validation_data: pd.DataFrame,
    *,
    datetime_col: str = "validation_datetime",
    serial_col: str = "serial",
    stop_col: str = "stop",
    minutes: int = 5,
    return_stats: bool = True,
):
    """
    Remove duplicate validations within a time window.
    For the same (serial, stop), keep only the first validation within `minutes`.

    Returns:
    - cleaned DataFrame
    - stats dict (if return_stats=True)
    """

    out = validation_data.copy()

    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out = out.sort_values([serial_col, stop_col, datetime_col], kind="mergesort")

    dt_diff = out.groupby([serial_col, stop_col], sort=False)[datetime_col].diff()

    threshold = pd.Timedelta(minutes=minutes)
    is_dup = dt_diff.notna() & (dt_diff <= threshold)

    removed = int(is_dup.sum())
    total_before = len(out)
    total_after = total_before - removed

    cleaned = out.loc[~is_dup].copy()

    if return_stats:
        stats = {
            "total_before": total_before,
            "total_after": total_after,
            "removed_duplicates": removed,
            "removed_percentage": round(removed / total_before * 100, 2) if total_before else 0.0,
            "window_minutes": minutes,
        }
        return cleaned, stats

    return cleaned
