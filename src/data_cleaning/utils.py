import re
import time
from datetime import timedelta
from typing import Any, Dict, Optional
import pandas as pd


def normalise_title(text: Any) -> Any:
    """
    Normalise title_description for case-insensitive exact matching.
    - strips leading/trailing spaces
    - collapses multiple spaces
    - normalises apostrophes
    - converts to upper case
    """
    if pd.isna(text):
        return text
    s = str(text).strip()
    s = s.replace("’", "'")  
    s = re.sub(r"\s+", " ", s)
    return s.upper()


def now_string() -> str:
    """Return a human-readable local timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed_timedelta(t_start: float, t_end: Optional[float] = None) -> timedelta:
    """Return elapsed time as a timedelta."""
    if t_end is None:
        t_end = time.time()
    return timedelta(seconds=round(t_end - t_start))


def print_processing_report(dedup_stats: Dict[str, Any], nan_ticket_stats: Dict[str, Any]) -> None:
    """Print pipeline statistics."""
    print("\n=== 5-minute de-duplication ===")
    print(
        f"Records before cleaning : {dedup_stats['total_before']:,}\n"
        f"Records after cleaning  : {dedup_stats['total_after']:,}\n"
        f"Removed duplicates      : {dedup_stats['removed_duplicates']:,} "
        f"({dedup_stats['removed_percentage']:.2f}%)"
    )

    print("\n=== Ticket classification filtering ===")
    print(
        f"Records before filtering : {nan_ticket_stats['total_before']:,}\n"
        f"Records after filtering  : {nan_ticket_stats['total_after']:,}\n"
        f"Removed records          : {nan_ticket_stats['removed_rows']:,} "
        f"({nan_ticket_stats['removed_percentage']:.2f}%)\n"
    )
