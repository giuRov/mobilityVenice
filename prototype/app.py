#!/usr/bin/env python3
"""
Dash prototype for visualising ACTV validations.

Usage (from repo root):
  python prototype/app.py --data data/processed/validations_processed.parquet
  python prototype/app.py --data data/processed/carnival_processed.csv

Notes:
- The input dataset is expected to contain at least:
  - validation_datetime (datetime or parseable string)
  - ticket_class (e.g., "1", "2", "3", "4", ...)
  - stop, stop_name, stop_latitude, stop_longitude
"""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from .data_loader import load_dataset
from .dash_app import create_app


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dash prototype app.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/validations_processed.parquet",
        help="Path to processed validations dataset (.csv or .parquet).",
    )

    # Stop reference inputs (optional, with defaults)
    parser.add_argument(
        "--stops-water",
        type=str,
        default="data/raw/stopsWater.csv",
        help="Path to water stops CSV (stop, stop_name, latitude, longitude).",
    )
    parser.add_argument(
        "--stops-land-mapped",
        type=str,
        default="data/raw/stopsLandMapped.csv",
        help="Path to land stop mapping CSV (stop, stop_mapping).",
    )
    parser.add_argument(
        "--land-key-areas",
        type=str,
        default="data/raw/landKeyAreas.csv",
        help="Path to land key areas CSV (stop, stop_name, latitude, longitude).",
    )

    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    return parser.parse_args()

def main() -> None:
    try:
        args = parse_args()
        data_path = Path(args.data).expanduser().resolve()

        stops_water_path = Path(args.stops_water).expanduser().resolve()
        stops_land_mapped_path = Path(args.stops_land_mapped).expanduser().resolve()
        land_key_areas_path = Path(args.land_key_areas).expanduser().resolve()

        df = load_dataset(
            data_path,
            stops_water_path=stops_water_path,
            stops_land_mapped_path=stops_land_mapped_path,
            land_key_areas_path=land_key_areas_path,
        )
        
        app = create_app(df)
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

    except Exception as e:
        print("\n[ERROR] App crashed before starting the server.\n", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()


# Example usage:
# python -m prototype.app --data data/processed/winterProcessed.csv

# Example with custom stop reference files:
# python -m prototype.app \
#   --data data/processed/winterProcessed.csv \
#   --stops-water data/raw/stopsWater.csv \
#   --stops-land-mapped data/raw/stopsLandMapped.csv \
#   --land-key-areas data/raw/landKeyAreas.csv
