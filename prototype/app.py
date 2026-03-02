#!/usr/bin/env python3
"""
Dash prototype for visualising ACTV validation data.

Usage (from the repository root):
  # Prototype 1 (existing): map + hourly histogram
  python -m prototype.app --prototype 1 --data data/processed/winter.csv

  # Prototype 2 (new): animated map with 3-hour bins
  python -m prototype.app --prototype 2 --data data/processed/winter.csv

Inputs
------
Processed validations dataset (.csv or .parquet) containing at least:
  - validation_datetime : validation timestamp
  - ticket_class : ticket class identifier (e.g. "D-1", "D-2", "D-3", "D-7")
  - user_category : derived user category
  - loc_id : unified stop identifier

Additional reference datasets (optional, default paths provided):
  - stopsWater.csv
  - stopsLandMapped.csv
  - landKeyAreas.csv

These reference tables are used to reconstruct stop names and coordinates
for the spatial visualisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from .data_loader import load_dataset
from .dash_app import create_app
from .dash_app_video import create_app_video


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

    parser.add_argument(
        "--prototype",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which prototype to run: 1 = map + hourly histogram, 2 = animated 3-hour map.",
    )
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
        
        if args.prototype == 1:
            app = create_app(df)
        else:
            app = create_app_video(df)
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

    except Exception as e:
        print("\n[ERROR] App crashed before starting the server.\n", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

