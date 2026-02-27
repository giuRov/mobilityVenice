import argparse
import json
from pathlib import Path
import pandas as pd

from data_cleaning.pipeline import process_validation_data
from data_cleaning.utils import print_processing_report

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run ACTV validation data cleaning pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw input CSVs.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Directory to write outputs.")
    parser.add_argument("--validations", type=str, default=None, help="Raw validations CSV filename.")
    parser.add_argument("--drop-missing-stops", action="store_true", help="Drop stops listed in missing_stops.csv if present.")
    parser.add_argument("--dedup-minutes", type=int, default=5, help="Time window (minutes) for duplicate removal.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output filename prefix (without extension). If not provided, derived from validations filename."
)

    args = parser.parse_args()

    # Input/output directories
    raw = args.raw_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Path to raw validations file
    validations_path = raw / args.validations

    # Derive output prefix from input filename if not explicitly provided
    # Example: winter_raw.csv -> winter
    if args.output_prefix is not None: output_prefix = args.output_prefix
    else: output_prefix = validations_path.stem.removesuffix('_raw')

    # Load stop datasets 
    stops_water = pd.read_csv(raw / "stopsWater.csv")
    stops_land = pd.read_csv(raw / "stopsLand.csv")
    land_key_areas = pd.read_csv(raw / "landKeyAreas.csv")
    stops_land_mapped = pd.read_csv(raw / "stopsLandMapped.csv")

    # Load validation records
    validation = pd.read_csv(raw / args.validations)

    # Run the processing pipeline
    processed, stop_id_map, dedup_stats, nan_ticket_stats = process_validation_data(
        validation_data=validation,
        stops_water=stops_water,
        stops_land=stops_land,
        land_key_areas=land_key_areas,
        stops_land_mapped=stops_land_mapped,
        dedup_minutes=args.dedup_minutes,
        verbose=True,
    )

    # # Save processed validation records
    processed.to_csv(out / f"{output_prefix}.csv", index=False)
    pd.DataFrame([{"stop_id": k, "area_id": v} for k, v in stop_id_map.items()])

    # Save processing statistics (JSON format)
    stats = {"dedup_stats": dedup_stats, "nan_ticket_stats": nan_ticket_stats}
    (out / f"{output_prefix}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Print summary
    print_processing_report(dedup_stats, nan_ticket_stats)


if __name__ == "__main__":
    main()
