#!/usr/bin/env python3
"""
Dash prototype for visualising ACTV validations.

Usage (from repo root):
  python prototype/app.py --data data/processed/validations_processed.parquet
  python prototype/app.py --data data/processed/carnival_processed.csv

Notes:
- The input dataset is expected to contain at least:
  - validation_datetime (datetime or parseable string)
  - ticket_code (e.g., "1", "2", "3", "4", ...)
  - stop, stop_name, stop_latitude, stop_longitude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.io as pio


# -----------------------------------------------------------------------------
# Plotly configuration
# -----------------------------------------------------------------------------
pio.templates.default = "plotly_white"


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the processed validations dataset from CSV or Parquet.

    The function is deliberately strict about required columns to make failures
    explicit and easy to debug for external users.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        # Parse timestamps on load (faster + fewer surprises)
        df = pd.read_csv(
            path,
            parse_dates=["validation_datetime"],
            low_memory=False,
        )
    else:
        raise ValueError("Unsupported file extension. Use .csv or .parquet")

    required = {
        "validation_datetime",
        "ticket_code",
        "stop",
        "stop_name",
        "stop_latitude",
        "stop_longitude",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Input dataset is missing required column(s): "
            + ", ".join(sorted(missing))
            + f"\nFound columns: {df.columns.tolist()}"
        )

    # Ensure datetime is parsed
    df["validation_datetime"] = pd.to_datetime(df["validation_datetime"], errors="coerce")
    df = df.dropna(subset=["validation_datetime"]).copy()

    # Performance-friendly dtypes
    df["ticket_code"] = df["ticket_code"].astype("category")
    df["date"] = df["validation_datetime"].dt.normalize()  # Timestamp at midnight

    # Optional: ensure numeric stop ids where possible
    # (won't break if they are already int)
    df["stop"] = pd.to_numeric(df["stop"], errors="coerce").astype("Int64")

    return df


def build_disabled_days(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    """
    Build DatePickerRange bounds and disabled days list.

    Returns:
      - min_day_str: "YYYY-MM-DD"
      - max_day_str: "YYYY-MM-DD"
      - disabled_days_str: ["YYYY-MM-DD", ...]
    """
    min_day = df["date"].min()
    max_day = df["date"].max()

    all_days = pd.date_range(min_day, max_day, freq="D")
    available = set(df["date"].unique())
    disabled = sorted(set(all_days) - available)

    min_day_str = min_day.date().isoformat()
    max_day_str = max_day.date().isoformat()
    disabled_days_str = [d.date().isoformat() for d in disabled]

    return min_day_str, max_day_str, disabled_days_str


def filter_df(df: pd.DataFrame, start_date: str, end_date: str, ticket_values: list[str]) -> pd.DataFrame:
    """
    Filter dataframe by ticket_code and date interval.

    start_date/end_date come from Dash DatePickerRange and are ISO strings.
    """
    if not ticket_values:
        return df.iloc[0:0]

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    # Guard: avoid empty/invalid ranges
    if start >= end:
        return df.iloc[0:0]

    out = df[df["ticket_code"].isin(ticket_values)]
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out


# -----------------------------------------------------------------------------
# Dash app factory
# -----------------------------------------------------------------------------
def create_app(df: pd.DataFrame) -> dash.Dash:
    min_day_str, max_day_str, disabled_days_str = build_disabled_days(df)

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        [
            # -------------------------
            # Left panel: controls
            # -------------------------
            html.Div(
                children=[
                    html.H3(
                        "ACTV Validation Multiple Dates",
                        style={
                            "textAlign": "left",
                            "fontSize": "40px",
                            "fontFamily": "Georgia, serif",
                            "fontWeight": "bold",
                            "color": "#2c3e50",
                            "textShadow": "1px 1px 1px rgba(0,0,0,0.1)",
                            "marginBottom": "50px",
                        },
                    ),

                    # Date range selector
                    html.Label("Date", style={"fontSize": "25px", "marginTop": "20px"}),
                    dcc.DatePickerRange(
                        id="my-date-picker-range",
                        min_date_allowed=min_day_str,
                        max_date_allowed=max_day_str,
                        initial_visible_month=min_day_str,
                        start_date=min_day_str,
                        # NOTE: end_date intentionally not set
                        disabled_days=disabled_days_str,
                    ),

                    # Warning area
                    html.Div(
                        id="warning-message",
                        style={"color": "red", "marginTop": "10px", "fontSize": "16px"},
                    ),

                    html.Br(),

                    # Ticket selector
                    html.Label("Ticket", style={"fontSize": "25px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="my-dynamic-dropdown",
                        options=[
                            {"label": "24 hours", "value": "1"},
                            {"label": "48 hours", "value": "2"},
                            {"label": "72 hours", "value": "3"},
                            {"label": "7 days", "value": "4"},
                        ],
                        multi=True,
                        value=["1"],  # IMPORTANT: values are strings
                        placeholder="Select a ticket type",
                        style={"width": 400, "align-items": "left", "justify-content": "left"},
                    ),
                ],
                style={
                    "padding": "20px",
                    "flex": "0.3",
                    "backgroundColor": "#f9f9f9",
                    "borderRadius": "10px",
                    "boxShadow": "2px 2px 10px rgba(0, 0, 0, 0.1)",
                    "margin": "20px",
                },
            ),

            # -------------------------
            # Right panel: outputs
            # -------------------------
            html.Div(
                children=[
                    dcc.Loading(dcc.Graph(id="mymap"), type="default"),
                    dcc.Loading(dcc.Graph(id="bar-chart", clickData=None), type="default"),
                ],
                style={"padding": 10, "flex": 1},
            ),
        ],
        style={"display": "flex", "flex-direction": "row"},
    )

    # -------------------------------------------------------------------------
    # Callback 1: warning for missing dates inside the selected interval
    # -------------------------------------------------------------------------
    @app.callback(
        Output("warning-message", "children"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
    )
    def check_date_continuity(start_date: str | None, end_date: str | None) -> str:
        if not start_date or not end_date:
            raise PreventUpdate

        available_days_set = set(df["date"].unique())

        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()
        if start >= end:
            return ""

        full_range = pd.date_range(start=start, end=end, freq="D")
        missing = [d for d in full_range if d not in available_days_set]

        if missing:
            return (
                f"Warning: the selected period contains {len(missing)} missing date(s), "
                f"e.g. {missing[0].date()}."
            )
        return ""

    # -------------------------------------------------------------------------
    # Callback 2: update the map (validations aggregated by stop)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("mymap", "figure"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
        Input("my-dynamic-dropdown", "value"),
    )
    def update_map(start_date: str | None, end_date: str | None, ticket_values: list[str] | None):
        if not start_date or not end_date or not ticket_values:
            raise PreventUpdate

        dff = filter_df(df, start_date, end_date, ticket_values)
        if dff.empty:
            raise PreventUpdate

        agg = (
            dff.groupby(["stop", "stop_name", "stop_latitude", "stop_longitude"], dropna=False)
            .size()
            .reset_index(name="counts")
            .sort_values(by=["stop"])
        )

        fig = px.scatter_mapbox(
            agg,
            lat="stop_latitude",
            lon="stop_longitude",
            color="counts",
            size="counts",
            mapbox_style="carto-positron",
            width=1390,
            height=500,
            zoom=11.1,
            color_continuous_scale="viridis",
            range_color=[agg["counts"].min(), agg["counts"].max()],
            center={"lon": 12.337817, "lat": 45.44},
            hover_data={
                "stop": True,
                "stop_name": True,
                "stop_latitude": False,
                "stop_longitude": False,
                "counts": True,
            },
            labels={"counts": "Number of validations"},
        )

        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 10}, font=dict(size=15))
        return fig

    # -------------------------------------------------------------------------
    # Callback 3: update the bar chart (validations aggregated by hour slot)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("bar-chart", "figure"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
        Input("my-dynamic-dropdown", "value"),
    )
    def update_bar_chart(start_date: str | None, end_date: str | None, ticket_values: list[str] | None):
        if not start_date or not end_date or not ticket_values:
            raise PreventUpdate

        dff = filter_df(df, start_date, end_date, ticket_values)
        if dff.empty:
            raise PreventUpdate

        # Hour slot from validation_datetime (fast and robust)
        dff = dff.copy()
        dff["time_slot"] = dff["validation_datetime"].dt.strftime("%H:00")

        grouped = (
            dff.groupby("time_slot")
            .size()
            .reset_index(name="counts")
            .sort_values(by="time_slot")
        )

        fig = px.bar(
            grouped,
            x="time_slot",
            y="counts",
            color="counts",
            color_continuous_scale="viridis",
            text_auto=".2s",
            height=400,
            width=1400,
            labels={"time_slot": "Time slot (hour)", "counts": "Number of validations"},
        )

        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(
            margin={"t": 15, "l": 0, "b": 0, "r": 12, "pad": 7},
            font=dict(size=16),
        )
        fig.update_xaxes(title_text="Time slot (hour)")
        fig.update_yaxes(title_text="Number of validations")
        return fig

    return app


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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()

    df = load_dataset(data_path)
    print(f"Loaded {len(df):,} rows from: {data_path}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    app = create_app(df)
    app.run_server(host=args.host, port=args.port, debug=args.debug, dev_tools_ui=args.debug)


if __name__ == "__main__":
    main()

