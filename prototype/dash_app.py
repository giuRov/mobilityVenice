#!/usr/bin/env python3
from __future__ import annotations

import dash
from dash import dcc, html, Input, Output, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.io as pio

from .data_loader import build_disabled_days, filter_df

# -----------------------------------------------------------------------------
# Plotly configuration
# -----------------------------------------------------------------------------
pio.templates.default = "plotly_white"


# -----------------------------------------------------------------------------
# Dash app factory
# -----------------------------------------------------------------------------
def create_app(df: pd.DataFrame) -> dash.Dash:
    min_day_str, max_day_str, disabled_days_str = build_disabled_days(df)

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        [
            # Left panel: controls
            html.Div(
                children=[
                    html.H3(
                        "Venice Mobility Data Explorer",
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
                    html.Label("Ticket Class", style={"fontSize": "25px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="my-dynamic-dropdown",
                        options=[
                            {"label": "1 Day", "value": "D-1"},
                            {"label": "2 Days", "value": "D-2"},
                            {"label": "3 Days", "value": "D-3"},
                            {"label": "7 Days", "value": "D-7"},
                            {"label": "Monthly Students", "value": "M-STUD"},
                            {"label": "Yearly Students", "value": "Y-STUD"},
                            {"label": "Monthly Residents", "value": "M-RES"},
                            {"label": "Yearly Residents", "value": "Y-RES"},
                            {"label": "Yearly Retirees", "value": "RET"},
                            {"label": "Occasional Travellers", "value": "75"},
                        ],
                        multi=True,
                        value=[],  # start empty; mutually exclusive with user category selector
                        placeholder="Select a ticket type",
                        style={"width": 400, "align-items": "left", "justify-content": "left"},
                        clearable=True,
                    ),

                    html.Br(),

                    # User category selector (macro categories)
                    html.Label("User Category", style={"fontSize": "25px", "marginTop": "10px"}),
                    dcc.Dropdown(
                        id="user-category-dropdown",
                        options=[
                            {"label": "Tourists", "value": "Tourists"},
                            {"label": "Residents", "value": "Residents"},
                            {"label": "Students", "value": "Students"},
                            {"label": "Retirees", "value": "Retirees"},
                            {"label": "Occasional Travellers", "value": "Occasional Travellers"},
                        ],
                        multi=False,
                        value=None,
                        placeholder="Select a user category",
                        style={"width": 400, "align-items": "left", "justify-content": "left"},
                        clearable=True,
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

            # Right panel: outputs
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

        available_days_set = set(pd.to_datetime(df["date"]).dt.normalize().unique())

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
    # Callback 2: mutually exclusive selectors (ticket class vs user category)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("my-dynamic-dropdown", "disabled"),
        Output("user-category-dropdown", "disabled"),
        Output("my-dynamic-dropdown", "value"),
        Output("user-category-dropdown", "value"),
        Input("my-dynamic-dropdown", "value"),
        Input("user-category-dropdown", "value"),
    )
    def toggle_selectors(ticket_values: list[str] | None, user_category: str | None):
        has_tickets = bool(ticket_values) and len(ticket_values) > 0
        has_category = bool(user_category)

        # If both selected (e.g., state restore), keep category and clear tickets
        if has_tickets and has_category:
            return True, False, [], user_category

        if has_category:
            # Category selected -> disable ticket selector
            return True, False, [], user_category

        if has_tickets:
            # Tickets selected -> disable category selector
            return False, True, ticket_values, None

        # Nothing selected -> both enabled
        return False, False, [], None

    # -------------------------------------------------------------------------
    # Helper: filter df by date + either ticket values OR user category
    # -------------------------------------------------------------------------
    def _apply_filters(
        base_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        ticket_values: list[str] | None,
        user_category: str | None,
    ) -> pd.DataFrame:
        dff = base_df

        # Filter by date (normalise day)
        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()

        # Ensure date column is comparable
        dff = dff.copy()
        dff["date"] = pd.to_datetime(dff["date"]).dt.normalize()
        dff = dff[(dff["date"] >= start) & (dff["date"] <= end)]

        if dff.empty:
            return dff

        # Mutually exclusive filters
        if user_category:
            if "user_category" not in dff.columns:
                return dff.iloc[0:0]
            dff = dff[dff["user_category"] == user_category]
        else:
            if not ticket_values:
                return dff.iloc[0:0]
            dff = filter_df(dff, start_date, end_date, ticket_values)

        return dff

    # -------------------------------------------------------------------------
    # Callback 3: update the map (validations aggregated by stop)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("mymap", "figure"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
        Input("my-dynamic-dropdown", "value"),
        Input("user-category-dropdown", "value"),
    )
    def update_map(
        start_date: str | None,
        end_date: str | None,
        ticket_values: list[str] | None,
        user_category: str | None,
    ):
        if not start_date or not end_date:
            raise PreventUpdate

        dff = _apply_filters(df, start_date, end_date, ticket_values, user_category)
        if dff.empty:
            raise PreventUpdate

        agg = (
            dff.groupby(["loc_id", "stop_name", "stop_lat", "stop_long"], dropna=False)
            .size()
            .reset_index(name="counts")
            .sort_values(by=["loc_id"])
        )

        fig = px.scatter_mapbox(
            agg,
            lat="stop_lat",
            lon="stop_long",
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
                "loc_id": True,
                "stop_name": True,
                "stop_lat": False,
                "stop_long": False,
                "counts": True,
            },
            labels={"counts": "No. of validations"},
        )

        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 10}, font=dict(size=15))
        return fig

    # -------------------------------------------------------------------------
    # Callback 4: update the bar chart (validations aggregated by hour slot)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("bar-chart", "figure"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
        Input("my-dynamic-dropdown", "value"),
        Input("user-category-dropdown", "value"),
    )
    def update_bar_chart(
        start_date: str | None,
        end_date: str | None,
        ticket_values: list[str] | None,
        user_category: str | None,
    ):
        if not start_date or not end_date:
            raise PreventUpdate

        dff = _apply_filters(df, start_date, end_date, ticket_values, user_category)
        if dff.empty:
            raise PreventUpdate

        dff = dff.copy()
        if "validation_datetime" not in dff.columns:
            raise PreventUpdate

        dff["time_slot"] = pd.to_datetime(dff["validation_datetime"]).dt.strftime("%H:00")

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
            labels={"time_slot": "Time slot (hour)", "counts": "No. of validations"},
        )

        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(
            margin={"t": 15, "l": 0, "b": 0, "r": 12, "pad": 7},
            font=dict(size=16),
        )
        fig.update_xaxes(title_text="Time slot (hour)")
        fig.update_yaxes(title_text="No. of validations")
        return fig

    return app
