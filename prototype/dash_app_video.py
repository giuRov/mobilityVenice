#!/usr/bin/env python3
from __future__ import annotations

"""Second prototype: animated ("video-like") map with 3-hour bins.

Keeps the same left-hand controls (date + ticket class OR user category),
but replaces the static map + histogram with a single animated map.

In Plotly/Dash, the closest native equivalent to a video is an animated
figure with frames + play/pause controls.
"""

import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.io as pio

from .data_loader import build_disabled_days, filter_df

pio.templates.default = "plotly_white"


def create_app_video(df: pd.DataFrame) -> dash.Dash:
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

                    html.Label("Date", style={"fontSize": "25px", "marginTop": "20px"}),
                    dcc.DatePickerRange(
                        id="my-date-picker-range",
                        min_date_allowed=min_day_str,
                        max_date_allowed=max_day_str,
                        initial_visible_month=min_day_str,
                        start_date=min_day_str,
                        disabled_days=disabled_days_str,
                    ),

                    html.Div(
                        id="warning-message",
                        style={"color": "red", "marginTop": "10px", "fontSize": "16px"},
                    ),

                    html.Br(),

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
                        value=[],
                        placeholder="Select a ticket type",
                        style={"width": 400, "align-items": "left", "justify-content": "left"},
                        clearable=True,
                    ),

                    html.Br(),

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

                    html.Br(),
                    html.Div(
                        [
                            html.Div(
                                "Use the play button on the map to animate the 3-hour time bins.",
                                style={"fontSize": "14px", "color": "#666"},
                            ),
                        ]
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

            # Right panel: animated map
            html.Div(
                children=[
                    dcc.Loading(dcc.Graph(id="animated-map"), type="default"),
                ],
                style={"padding": 10, "flex": 1},
            ),
        ],
        style={"display": "flex", "flex-direction": "row"},
    )

    # 1) warning for missing dates inside the selected interval
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

    # 2) mutually exclusive selectors
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

        if has_tickets and has_category:
            return True, False, [], user_category

        if has_category:
            return True, False, [], user_category

        if has_tickets:
            return False, True, ticket_values, None

        return False, False, [], None

    def _apply_filters(
        base_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        ticket_values: list[str] | None,
        user_category: str | None,
    ) -> pd.DataFrame:
        dff = base_df

        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()

        dff = dff.copy()
        dff["date"] = pd.to_datetime(dff["date"]).dt.normalize()
        dff = dff[(dff["date"] >= start) & (dff["date"] <= end)]

        if dff.empty:
            return dff

        if user_category:
            if "user_category" not in dff.columns:
                return dff.iloc[0:0]
            dff = dff[dff["user_category"] == user_category]
        else:
            if not ticket_values:
                return dff.iloc[0:0]
            dff = filter_df(dff, start_date, end_date, ticket_values)

        return dff

    # 3) animated map (3-hour bins)
    @app.callback(
        Output("animated-map", "figure"),
        Input("my-date-picker-range", "start_date"),
        Input("my-date-picker-range", "end_date"),
        Input("my-dynamic-dropdown", "value"),
        Input("user-category-dropdown", "value"),
    )
    def update_animated_map(
        start_date: str | None,
        end_date: str | None,
        ticket_values: list[str] | None,
        user_category: str | None,
    ):
        if not start_date or not end_date:
            raise PreventUpdate

        dff = _apply_filters(df, start_date, end_date, ticket_values, user_category)
        if dff.empty or "validation_datetime" not in dff.columns:
            raise PreventUpdate

        dff = dff.copy()
        dt = pd.to_datetime(dff["validation_datetime"])

        # Floor timestamps to 3-hour bins (00:00, 03:00, 06:00, ...)
        dff["time_bin"] = dt.dt.floor("3H")
        dff["time_bin_label"] = dff["time_bin"].dt.strftime("%Y-%m-%d %H:%M")

        agg = (
            dff.groupby(
                ["time_bin_label", "loc_id", "stop_name", "stop_lat", "stop_long"],
                dropna=False,
            )
            .size()
            .reset_index(name="counts")
            .sort_values(["time_bin_label", "loc_id"])
        )

        if agg.empty:
            raise PreventUpdate

        cmin = float(agg["counts"].min())
        cmax = float(agg["counts"].max())

        fig = px.scatter_mapbox(
            agg,
            lat="stop_lat",
            lon="stop_long",
            color="counts",
            size="counts",
            animation_frame="time_bin_label",
            animation_group="loc_id",
            mapbox_style="carto-positron",
            width=1390,
            height=700,
            zoom=11.1,
            color_continuous_scale="viridis",
            range_color=[cmin, cmax],
            center={"lon": 12.337817, "lat": 45.44},
            hover_data={
                "time_bin_label": True,
                "loc_id": True,
                "stop_name": True,
                "stop_lat": False,
                "stop_long": False,
                "counts": True,
            },
            labels={"counts": "No. of validations", "time_bin_label": "Time (3h bin)"},
        )

        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 10}, font=dict(size=15))

        # Speed controls (milliseconds)
        if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
            fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

        return fig

    return app
