#!/usr/bin/env python3
"""Render a cinematic Plotly video showing DC risk spikes versus U.S. states.

Usage from a notebook:

    from src.plotly_dc_cinematic_map import build_cinematic_video
    build_cinematic_video(merged_df_table)

Usage from the CLI:

    ./.venv/bin/python src/plotly_dc_cinematic_map.py

Offline CSV fallback:

    ./.venv/bin/python src/plotly_dc_cinematic_map.py --source csv --input-csv merged_df_table.csv

Required columns:
    - State
    - Year
    - risk_per_100k
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ALL_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
]

STATE_NAME_TO_ABBREV = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}

DC_LAT = 38.9072
DC_LON = -77.0369

WIDTH = 1600
HEIGHT = 900
BACKGROUND = "#050816"
PANEL = "#091224"
TEXT_PRIMARY = "#F5F7FB"
TEXT_SECONDARY = "#B9C3D6"
TEXT_MUTED = "#6F7C93"
STATE_LINE = "#172338"
NEUTRAL_GREY = "#7A8597"
MUTED_GREY = "#C9D1DC"
SPIKE_RED = "#FF4D6D"
SPIKE_RED_SOFT = "#FF6B83"
NEON_RED = "#FF3B5C"
NEON_RED_GLOW = "#FF889E"
BLUE = "#4E8EF7"

FONT_FAMILY = "Avenir Next, Helvetica Neue, Arial, sans-serif"

INTRO_TITLE = "School Shooting Risk per 100k Students"
INTRO_SUBTITLE = "2018–2025 — State Comparison"
FOCUS_TEXT = "DC shows disproportionate spikes relative to national baseline"
OUTRO_TEXT = "Key Insight: Risk concentration is not uniform across states"
OUTRO_OPTIONAL = "DC consistently exhibits elevated spikes post-2020"


def rgba(hex_color: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def ensure_kaleido() -> None:
    if importlib.util.find_spec("kaleido") is None:
        raise RuntimeError(
            "kaleido is required for Plotly PNG export. "
            "Install it in the active environment, then rerun the script."
        )


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to assemble the MP4 video.")


def load_panel_from_csv(path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    return prepare_panel(panel)


def fetch_all_rows_supabase(supabase_client, table_name: str, page_size: int = 1000) -> pd.DataFrame:
    start = 0
    rows: list[dict] = []
    while True:
        response = (
            supabase_client.table(table_name)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        chunk = response.data or []
        if not chunk:
            break
        rows.extend(chunk)
        start += page_size
    return pd.DataFrame(rows)


def build_panel_from_supabase() -> pd.DataFrame:
    try:
        from dotenv import load_dotenv
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Supabase loading requires `python-dotenv` and `supabase` in the active environment."
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_KEY. "
            "Either configure `.env` or use `--source csv --input-csv <path>`."
        )

    supabase_client = create_client(supabase_url, supabase_key)
    incident_df = fetch_all_rows_supabase(supabase_client, "incident")
    enrollment_df = fetch_all_rows_supabase(supabase_client, "enrollment_state_year_mat")

    incident_df["State"] = incident_df["State"].astype(str).str.upper().str.strip()
    incident_df["Year"] = pd.to_numeric(incident_df["Year"], errors="coerce")
    incident_df = incident_df[
        incident_df["State"].isin(ALL_STATES) & incident_df["Year"].notna()
    ].copy()
    incident_df["Year"] = incident_df["Year"].astype(int)

    incident_counts = (
        incident_df.groupby(["State", "Year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    enrollment_df["state"] = enrollment_df["state"].astype(str).str.upper().str.strip()
    enrollment_df["State"] = enrollment_df["state"].map(STATE_NAME_TO_ABBREV)
    enrollment_df["Year"] = pd.to_numeric(enrollment_df["year"], errors="coerce")
    enrollment_df["total_students"] = pd.to_numeric(
        enrollment_df["total_students"], errors="coerce"
    )
    enrollment_df = (
        enrollment_df.dropna(subset=["State", "Year", "total_students"])
        .sort_values("total_students", ascending=False)
        .drop_duplicates(["State", "Year"])
        .loc[:, ["State", "Year", "total_students"]]
        .copy()
    )
    enrollment_df["Year"] = enrollment_df["Year"].astype(int)

    merged_df_table = enrollment_df.merge(
        incident_counts,
        on=["State", "Year"],
        how="left",
    )
    merged_df_table["incident_count"] = merged_df_table["incident_count"].fillna(0.0)
    merged_df_table["risk_per_100k"] = np.where(
        merged_df_table["total_students"] > 0,
        merged_df_table["incident_count"] / merged_df_table["total_students"] * 100000.0,
        np.nan,
    )
    return prepare_panel(merged_df_table)


def load_input_panel(
    source: str,
    input_csv: str | None,
    write_csv_path: str | None,
) -> pd.DataFrame:
    if source == "csv":
        if not input_csv:
            raise RuntimeError("`--input-csv` is required when `--source csv` is used.")
        csv_path = Path(input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. "
                "Either point to an existing file or use `--source supabase`."
            )
        return load_panel_from_csv(csv_path)

    panel = build_panel_from_supabase()
    if write_csv_path:
        output_path = Path(write_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
    return panel


def prepare_panel(
    merged_df_table: pd.DataFrame,
    start_year: int = 2018,
    end_year: int = 2025,
) -> pd.DataFrame:
    required = {"State", "Year", "risk_per_100k"}
    missing = required - set(merged_df_table.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input data is missing required columns: {missing_list}")

    panel = merged_df_table.loc[:, ["State", "Year", "risk_per_100k"]].copy()
    panel["State"] = panel["State"].astype(str).str.upper().str.strip()
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce")
    panel["risk_per_100k"] = pd.to_numeric(panel["risk_per_100k"], errors="coerce")
    panel = panel.dropna(subset=["State", "Year"])
    panel["Year"] = panel["Year"].astype(int)
    panel = panel[
        panel["State"].isin(ALL_STATES)
        & panel["Year"].between(start_year, end_year)
    ].copy()

    if panel.empty:
        raise ValueError("No valid rows remain after filtering to 2018–2025 and U.S. states.")

    grouped = (
        panel.groupby(["State", "Year"], as_index=False)["risk_per_100k"]
        .mean()
        .sort_values(["State", "Year"])
    )

    full_index = pd.MultiIndex.from_product(
        [ALL_STATES, range(start_year, end_year + 1)],
        names=["State", "Year"],
    )
    panel = (
        grouped.set_index(["State", "Year"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["State", "Year"])
        .reset_index(drop=True)
    )

    panel["risk_per_100k"] = panel.groupby("State")["risk_per_100k"].transform(
        lambda series: series.interpolate(limit_direction="both")
    )
    panel["risk_per_100k"] = panel["risk_per_100k"].fillna(0.0)

    stats = panel.groupby("Year", as_index=False)["risk_per_100k"].agg(
        mean_year="mean",
        std_year="std",
    )
    panel = panel.merge(stats, on="Year", how="left")
    panel["std_year"] = panel["std_year"].replace(0, np.nan)
    panel["risk_zscore"] = (
        (panel["risk_per_100k"] - panel["mean_year"]) / panel["std_year"]
    ).fillna(0.0)
    panel["spike_flag"] = (panel["risk_zscore"] >= 1.0).astype(float)
    return panel


def build_metric_pivot(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    pivot = panel.pivot(index="State", columns="Year", values=column)
    return pivot.reindex(index=ALL_STATES)


def interpolate_series(
    pivot: pd.DataFrame,
    year_position: float,
    start_year: int,
    end_year: int,
) -> pd.Series:
    year_position = float(np.clip(year_position, start_year, end_year))
    lower_year = int(math.floor(year_position))
    upper_year = int(math.ceil(year_position))
    lower_year = max(start_year, min(end_year, lower_year))
    upper_year = max(start_year, min(end_year, upper_year))
    if lower_year == upper_year:
        return pivot[lower_year].copy()

    alpha = year_position - lower_year
    return pivot[lower_year] * (1.0 - alpha) + pivot[upper_year] * alpha


def scale_dc_sizes(dc_risk: pd.Series) -> pd.Series:
    low = float(dc_risk.min())
    high = float(dc_risk.max())
    if math.isclose(low, high):
        return pd.Series(22.0, index=dc_risk.index)
    scaled = 14.0 + (dc_risk - low) * (34.0 - 14.0) / (high - low)
    return scaled


def timeline_years(
    start_year: int,
    end_year: int,
    transition_frames: int,
    hold_frames: int,
    opening_hold: int = 0,
    closing_hold: int = 0,
) -> list[float]:
    years: list[float] = [float(start_year)] * opening_hold
    for year in range(start_year, end_year):
        for frame in range(transition_frames):
            years.append(year + frame / transition_frames)
        years.extend([float(year + 1)] * hold_frames)
    years.extend([float(end_year)] * closing_hold)
    return years


def scene_base_layout(
    title: str,
    subtitle: str | None = None,
    year_label: str | None = None,
    title_opacity: float = 1.0,
    subtitle_opacity: float = 1.0,
    background: str = BACKGROUND,
) -> dict:
    annotations: list[dict] = [
        dict(
            x=0.03,
            y=0.965,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            text=title,
            showarrow=False,
            font=dict(
                family=FONT_FAMILY,
                size=28,
                color=rgba(TEXT_PRIMARY, title_opacity),
            ),
        )
    ]
    if subtitle:
        annotations.append(
            dict(
                x=0.03,
                y=0.92,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                text=subtitle,
                showarrow=False,
                font=dict(
                    family=FONT_FAMILY,
                    size=15,
                    color=rgba(TEXT_SECONDARY, subtitle_opacity),
                ),
            )
        )
    if year_label:
        annotations.append(
            dict(
                x=0.97,
                y=0.08,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                text=year_label,
                showarrow=False,
                font=dict(
                    family=FONT_FAMILY,
                    size=54,
                    color=rgba(TEXT_MUTED, 0.34),
                ),
            )
        )

    return dict(
        width=WIDTH,
        height=HEIGHT,
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY),
        annotations=annotations,
    )


def add_dc_marker(
    fig: go.Figure,
    year_position: float,
    dc_size_pivot: pd.DataFrame,
    pulse_phase: float,
    label_opacity: float = 1.0,
) -> None:
    interpolated_size = float(
        interpolate_series(dc_size_pivot, year_position, int(dc_size_pivot.columns.min()), int(dc_size_pivot.columns.max())).loc["DC"]
    )
    pulse = 2.4 * math.sin(pulse_phase * 2.0 * math.pi)
    glow_size = max(16.0, interpolated_size * 1.85 + pulse * 1.6)
    core_size = max(10.0, interpolated_size + pulse)

    fig.add_trace(
        go.Scattergeo(
            lon=[DC_LON],
            lat=[DC_LAT],
            mode="markers",
            hoverinfo="skip",
            marker=dict(
                size=glow_size,
                color=rgba(NEON_RED_GLOW, 0.22),
                line=dict(width=0),
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=[DC_LON],
            lat=[DC_LAT],
            mode="markers+text",
            hoverinfo="skip",
            text=["Washington, DC"],
            textposition="top right",
            textfont=dict(
                family=FONT_FAMILY,
                size=16,
                color=rgba(TEXT_PRIMARY, label_opacity),
            ),
            marker=dict(
                size=core_size,
                color=NEON_RED,
                line=dict(width=1.8, color=rgba("#FFFFFF", 0.9)),
            ),
            showlegend=False,
        )
    )


def build_intro_figure(opacity: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=ALL_STATES,
            locationmode="USA-states",
            z=np.ones(len(ALL_STATES)),
            zmin=0,
            zmax=1,
            colorscale=[[0.0, NEUTRAL_GREY], [1.0, NEUTRAL_GREY]],
            showscale=False,
            marker_line_color=rgba(STATE_LINE, 0.95),
            marker_line_width=0.9,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        scene_base_layout(
            INTRO_TITLE,
            INTRO_SUBTITLE,
            title_opacity=opacity,
            subtitle_opacity=max(0.0, opacity - 0.12),
        )
    )
    fig.update_geos(
        scope="usa",
        bgcolor=BACKGROUND,
        showland=True,
        landcolor=BACKGROUND,
        subunitcolor=rgba(STATE_LINE, 0.8),
        countrycolor=BACKGROUND,
        lakecolor=BACKGROUND,
        showlakes=False,
        showframe=False,
    )
    return fig


def build_baseline_figure(
    zscore_pivot: pd.DataFrame,
    dc_size_pivot: pd.DataFrame,
    year_position: float,
    z_limit: float,
    pulse_phase: float,
    show_dc_marker: bool,
    title: str,
    subtitle: str,
) -> go.Figure:
    interpolated = interpolate_series(
        zscore_pivot,
        year_position,
        int(zscore_pivot.columns.min()),
        int(zscore_pivot.columns.max()),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=ALL_STATES,
            locationmode="USA-states",
            z=interpolated.reindex(ALL_STATES).to_numpy(),
            zmin=-z_limit,
            zmax=z_limit,
            zmid=0,
            colorscale="RdYlBu_r",
            colorbar=dict(
                title=dict(
                    text="z-score",
                    font=dict(color=TEXT_SECONDARY, size=12),
                ),
                tickfont=dict(color=TEXT_SECONDARY, size=11),
                len=0.52,
                thickness=14,
                x=0.965,
                y=0.46,
                outlinewidth=0,
                bgcolor=rgba(PANEL, 0.0),
            ),
            marker_line_color=rgba(STATE_LINE, 0.92),
            marker_line_width=0.8,
            hoverinfo="skip",
        )
    )
    if show_dc_marker:
        add_dc_marker(fig, year_position, dc_size_pivot, pulse_phase, label_opacity=1.0)

    fig.update_layout(
        scene_base_layout(
            title,
            subtitle,
            year_label=str(int(round(year_position))),
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        bgcolor=BACKGROUND,
        showland=True,
        landcolor=BACKGROUND,
        subunitcolor=rgba(STATE_LINE, 0.65),
        lakecolor=BACKGROUND,
        showlakes=False,
        showframe=False,
    )
    return fig


def build_spike_figure(
    spike_pivot: pd.DataFrame,
    dc_size_pivot: pd.DataFrame,
    year_position: float,
    pulse_phase: float,
) -> go.Figure:
    interpolated = interpolate_series(
        spike_pivot,
        year_position,
        int(spike_pivot.columns.min()),
        int(spike_pivot.columns.max()),
    ).reindex(ALL_STATES)

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=ALL_STATES,
            locationmode="USA-states",
            z=interpolated.to_numpy(),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, MUTED_GREY],
                [0.499, MUTED_GREY],
                [0.5, SPIKE_RED],
                [1.0, SPIKE_RED],
            ],
            showscale=False,
            marker_line_color=rgba(STATE_LINE, 0.85),
            marker_line_width=0.8,
            hoverinfo="skip",
        )
    )
    add_dc_marker(fig, year_position, dc_size_pivot, pulse_phase, label_opacity=1.0)
    fig.update_layout(
        scene_base_layout(
            "Spike Contrast View",
            "States at z ≥ 1 turn red; everything else fades to gray",
            year_label=str(int(round(year_position))),
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        bgcolor=BACKGROUND,
        showland=True,
        landcolor=BACKGROUND,
        subunitcolor=rgba(STATE_LINE, 0.65),
        lakecolor=BACKGROUND,
        showlakes=False,
        showframe=False,
    )
    return fig


def build_zoom_figure(
    zscore_pivot: pd.DataFrame,
    dc_size_pivot: pd.DataFrame,
    progress: float,
    z_limit: float,
) -> go.Figure:
    progress = float(np.clip(progress, 0.0, 1.0))
    latest_year = int(zscore_pivot.columns.max())
    latest_values = zscore_pivot[latest_year].reindex(ALL_STATES)

    fade_alpha = max(0.08, 0.72 * (1.0 - progress))
    center_lat = 39.5 + (DC_LAT - 39.5) * progress
    center_lon = -98.35 + (DC_LON + 98.35) * progress
    projection_scale = 0.95 + 7.8 * progress
    lon_half_window = 32.0 - 28.0 * progress
    lat_half_window = 12.5 - 9.3 * progress

    text_alpha = max(0.0, min(1.0, (progress - 0.18) / 0.62))

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=[state for state in ALL_STATES if state != "DC"],
            locationmode="USA-states",
            z=np.zeros(len(ALL_STATES) - 1),
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, rgba(MUTED_GREY, fade_alpha)],
                [1.0, rgba(MUTED_GREY, fade_alpha)],
            ],
            showscale=False,
            marker_line_color=rgba(STATE_LINE, 0.85),
            marker_line_width=0.7,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Choropleth(
            locations=["DC"],
            locationmode="USA-states",
            z=[float(np.clip(latest_values.loc["DC"], -z_limit, z_limit))],
            zmin=-z_limit,
            zmax=z_limit,
            zmid=0,
            colorscale="RdYlBu_r",
            showscale=False,
            marker_line_color=rgba("#FFFFFF", 0.95),
            marker_line_width=1.1,
            hoverinfo="skip",
        )
    )
    add_dc_marker(
        fig,
        float(latest_year),
        dc_size_pivot,
        pulse_phase=progress * 2.2,
        label_opacity=max(0.0, 0.35 + 0.65 * progress),
    )

    fig.update_layout(
        scene_base_layout(
            "DC Focus",
            None,
            year_label=str(latest_year),
        )
    )
    fig.add_annotation(
        x=0.03,
        y=0.90,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=FOCUS_TEXT,
        showarrow=False,
        font=dict(
            family=FONT_FAMILY,
            size=20,
            color=rgba(TEXT_PRIMARY, text_alpha),
        ),
        align="left",
    )
    fig.update_geos(
        scope="north america",
        projection_type="mercator",
        projection_scale=projection_scale,
        center=dict(lat=center_lat, lon=center_lon),
        lataxis_range=[center_lat - lat_half_window, center_lat + lat_half_window],
        lonaxis_range=[center_lon - lon_half_window, center_lon + lon_half_window],
        bgcolor=BACKGROUND,
        showland=True,
        landcolor=rgba(PANEL, 0.96),
        subunitcolor=rgba(STATE_LINE, max(0.12, 0.55 * (1.0 - progress))),
        countrycolor=rgba(STATE_LINE, max(0.10, 0.4 * (1.0 - progress))),
        showlakes=False,
        showframe=False,
        showsubunits=True,
        showcountries=False,
    )
    return fig


def build_outro_figure(primary_opacity: float, secondary_opacity: float, include_optional: bool) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY),
        annotations=[
            dict(
                x=0.5,
                y=0.54,
                xref="paper",
                yref="paper",
                text=OUTRO_TEXT,
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(
                    family=FONT_FAMILY,
                    size=28,
                    color=rgba(TEXT_PRIMARY, primary_opacity),
                ),
            ),
            dict(
                x=0.5,
                y=0.47,
                xref="paper",
                yref="paper",
                text=OUTRO_OPTIONAL if include_optional else "",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(
                    family=FONT_FAMILY,
                    size=17,
                    color=rgba(TEXT_SECONDARY, secondary_opacity if include_optional else 0.0),
                ),
            ),
        ],
    )
    return fig


def write_frame(fig: go.Figure, frame_path: Path, scale: int) -> None:
    fig.write_image(frame_path, width=WIDTH, height=HEIGHT, scale=scale)


def clear_existing_frames(frames_dir: Path) -> None:
    for path in frames_dir.glob("frame_*.png"):
        path.unlink()


def dc_post_2020_spike_flag(panel: pd.DataFrame) -> bool:
    dc_panel = panel[(panel["State"] == "DC") & (panel["Year"] >= 2021)].copy()
    if dc_panel.empty:
        return False
    return bool(dc_panel["spike_flag"].mean() >= 0.5)


def build_cinematic_video(
    merged_df_table: pd.DataFrame,
    frames_dir: str | Path = "frames",
    output_video: str | Path = "risk_video.mp4",
    fps: int = 30,
    scale: int = 2,
) -> dict[str, str | int]:
    ensure_kaleido()
    ensure_ffmpeg()

    panel = prepare_panel(merged_df_table)
    frames_path = Path(frames_dir)
    video_path = Path(output_video)
    frames_path.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    clear_existing_frames(frames_path)

    start_year = int(panel["Year"].min())
    end_year = int(panel["Year"].max())
    zscore_pivot = build_metric_pivot(panel, "risk_zscore")
    spike_pivot = build_metric_pivot(panel, "spike_flag")
    dc_risk = (
        panel.loc[panel["State"] == "DC", ["Year", "risk_per_100k"]]
        .set_index("Year")["risk_per_100k"]
        .reindex(range(start_year, end_year + 1))
        .fillna(0.0)
    )
    dc_sizes = scale_dc_sizes(dc_risk)
    dc_size_pivot = pd.DataFrame([dc_sizes.values], index=["DC"], columns=dc_sizes.index)
    z_limit = max(1.8, float(np.nanquantile(np.abs(panel["risk_zscore"]), 0.96)))

    frame_number = 0

    def save(fig: go.Figure) -> None:
        nonlocal frame_number
        frame_path = frames_path / f"frame_{frame_number:04d}.png"
        write_frame(fig, frame_path, scale=scale)
        frame_number += 1

    intro_frames = 78
    for frame in range(intro_frames):
        opacity = min(1.0, frame / 34.0)
        save(build_intro_figure(opacity=opacity))

    baseline_years = timeline_years(
        start_year,
        end_year,
        transition_frames=18,
        hold_frames=6,
        opening_hold=8,
        closing_hold=14,
    )
    for index, year_position in enumerate(baseline_years):
        pulse_phase = index / max(1, len(baseline_years) - 1)
        save(
            build_baseline_figure(
                zscore_pivot=zscore_pivot,
                dc_size_pivot=dc_size_pivot,
                year_position=year_position,
                z_limit=z_limit,
                pulse_phase=pulse_phase,
                show_dc_marker=False,
                title="National Baseline",
                subtitle="Standardized each year: blue is below average, red is above average",
            )
        )

    emphasis_years = timeline_years(
        start_year,
        end_year,
        transition_frames=16,
        hold_frames=8,
        opening_hold=6,
        closing_hold=18,
    )
    for index, year_position in enumerate(emphasis_years):
        pulse_phase = index / max(1, len(emphasis_years) - 1)
        save(
            build_baseline_figure(
                zscore_pivot=zscore_pivot,
                dc_size_pivot=dc_size_pivot,
                year_position=year_position,
                z_limit=z_limit,
                pulse_phase=pulse_phase,
                show_dc_marker=True,
                title="DC Emphasis",
                subtitle="A persistent marker tracks Washington, DC as risk changes over time",
            )
        )

    spike_years = timeline_years(
        start_year,
        end_year,
        transition_frames=14,
        hold_frames=9,
        opening_hold=6,
        closing_hold=18,
    )
    for index, year_position in enumerate(spike_years):
        pulse_phase = index / max(1, len(spike_years) - 1)
        save(build_spike_figure(spike_pivot, dc_size_pivot, year_position, pulse_phase))

    zoom_frames = 96
    for frame in range(zoom_frames):
        progress = frame / max(1, zoom_frames - 1)
        save(build_zoom_figure(zscore_pivot, dc_size_pivot, progress, z_limit))

    include_optional = dc_post_2020_spike_flag(panel)
    outro_frames = 72
    for frame in range(outro_frames):
        primary_opacity = min(1.0, frame / 24.0)
        secondary_opacity = max(0.0, min(1.0, (frame - 12) / 28.0))
        save(
            build_outro_figure(
                primary_opacity=primary_opacity,
                secondary_opacity=secondary_opacity,
                include_optional=include_optional,
            )
        )

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_path / "frame_%04d.png"),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(ffmpeg_command, check=True)

    return {
        "frames_dir": str(frames_path),
        "output_video": str(video_path),
        "frame_count": frame_number,
        "fps": fps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a cinematic Plotly MP4 showing DC risk spikes versus U.S. states.",
    )
    parser.add_argument(
        "--source",
        choices=["supabase", "csv"],
        default="supabase",
        help="Data source for the merged state-year panel. Defaults to Supabase.",
    )
    parser.add_argument(
        "--input-csv",
        help="CSV with State, Year, and risk_per_100k columns. Used when `--source csv` is selected.",
    )
    parser.add_argument(
        "--write-panel-csv",
        help="Optional path to write the prepared panel after loading from Supabase.",
    )
    parser.add_argument(
        "--frames-dir",
        default="frames",
        help="Directory for exported PNG frames.",
    )
    parser.add_argument(
        "--output-video",
        default="risk_video.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video framerate for ffmpeg export.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Plotly image scale multiplier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_input_panel(
        source=args.source,
        input_csv=args.input_csv,
        write_csv_path=args.write_panel_csv,
    )
    result = build_cinematic_video(
        panel,
        frames_dir=args.frames_dir,
        output_video=args.output_video,
        fps=args.fps,
        scale=args.scale,
    )
    print(
        f"Rendered {result['frame_count']} frames to {result['frames_dir']} "
        f"and exported {result['output_video']} at {result['fps']} fps."
    )


if __name__ == "__main__":
    main()
