#!/usr/bin/env python3
"""Build insight-focused Plotly charts from Supabase data.

Outputs:
- national_risk_regime.html
- state_burden_volatility_quadrants.html
- insights_summary.md
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from supabase import create_client


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {normalized(c): c for c in df.columns}
    for candidate in candidates:
        key = normalized(candidate)
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find any of {candidates} in columns: {list(df.columns)}"
    )


def fetch_all_rows(supabase_client, table_name: str, page_size: int = 1000) -> pd.DataFrame:
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


def compute_slope(values: pd.Series) -> float:
    y = values.dropna().tolist()
    n = len(y)
    if n < 2:
        return 0.0
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_national_chart(
    rate_df: pd.DataFrame, enrollment_df: pd.DataFrame
) -> tuple[go.Figure, list[str]]:
    year_rate_col = pick_col(rate_df, ["year", "Year"])
    rate_col = pick_col(
        rate_df,
        [
            "incidents_per_100k_students",
            "incident_rate_per_100k",
            "incidents_per_100k",
        ],
    )
    incident_count_col = pick_col(rate_df, ["incident_count", "incidents"])

    year_enroll_col = pick_col(enrollment_df, ["year", "Year"])
    enrollment_col = pick_col(
        enrollment_df, ["national_enrollment", "enrollment", "total_students"]
    )

    r = rate_df[[year_rate_col, rate_col, incident_count_col]].copy()
    e = enrollment_df[[year_enroll_col, enrollment_col]].copy()
    r.columns = ["year", "rate_per_100k", "incident_count"]
    e.columns = ["year", "national_enrollment"]

    r["year"] = pd.to_numeric(r["year"], errors="coerce")
    r["rate_per_100k"] = pd.to_numeric(r["rate_per_100k"], errors="coerce")
    r["incident_count"] = pd.to_numeric(r["incident_count"], errors="coerce")
    e["year"] = pd.to_numeric(e["year"], errors="coerce")
    e["national_enrollment"] = pd.to_numeric(e["national_enrollment"], errors="coerce")

    merged = (
        r.merge(e, on="year", how="inner")
        .dropna(subset=["year", "rate_per_100k"])
        .sort_values("year")
        .reset_index(drop=True)
    )

    merged["rate_yoy"] = merged["rate_per_100k"].diff()
    merged["rate_roll3"] = merged["rate_per_100k"].rolling(3, min_periods=1).mean()
    merged["enrollment_yoy_pct"] = merged["national_enrollment"].pct_change() * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Incident Risk Regime: Rate, Momentum, and Event Volume",
            "Enrollment Trend and Year-over-Year Shift",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["rate_per_100k"],
            name="Incidents per 100k",
            mode="lines+markers",
            line=dict(width=2.8, color="#0C6EFD"),
            marker=dict(size=6),
            hovertemplate="Year %{x}<br>Rate %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["rate_roll3"],
            name="3-year rolling rate",
            mode="lines",
            line=dict(width=2, dash="dash", color="#034078"),
            hovertemplate="Year %{x}<br>3Y avg %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    bar_colors = ["#C62828" if v > 0 else "#2E7D32" for v in merged["rate_yoy"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=merged["year"],
            y=merged["rate_yoy"],
            name="YoY change in rate",
            marker_color=bar_colors,
            opacity=0.35,
            hovertemplate="Year %{x}<br>YoY Δ %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["national_enrollment"],
            name="National enrollment",
            mode="lines",
            fill="tozeroy",
            line=dict(width=2, color="#5B8E7D"),
            fillcolor="rgba(91,142,125,0.22)",
            hovertemplate="Year %{x}<br>Enrollment %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=merged["year"],
            y=merged["enrollment_yoy_pct"],
            name="Enrollment YoY %",
            mode="lines+markers",
            line=dict(width=1.8, color="#BC6C25"),
            marker=dict(size=5),
            hovertemplate="Year %{x}<br>YoY %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if not merged["rate_yoy"].dropna().empty:
        top_spike = merged.loc[merged["rate_yoy"].idxmax()]
        top_drop = merged.loc[merged["rate_yoy"].idxmin()]
        fig.add_annotation(
            x=top_spike["year"],
            y=top_spike["rate_per_100k"],
            text=f"Biggest rate jump: {int(top_spike['year'])}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=top_drop["year"],
            y=top_drop["rate_per_100k"],
            text=f"Biggest rate drop: {int(top_drop['year'])}",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=40,
            row=1,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        title="U.S. School Shooting Risk Context (National)",
        height=860,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=100, l=70, r=40, b=60),
    )
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Incidents per 100k", row=1, col=1)
    fig.update_yaxes(title_text="Enrollment / YoY %", row=2, col=1)

    insights: list[str] = []
    if len(merged) > 1:
        latest = merged.iloc[-1]
        prev = merged.iloc[-2]
        latest_rate = float(latest["rate_per_100k"])
        delta_rate = float(latest["rate_per_100k"] - prev["rate_per_100k"])
        latest_enroll_yoy = float(latest["enrollment_yoy_pct"])
        max_year = int(merged.loc[merged["rate_per_100k"].idxmax(), "year"])
        max_rate = float(merged["rate_per_100k"].max())

        insights.append(
            f"Latest year {int(latest['year'])}: rate is {latest_rate:.2f} per 100k "
            f"({'+' if delta_rate >= 0 else ''}{delta_rate:.2f} vs prior year)."
        )
        insights.append(
            f"Peak incident rate in the available series is {max_rate:.2f} per 100k in {max_year}."
        )
        insights.append(
            f"Latest enrollment YoY change is {latest_enroll_yoy:.2f}%, useful for "
            "separating incident-count changes from population changes."
        )

    return fig, insights


def build_state_quadrant_chart(state_year_df: pd.DataFrame) -> tuple[go.Figure, list[str]]:
    state_col = pick_col(state_year_df, ["state", "state_name", "State", "State_Name"])
    year_col = pick_col(state_year_df, ["year", "Year"])
    count_col = pick_col(state_year_df, ["incident_count", "incidents", "count"])

    s = state_year_df[[state_col, year_col, count_col]].copy()
    s.columns = ["state", "year", "incident_count"]
    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s["incident_count"] = pd.to_numeric(s["incident_count"], errors="coerce")
    s = s.dropna(subset=["state", "year", "incident_count"])
    s["state"] = s["state"].astype(str).str.strip()

    latest_year = int(s["year"].max())
    start_recent = latest_year - 4

    full_stats = (
        s.groupby("state", as_index=False)["incident_count"]
        .agg(total_incidents="sum", volatility="std", long_run_mean="mean")
        .fillna({"volatility": 0})
    )

    recent_stats = (
        s.loc[s["year"] >= start_recent]
        .groupby("state", as_index=False)["incident_count"]
        .mean()
        .rename(columns={"incident_count": "recent_5y_mean"})
    )

    slope_stats = (
        s.loc[s["year"] >= start_recent]
        .sort_values(["state", "year"])
        .groupby("state", as_index=False)["incident_count"]
        .apply(compute_slope, include_groups=False)
        .rename(columns={"incident_count": "recent_slope"})
    )

    state_stats = (
        full_stats.merge(recent_stats, on="state", how="left")
        .merge(slope_stats, on="state", how="left")
        .fillna({"recent_5y_mean": 0, "recent_slope": 0})
    )

    x_med = float(state_stats["volatility"].median())
    y_med = float(state_stats["recent_5y_mean"].median())

    def classify(row: pd.Series) -> str:
        high_vol = row["volatility"] >= x_med
        high_burden = row["recent_5y_mean"] >= y_med
        if high_burden and high_vol:
            return "High burden / High volatility"
        if high_burden and not high_vol:
            return "High burden / Stable"
        if not high_burden and high_vol:
            return "Low burden / Volatile"
        return "Lower burden / Stable"

    state_stats["segment"] = state_stats.apply(classify, axis=1)

    fig = go.Figure()
    palette = {
        "High burden / High volatility": "#B22222",
        "High burden / Stable": "#FF8C00",
        "Low burden / Volatile": "#6A5ACD",
        "Lower burden / Stable": "#2E8B57",
    }

    for segment, segment_df in state_stats.groupby("segment"):
        fig.add_trace(
            go.Scatter(
                x=segment_df["volatility"],
                y=segment_df["recent_5y_mean"],
                mode="markers+text",
                text=segment_df["state"],
                textposition="top center",
                name=segment,
                marker=dict(
                    size=(segment_df["total_incidents"] ** 0.5).clip(lower=8) * 2.1,
                    color=palette.get(segment, "#444444"),
                    opacity=0.75,
                    line=dict(width=1, color="white"),
                ),
                customdata=segment_df[["total_incidents", "recent_slope"]],
                hovertemplate=(
                    "State %{text}<br>"
                    "Volatility %{x:.2f}<br>"
                    "Recent 5Y mean %{y:.2f}<br>"
                    "Total incidents %{customdata[0]:.0f}<br>"
                    "Recent slope %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_shape(
        type="line",
        x0=x_med,
        x1=x_med,
        y0=state_stats["recent_5y_mean"].min(),
        y1=state_stats["recent_5y_mean"].max(),
        line=dict(color="#555555", width=1.5, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=state_stats["volatility"].min(),
        x1=state_stats["volatility"].max(),
        y0=y_med,
        y1=y_med,
        line=dict(color="#555555", width=1.5, dash="dot"),
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            "State Risk Prioritization Map: Recent Burden vs Volatility "
            f"(Recent window: {start_recent}-{latest_year})"
        ),
        xaxis_title="Volatility (std dev of annual incidents)",
        yaxis_title="Recent 5-year mean incidents",
        legend_title="Risk segment",
        height=780,
        margin=dict(t=90, l=70, r=30, b=60),
    )

    top_states = (
        state_stats.sort_values(
            by=["recent_5y_mean", "volatility", "total_incidents"], ascending=False
        )
        .head(5)
    )

    insights: list[str] = []
    insights.append(
        "States in the upper-right quadrant combine high recent burden and high volatility; "
        "these are strongest candidates for near-term risk monitoring."
    )
    if not top_states.empty:
        leaders = ", ".join(
            f"{row.state} ({row.recent_5y_mean:.1f} avg, vol {row.volatility:.1f})"
            for row in top_states.itertuples()
        )
        insights.append(f"Top states by recent burden/volatility profile: {leaders}.")

    fast_rising = state_stats.sort_values("recent_slope", ascending=False).head(3)
    if not fast_rising.empty:
        rising = ", ".join(
            f"{row.state} (slope {row.recent_slope:.2f}/yr)" for row in fast_rising.itertuples()
        )
        insights.append(f"Fastest-rising recent trajectories: {rising}.")

    return fig, insights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate insight-focused Plotly charts from Supabase tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plotly_insights"),
        help="Directory to write HTML charts and summary markdown.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    supabase_client = create_client(supabase_url, supabase_key)

    incident_rate_df = fetch_all_rows(supabase_client, "incident_rate_per_100k")
    enrollment_df = fetch_all_rows(supabase_client, "national_enrollment_trend")
    incident_state_year_df = fetch_all_rows(supabase_client, "incident_state_year")

    national_fig, national_insights = build_national_chart(incident_rate_df, enrollment_df)
    state_fig, state_insights = build_state_quadrant_chart(incident_state_year_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    national_path = args.output_dir / "national_risk_regime.html"
    state_path = args.output_dir / "state_burden_volatility_quadrants.html"
    summary_path = args.output_dir / "insights_summary.md"

    national_fig.write_html(national_path, include_plotlyjs="cdn")
    state_fig.write_html(state_path, include_plotlyjs="cdn")

    summary_lines = [
        "# Plotly Insight Summary",
        "",
        "## National Risk Regime",
        *[f"- {line}" for line in national_insights],
        "",
        "## State Burden vs Volatility",
        *[f"- {line}" for line in state_insights],
        "",
        "## Files",
        f"- `{national_path}`",
        f"- `{state_path}`",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Generated insight charts:")
    print(f"- {national_path}")
    print(f"- {state_path}")
    print("Generated summary:")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
