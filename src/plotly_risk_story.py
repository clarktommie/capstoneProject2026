from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


YEARS = (1987, 2025)
POST_2018_YEAR = 2018
OUTPUT_BASENAMES = {
    "national_trend": "01_national_trend",
    "distribution_shift": "02_distribution_shift",
    "dc_vs_rest": "03_dc_vs_rest",
    "volatility_burstiness": "04_volatility_burstiness",
}
DC_ALIASES = {
    "DC",
    "D C",
    "DISTRICT OF COLUMBIA",
    "WASHINGTON DC",
    "WASHINGTON D C",
}
COLORS = {
    "navy": "#143C5A",
    "blue": "#2F6B8A",
    "red": "#B14E3A",
    "gold": "#C99A2E",
    "slate": "#5C6B73",
    "light_blue": "rgba(47, 107, 138, 0.14)",
    "light_red": "rgba(177, 78, 58, 0.10)",
    "grid": "rgba(20, 60, 90, 0.10)",
}


def _normalize_state_label(value: object) -> str:
    normalized = " ".join(
        str(value).upper().replace(".", " ").replace(",", " ").strip().split()
    )
    if normalized in DC_ALIASES:
        return "DC"
    return normalized


def _prepare_panel(merged_df_table: pd.DataFrame) -> pd.DataFrame:
    panel = merged_df_table.copy()

    if "risk_per_100k" not in panel.columns and "incident_rate_per_100k" in panel.columns:
        panel = panel.rename(columns={"incident_rate_per_100k": "risk_per_100k"})

    if "risk_per_100k" not in panel.columns:
        if {"incident_count", "total_students"}.issubset(panel.columns):
            panel["risk_per_100k"] = (
                pd.to_numeric(panel["incident_count"], errors="coerce")
                / pd.to_numeric(panel["total_students"], errors="coerce")
                * 100000.0
            )
        else:
            raise ValueError(
                "merged_df_table must include `risk_per_100k` or both `incident_count` and `total_students`."
            )

    required = {"State", "Year", "risk_per_100k", "incident_count", "total_students"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    panel = panel.loc[:, ["State", "Year", "incident_count", "total_students", "risk_per_100k"]].copy()
    panel["State"] = panel["State"].map(_normalize_state_label)
    panel["Year"] = pd.to_numeric(panel["Year"], errors="coerce")
    panel["incident_count"] = pd.to_numeric(panel["incident_count"], errors="coerce")
    panel["total_students"] = pd.to_numeric(panel["total_students"], errors="coerce")
    panel["risk_per_100k"] = pd.to_numeric(panel["risk_per_100k"], errors="coerce")
    panel = panel.dropna(subset=["State", "Year", "incident_count", "total_students", "risk_per_100k"])
    panel["Year"] = panel["Year"].astype(int)
    panel = panel[panel["Year"].between(YEARS[0], YEARS[1])].copy()
    panel = panel[panel["total_students"] > 0].copy()
    panel = panel.drop_duplicates(
        subset=["State", "Year", "incident_count", "total_students", "risk_per_100k"]
    )

    # Collapse accidental repeated state-year rows without inflating counts.
    panel = (
        panel.groupby(["State", "Year"], as_index=False)
        .agg(
            incident_count=("incident_count", "mean"),
            total_students=("total_students", "mean"),
        )
        .sort_values(["Year", "State"])
        .reset_index(drop=True)
    )
    panel["risk_per_100k"] = panel["incident_count"] / panel["total_students"] * 100000.0

    if panel.empty:
        raise ValueError("No valid rows remain after filtering the panel.")

    return panel


def _apply_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    *,
    xaxis_title: str = "Year",
    height: int = 620,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=25, color="#102A43"),
        ),
        font=dict(size=16, color="#243B53"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=height,
        width=1200,
        margin=dict(t=110, l=85, r=40, b=75),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.88)",
        ),
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        showline=True,
        linewidth=1.2,
        linecolor=COLORS["grid"],
        gridcolor=COLORS["grid"],
        tickfont=dict(size=14),
        title_font=dict(size=16),
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        showline=True,
        linewidth=1.2,
        linecolor=COLORS["grid"],
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(size=14),
        title_font=dict(size=16),
    )
    return fig


def build_national_trend(merged_df_table: pd.DataFrame) -> go.Figure:
    panel = _prepare_panel(merged_df_table)

    national = (
        panel.groupby("Year", as_index=False)
        .agg(
            incident_count=("incident_count", "sum"),
            total_students=("total_students", "sum"),
        )
        .sort_values("Year")
    )
    national["risk_per_100k"] = national["incident_count"] / national["total_students"] * 100000.0

    pre_mean = national.loc[national["Year"] < POST_2018_YEAR, "risk_per_100k"].mean()
    post_mean = national.loc[national["Year"] >= POST_2018_YEAR, "risk_per_100k"].mean()
    multiplier = post_mean / pre_mean if pre_mean and pre_mean > 0 else float("nan")
    peak_row = national.loc[national["risk_per_100k"].idxmax()]

    fig = go.Figure()
    fig.add_vrect(
        x0=POST_2018_YEAR,
        x1=YEARS[1] + 0.5,
        fillcolor=COLORS["light_red"],
        line_width=0,
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=national["Year"],
            y=national["risk_per_100k"],
            mode="lines+markers",
            name="National risk per 100k",
            line=dict(color=COLORS["navy"], width=4),
            marker=dict(size=8, color=COLORS["navy"]),
            hovertemplate="Year %{x}<br>Risk %{y:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=POST_2018_YEAR,
        line_width=2,
        line_dash="dash",
        line_color=COLORS["red"],
    )
    fig.add_annotation(
        x=2021.3,
        y=float(national["risk_per_100k"].max()) * 0.92,
        text=(
            f"Post-2018 average risk is {multiplier:.1f}x the 1987-2017 baseline"
            if pd.notna(multiplier)
            else "Post-2018 risk settles into a visibly higher national regime"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor=COLORS["red"],
        borderwidth=1.2,
        font=dict(size=15, color="#102A43"),
    )
    fig.add_annotation(
        x=peak_row["Year"],
        y=peak_row["risk_per_100k"],
        text=f"Peak year: {int(peak_row['Year'])}",
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-45,
        arrowcolor=COLORS["navy"],
        bgcolor="rgba(255,255,255,0.90)",
    )

    fig.update_xaxes(range=[YEARS[0], YEARS[1]], tick0=YEARS[0], dtick=4)
    return _apply_layout(
        fig,
        "National Risk Rises Sharply After 2018 and Stays Elevated",
        "Risk per 100k students",
    )


def build_distribution_shift(merged_df_table: pd.DataFrame) -> go.Figure:
    panel = _prepare_panel(merged_df_table)
    dist = panel.copy()
    dist["Period"] = dist["Year"].map(
        lambda year: "Pre-2018" if year < POST_2018_YEAR else "2018-2025"
    )
    period_state = (
        dist.groupby(["State", "Period"], as_index=False)["risk_per_100k"]
        .mean()
        .rename(columns={"risk_per_100k": "mean_risk_per_100k"})
    )

    summary = (
        period_state.groupby("Period")["mean_risk_per_100k"]
        .quantile([0.25, 0.5, 0.75, 0.9])
        .unstack()
        .rename(columns={0.25: "q1", 0.5: "median", 0.75: "q3", 0.9: "p90"})
    )
    iqr_pre = summary.loc["Pre-2018", "q3"] - summary.loc["Pre-2018", "q1"]
    iqr_post = summary.loc["2018-2025", "q3"] - summary.loc["2018-2025", "q1"]

    fig = go.Figure()
    for period, color in [("Pre-2018", COLORS["blue"]), ("2018-2025", COLORS["red"])]:
        fig.add_trace(
            go.Violin(
                x=period_state.loc[period_state["Period"] == period, "Period"],
                y=period_state.loc[period_state["Period"] == period, "mean_risk_per_100k"],
                name=period,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.04,
                pointpos=0,
                marker=dict(size=4, opacity=0.45, color=color),
                line_color=color,
                fillcolor=color,
                opacity=0.48,
                hovertemplate=f"{period}<br>State mean risk %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_annotation(
        x=0.5,
        y=float(period_state["mean_risk_per_100k"].quantile(0.98)),
        xref="paper",
        text=(
            f"Median state risk rises from {summary.loc['Pre-2018', 'median']:.2f} to "
            f"{summary.loc['2018-2025', 'median']:.2f}; the IQR widens from {iqr_pre:.2f} to {iqr_post:.2f}"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor=COLORS["slate"],
        borderwidth=1.1,
        font=dict(size=15, color="#102A43"),
    )

    fig.update_xaxes(showgrid=False, title_text="")
    return _apply_layout(
        fig,
        "Post-2018 Risk Shifts Up Across States While the Spread Widens",
        "Mean annual state risk per 100k",
        xaxis_title="",
    )


def build_dc_vs_rest(merged_df_table: pd.DataFrame) -> go.Figure:
    panel = _prepare_panel(merged_df_table)
    dc = (
        panel.loc[panel["State"] == "DC", ["Year", "risk_per_100k"]]
        .groupby("Year", as_index=False)["risk_per_100k"]
        .mean()
        .sort_values("Year")
    )
    if dc.empty:
        raise ValueError("No D.C. rows found after state normalization.")

    others = (
        panel.loc[panel["State"] != "DC", ["Year", "risk_per_100k"]]
        .groupby("Year")["risk_per_100k"]
        .median()
        .reset_index(name="median_risk_other_states")
        .sort_values("Year")
    )
    merged = dc.merge(others, on="Year", how="inner")
    post_2018 = merged.loc[merged["Year"] >= POST_2018_YEAR].copy()
    dc_std = post_2018["risk_per_100k"].std(ddof=0)
    other_std = post_2018["median_risk_other_states"].std(ddof=0)
    volatility_ratio = dc_std / other_std if other_std and other_std > 0 else float("nan")
    dc_peak = merged.loc[merged["risk_per_100k"].idxmax()]

    fig = go.Figure()
    fig.add_vrect(
        x0=POST_2018_YEAR,
        x1=YEARS[1] + 0.5,
        fillcolor=COLORS["light_red"],
        line_width=0,
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            x=merged["Year"],
            y=merged["risk_per_100k"],
            mode="lines+markers",
            name="D.C.",
            line=dict(color=COLORS["red"], width=4),
            marker=dict(size=8, color=COLORS["red"]),
            hovertemplate="D.C.<br>Year %{x}<br>Risk %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged["Year"],
            y=merged["median_risk_other_states"],
            mode="lines",
            name="Median of all other states",
            line=dict(color=COLORS["navy"], width=3, dash="dash"),
            hovertemplate="Other states median<br>Year %{x}<br>Risk %{y:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=POST_2018_YEAR,
        line_width=2,
        line_dash="dash",
        line_color=COLORS["red"],
    )
    fig.add_annotation(
        x=dc_peak["Year"],
        y=dc_peak["risk_per_100k"],
        text=f"D.C. peak: {dc_peak['risk_per_100k']:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-55,
        arrowcolor=COLORS["red"],
        bgcolor="rgba(255,255,255,0.94)",
    )
    fig.add_annotation(
        x=2020.7,
        y=float(merged["risk_per_100k"].max()) * 0.78,
        text=(
            f"Since 2018, D.C.'s volatility is {volatility_ratio:.1f}x the typical-state median trend"
            if pd.notna(volatility_ratio)
            else "D.C. swings sharply while the typical-state median remains comparatively flat"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor=COLORS["red"],
        borderwidth=1.1,
        font=dict(size=15, color="#102A43"),
    )

    fig.update_xaxes(range=[YEARS[0], YEARS[1]], tick0=YEARS[0], dtick=4)
    return _apply_layout(
        fig,
        "D.C. Shows Extreme Swings While the Typical State Stays Relatively Stable",
        "Risk per 100k students",
    )


def build_volatility_burstiness(merged_df_table: pd.DataFrame) -> go.Figure:
    panel = _prepare_panel(merged_df_table)
    post_2018 = panel.loc[panel["Year"] >= POST_2018_YEAR].copy()

    volatility = (
        post_2018.groupby("State", as_index=False)["risk_per_100k"]
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns={"risk_per_100k": "volatility"})
        .sort_values("volatility", ascending=False)
        .reset_index(drop=True)
    )
    top10 = volatility.head(10).sort_values("volatility", ascending=True).reset_index(drop=True)
    bar_colors = [COLORS["red"] if state == "DC" else COLORS["blue"] for state in top10["State"]]

    fig = go.Figure(
        go.Bar(
            x=top10["volatility"],
            y=top10["State"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="white", width=1.2)),
            text=[f"{value:.2f}" for value in top10["volatility"]],
            textposition="outside",
            hovertemplate="%{y}<br>Post-2018 SD %{x:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    if "DC" in top10["State"].values:
        dc_row = top10.loc[top10["State"] == "DC"].iloc[0]
        dc_rank = int(volatility.index[volatility["State"] == "DC"][0] + 1)
        fig.add_annotation(
            x=dc_row["volatility"],
            y=dc_row["State"],
            text=f"D.C. ranks #{dc_rank} in post-2018 volatility",
            showarrow=True,
            arrowhead=2,
            ax=110,
            ay=0,
            arrowcolor=COLORS["red"],
            bgcolor="rgba(255,255,255,0.94)",
        )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(title_text="", categoryorder="array", categoryarray=top10["State"].tolist())
    return _apply_layout(
        fig,
        "Post-2018 Instability Is Concentrated in a Small Group of States",
        "Post-2018 standard deviation of risk per 100k",
        xaxis_title="Standard deviation",
    )


def build_risk_story_figures(merged_df_table: pd.DataFrame) -> dict[str, go.Figure]:
    return {
        "national_trend": build_national_trend(merged_df_table),
        "distribution_shift": build_distribution_shift(merged_df_table),
        "dc_vs_rest": build_dc_vs_rest(merged_df_table),
        "volatility_burstiness": build_volatility_burstiness(merged_df_table),
    }


def write_risk_story_figures(
    figures: dict[str, go.Figure],
    output_dir: str | Path = "outputs/plotly_risk_story",
    *,
    write_png: bool = False,
    image_scale: int = 2,
) -> dict[str, list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {}
    for key, figure in figures.items():
        base_name = OUTPUT_BASENAMES.get(key, key)
        html_path = output_path / f"{base_name}.html"
        figure.write_html(html_path, include_plotlyjs="cdn")
        paths = [html_path]
        if write_png:
            png_path = output_path / f"{base_name}.png"
            figure.write_image(png_path, scale=image_scale)
            paths.append(png_path)
        written[key] = paths
    return written


def _load_panel(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    raise ValueError("Input path must be a .csv, .parquet, or .feather file.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build four presentation-ready Plotly visuals from a merged state-year risk panel."
    )
    parser.add_argument("input_path", help="Path to a CSV, Parquet, or Feather file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/plotly_risk_story",
        help="Directory for exported HTML files.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also export PNG files with Plotly image export.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures after writing them.",
    )
    args = parser.parse_args()

    panel = _load_panel(args.input_path)
    figures = build_risk_story_figures(panel)
    write_risk_story_figures(figures, args.output_dir, write_png=args.write_png)

    if args.show:
        for figure in figures.values():
            figure.show()


if __name__ == "__main__":
    main()
elif "merged_df_table" in globals():
    figures = build_risk_story_figures(merged_df_table)
    fig_national_trend = figures["national_trend"]
    fig_distribution_shift = figures["distribution_shift"]
    fig_dc_vs_rest = figures["dc_vs_rest"]
    fig_volatility_burstiness = figures["volatility_burstiness"]
