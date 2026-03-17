from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from manim import *
from supabase import create_client


STATE_ABBREV = {
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


def build_state_year_panel() -> tuple[pd.DataFrame, list[str], int, int]:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=True)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    supabase_client = create_client(supabase_url, supabase_key)

    incident_df = fetch_all_rows(supabase_client, "incident")
    enrollment_df = fetch_all_rows(supabase_client, "enrollment_state_year_mat")

    incident_df["Year"] = pd.to_numeric(incident_df["Year"], errors="coerce")
    incident_df["State"] = incident_df["State"].astype(str).str.upper().str.strip()
    incident_df = incident_df[
        incident_df["Year"].notna()
        & (incident_df["Year"] >= 1987)
        & incident_df["State"].str.match(r"^[A-Z]{2}$", na=False)
    ].copy()

    incident_counts = (
        incident_df.groupby(["State", "Year"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    enrollment_df["state"] = enrollment_df["state"].astype(str).str.upper().str.strip()
    enrollment_df["State"] = enrollment_df["state"].map(STATE_ABBREV)
    enrollment_df["Year"] = pd.to_numeric(enrollment_df["year"], errors="coerce")
    enrollment_df["total_students"] = pd.to_numeric(
        enrollment_df["total_students"], errors="coerce"
    )
    enrollment_df = (
        enrollment_df.dropna(subset=["State", "Year", "total_students"])
        .loc[lambda df: df["Year"] >= 1987]
        .sort_values("total_students", ascending=False)
        .drop_duplicates(["State", "Year"])
    )

    merged = enrollment_df[["State", "Year", "total_students"]].merge(
        incident_counts, on=["State", "Year"], how="left"
    )
    merged["incident_count"] = merged["incident_count"].fillna(0)
    merged["incident_rate_per_100k"] = (
        merged["incident_count"] / merged["total_students"] * 100000
    )

    latest_year = int(merged["Year"].max())
    selected_states = sorted(merged["State"].dropna().unique().tolist())

    year_min = int(merged["Year"].min())
    year_max = latest_year
    full_index = pd.MultiIndex.from_product(
        [selected_states, range(year_min, year_max + 1)], names=["State", "Year"]
    )

    panel = (
        merged.loc[
            merged["State"].isin(selected_states),
            ["State", "Year", "total_students", "incident_count", "incident_rate_per_100k"],
        ]
        .set_index(["State", "Year"])
        .reindex(full_index)
        .reset_index()
    )

    panel["incident_count"] = panel["incident_count"].fillna(0)
    panel["total_students"] = panel.groupby("State")["total_students"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    panel["incident_rate_per_100k"] = (
        panel["incident_count"] / panel["total_students"] * 100000
    )

    return panel, selected_states, year_min, year_max


class StateBubbleMotion(Scene):
    def construct(self):
        self.camera.background_color = "#F8FAFC"

        panel, states, year_min, year_max = build_state_year_panel()
        panel["log_students"] = panel["total_students"].clip(lower=1).map(math.log10)

        latest_year = year_max
        latest_snapshot = panel.loc[panel["Year"] == latest_year].copy()
        latest_snapshot["rank"] = latest_snapshot["incident_rate_per_100k"].rank(
            ascending=False, method="first"
        )
        top_rate_states = (
            latest_snapshot.sort_values("incident_rate_per_100k", ascending=False)
            .head(3)["State"]
            .tolist()
        )
        recent_start = max(year_min, year_max - 4)
        recent_panel = panel.loc[panel["Year"] >= recent_start].copy()
        slope_rows: list[dict[str, float | str]] = []
        for state, state_df in recent_panel.groupby("State"):
            ordered = state_df.sort_values("Year")
            x_vals = ordered["Year"].astype(float).tolist()
            y_vals = ordered["incident_rate_per_100k"].astype(float).tolist()
            if len(x_vals) < 2:
                slope = 0.0
            else:
                x_mean = sum(x_vals) / len(x_vals)
                y_mean = sum(y_vals) / len(y_vals)
                numerator = sum(
                    (xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_vals, y_vals)
                )
                denominator = sum((xv - x_mean) ** 2 for xv in x_vals)
                slope = numerator / denominator if denominator else 0.0
            slope_rows.append({"State": state, "recent_slope": slope})
        slope_df = pd.DataFrame(slope_rows)

        latest_with_slope = latest_snapshot.merge(slope_df, on="State", how="left").fillna(
            {"recent_slope": 0.0}
        )
        large_high_count_state = (
            latest_with_slope.sort_values("incident_count", ascending=False).iloc[0]["State"]
        )
        large_states_for_story = (
            latest_with_slope.sort_values("total_students", ascending=False)
            .head(2)["State"]
            .tolist()
        )
        high_rate_state = (
            latest_with_slope.sort_values("incident_rate_per_100k", ascending=False).iloc[0]["State"]
        )
        fast_rising_state = (
            latest_with_slope.sort_values("recent_slope", ascending=False).iloc[0]["State"]
        )
        stable_candidates = latest_with_slope.loc[
            latest_with_slope["total_students"]
            >= latest_with_slope["total_students"].median()
        ].copy()
        stable_state = (
            stable_candidates.assign(stability_gap=lambda df: (df["recent_slope"]).abs())
            .sort_values(["stability_gap", "incident_count"], ascending=[True, False])
            .iloc[0]["State"]
        )
        mid_candidates = latest_with_slope.loc[
            latest_with_slope["total_students"].between(
                latest_with_slope["total_students"].quantile(0.35),
                latest_with_slope["total_students"].quantile(0.75),
            )
        ].copy()
        mid_sized_state = (
            mid_candidates.sort_values(
                ["recent_slope", "incident_count"], ascending=False
            ).iloc[0]["State"]
            if not mid_candidates.empty
            else latest_with_slope.sort_values(
                ["recent_slope", "incident_count"], ascending=False
            ).iloc[1]["State"]
        )
        focal_states = []
        for state in [
            large_high_count_state,
            high_rate_state,
            fast_rising_state,
            stable_state,
            mid_sized_state,
        ]:
            if state not in focal_states:
                focal_states.append(state)
        labeled_states = set(
            latest_snapshot.sort_values("incident_count", ascending=False).head(12)["State"].tolist()
        )
        top_rate_states_by_year = {
            int(year): set(
                year_df.sort_values("incident_rate_per_100k", ascending=False)
                .head(3)["State"]
                .tolist()
            )
            for year, year_df in panel.groupby("Year")
        }

        records = {
            state: {
                int(row.Year): {
                    "students": float(row.total_students),
                    "log_students": float(row.log_students),
                    "rate": float(row.incident_rate_per_100k),
                    "incidents": float(row.incident_count),
                }
                for row in panel.loc[panel["State"] == state].itertuples()
            }
            for state in states
        }

        x_min = float(panel["log_students"].min())
        x_max = float(panel["log_students"].max())
        x_span = max(x_max - x_min, 1e-6)
        x_padding = max(0.08, x_span * 0.08)
        x_range_min = x_min - x_padding
        x_range_max = x_max + x_padding

        incident_q95 = max(float(panel["incident_count"].quantile(0.95)), 1.0)
        min_radius = 0.08
        max_radius = 0.38
        radius_scale = (max_radius - min_radius) / math.sqrt(incident_q95)

        def radius_from_incidents(incidents: float) -> float:
            return max(
                min_radius,
                min(max_radius, min_radius + math.sqrt(max(incidents, 0.0)) * radius_scale),
            )

        axes_y_length = 5.5
        y_data = panel["incident_rate_per_100k"]
        # Use robust limits so a few extreme years do not flatten visible motion.
        y_min = float(y_data.quantile(0.02))
        y_max = float(y_data.quantile(0.98))
        y_span = max(y_max - y_min, 1e-6)
        y_padding = max(0.22, y_span * 0.08)
        y_range_min = max(0.0, y_min - y_padding)
        y_range_max = y_max + y_padding
        y_step = max((y_range_max - y_range_min) / 5, 0.1)

        title = Text(
            "All 50 States Over Time",
            font_size=34,
            color="#0F172A",
        ).to_edge(UP)
        subtitle = Text(
            "Enrollment on the x-axis, incident rate on the y-axis, bubble size shows incident count",
            font_size=17,
            color="#475569",
        ).next_to(title, DOWN, buff=0.12)
        subtitle.scale_to_fit_width(config.frame_width - 0.8)

        axes = Axes(
            x_range=[x_range_min, x_range_max, max((x_range_max - x_range_min) / 5, 0.2)],
            y_range=[y_range_min, y_range_max, y_step],
            x_length=10.6,
            y_length=axes_y_length,
            axis_config={"color": "#64748B", "font_size": 20},
            tips=False,
        ).shift(DOWN * 0.45)

        plot_left = axes.c2p(x_range_min, y_range_min)[0]
        plot_right = axes.c2p(x_range_max, y_range_min)[0]
        plot_bottom = axes.c2p(x_range_min, y_range_min)[1]
        plot_top = axes.c2p(x_range_min, y_range_max)[1]
        def bounded_center(log_students: float, rate: float, radius: float) -> np.ndarray:
            target = axes.c2p(log_students, rate)
            x_margin = radius
            y_margin = radius
            bounded_x = min(max(target[0], plot_left + x_margin), plot_right - x_margin)
            bounded_y = min(max(target[1], plot_bottom + y_margin), plot_top - y_margin)
            # Guard against degenerate axis spans where margins exceed plot width/height.
            if plot_right - plot_left <= 2 * x_margin:
                bounded_x = (plot_left + plot_right) / 2
            if plot_top - plot_bottom <= 2 * y_margin:
                bounded_y = (plot_bottom + plot_top) / 2
            return np.array([bounded_x, bounded_y, 0.0])

        x_label = Text("State enrollment", font_size=20, color="#334155").next_to(
            axes.x_axis, DOWN, buff=0.4
        )
        y_label = Text("Incidents per 100k", font_size=20, color="#334155").rotate(
            PI / 2
        ).next_to(axes.y_axis, LEFT, buff=0.3)
        log_note = Text("log scale", font_size=13, color="#64748B").next_to(
            x_label, RIGHT, buff=0.2
        )

        def format_enrollment_tick(log_value: float) -> str:
            actual = 10 ** log_value
            if actual >= 1_000_000:
                value = actual / 1_000_000
                if value >= 10 or abs(value - round(value)) < 0.05:
                    return f"{value:.0f}M"
                return f"{value:.1f}M"
            value = actual / 1_000
            if value >= 100 or abs(value - round(value)) < 0.5:
                return f"{value:.0f}K"
            return f"{value:.1f}K"

        tick_values = [
            x_range_min + (x_range_max - x_range_min) * idx / 4
            for idx in range(5)
        ]
        x_ticks = VGroup()
        for tick_value in tick_values:
            tick_line = Line(
                axes.c2p(tick_value, 0),
                axes.c2p(tick_value, y_range_min - 0.04 * (y_range_max - y_range_min)),
                stroke_width=1.5,
                color="#94A3B8",
            )
            tick_label = Text(
                format_enrollment_tick(tick_value),
                font_size=14,
                color="#64748B",
            ).next_to(tick_line, DOWN, buff=0.08)
            x_ticks.add(VGroup(tick_line, tick_label))

        note_box = RoundedRectangle(
            corner_radius=0.18,
            width=2.95,
            height=0.92,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).to_edge(RIGHT, buff=0.45).shift(UP * 2.0)
        note_text = Text(
            "x: enrollment | y: rate",
            font_size=14,
            color="#334155",
            line_spacing=0.82,
        ).move_to(note_box.get_center() + UP * 0.1)
        color_key = VGroup(
            Dot(point=ORIGIN, radius=0.06, color="#DC2626"),
            Text("red = top yearly rate", font_size=12, color="#334155"),
        ).arrange(RIGHT, buff=0.1).move_to(note_box.get_center() + DOWN * 0.2)

        size_box = RoundedRectangle(
            corner_radius=0.18,
            width=2.95,
            height=1.28,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).next_to(note_box, DOWN, buff=0.14)

        size_legend = VGroup()
        size_title = Text("Bubble size = incidents", font_size=13, color="#0F172A")
        size_title.move_to(size_box.get_top() + DOWN * 0.2)
        size_values = [1, 8, 25]
        size_circles = VGroup()
        size_labels = VGroup()
        base_x = size_box.get_left()[0] + 0.62
        for idx, incident_value in enumerate(size_values):
            radius = radius_from_incidents(float(incident_value))
            circle = Circle(
                radius=radius,
                stroke_color="#94A3B8",
                stroke_width=1.1,
                fill_color="#94A3B8",
                fill_opacity=0.28,
            ).move_to(
                np.array([base_x + idx * 0.82, size_box.get_bottom()[1] + 0.34 + radius, 0])
            )
            label = Text(str(incident_value), font_size=11, color="#64748B").next_to(
                circle, DOWN, buff=0.05
            )
            size_circles.add(circle)
            size_labels.add(label)
        size_legend.add(size_title, size_circles, size_labels)

        year_tracker = ValueTracker(float(year_min))
        year_text = Text(
            str(year_min),
            font_size=196,
            color="#94A3B8",
            fill_opacity=0.32,
        ).move_to(axes.c2p((x_range_min + x_range_max) / 2, y_range_max * 0.48))
        year_text.add_updater(
            lambda mob: mob.become(
                Text(
                    str(int(round(year_tracker.get_value()))),
                    font_size=196,
                    color="#94A3B8",
                    fill_opacity=0.32,
                ).move_to(axes.c2p((x_range_min + x_range_max) / 2, y_range_max * 0.48))
            )
        )

        def interpolated_values(state: str, year_value: float) -> tuple[float, float, float]:
            lower_year = max(year_min, min(int(math.floor(year_value)), year_max))
            upper_year = max(year_min, min(int(math.ceil(year_value)), year_max))
            lower = records[state][lower_year]
            upper = records[state][upper_year]
            if upper_year == lower_year:
                alpha = 0.0
            else:
                alpha = (year_value - lower_year) / (upper_year - lower_year)

            log_students = lower["log_students"] + (
                upper["log_students"] - lower["log_students"]
            ) * alpha
            rate = lower["rate"] + (upper["rate"] - lower["rate"]) * alpha
            incidents = lower["incidents"] + (upper["incidents"] - lower["incidents"]) * alpha
            return log_students, rate, incidents

        bubble_mobs = VGroup()
        label_mobs = VGroup()
        bubble_map: dict[str, Circle] = {}
        highlight_states: set[str] = set()
        highlight_strength = ValueTracker(0.0)
        default_fill_color = ManimColor("#4F83CC")
        muted_fill_color = ManimColor("#A8B3C2")
        active_fill_color = ManimColor("#D62828")

        for state in states:
            initial_log_students, initial_rate, initial_incidents = interpolated_values(
                state, year_tracker.get_value()
            )
            initial_radius = radius_from_incidents(initial_incidents)

            bubble = Circle(
                radius=initial_radius,
                stroke_color=WHITE,
                stroke_width=1.0,
                fill_color=default_fill_color,
                fill_opacity=0.66,
            )
            bubble.move_to(
                bounded_center(initial_log_students, initial_rate, initial_radius)
            )
            bubble_map[state] = bubble

            def bubble_updater(mob: Circle, state_name: str = state) -> Circle:
                log_students, rate, incidents = interpolated_values(
                    state_name, year_tracker.get_value()
                )
                radius = radius_from_incidents(incidents)
                emphasis = highlight_strength.get_value()
                has_pause_highlight = bool(highlight_states)
                is_pause_target = state_name in highlight_states

                if has_pause_highlight and is_pause_target:
                    fill_color = interpolate_color(
                        default_fill_color, active_fill_color, 0.45 + 0.55 * emphasis
                    )
                    fill_opacity = 0.66 + 0.32 * emphasis
                    stroke_width = 1.0 + 0.45 * emphasis
                    display_radius = radius * (1 + 0.1 * emphasis)
                elif has_pause_highlight:
                    fill_color = interpolate_color(default_fill_color, muted_fill_color, emphasis)
                    fill_opacity = 0.66 - 0.42 * emphasis
                    stroke_width = max(0.7, 1.0 - 0.2 * emphasis)
                    display_radius = radius
                else:
                    fill_color = default_fill_color
                    fill_opacity = 0.66
                    stroke_width = 1.0
                    display_radius = radius

                mob.set(width=display_radius * 2, height=display_radius * 2)
                # Keep bubbles fully visible inside the plot area.
                mob.move_to(bounded_center(log_students, rate, display_radius))
                mob.set_style(
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_color=WHITE,
                    stroke_width=stroke_width,
                )
                return mob

            bubble.add_updater(bubble_updater)
            bubble_mobs.add(bubble)

            if state in labeled_states:
                label = Text(state, font_size=13, color="#0F172A")
                label.next_to(bubble, UP, buff=0.04)

                def label_updater(mob: Text, state_name: str = state, attached_bubble: Circle = bubble) -> Text:
                    emphasis = highlight_strength.get_value()
                    has_pause_highlight = bool(highlight_states)
                    is_pause_target = state_name in highlight_states
                    if has_pause_highlight and is_pause_target:
                        mob.set_opacity(0.58 + 0.42 * emphasis)
                        mob.set_color("#991B1B")
                    elif has_pause_highlight:
                        mob.set_opacity(0.55 - 0.33 * emphasis)
                        mob.set_color("#475569")
                    else:
                        mob.set_opacity(0.55)
                        mob.set_color("#0F172A")
                    mob.next_to(attached_bubble, UP, buff=0.04)
                    return mob

                label.add_updater(label_updater)
                label_mobs.add(label)

        callout_box = RoundedRectangle(
            corner_radius=0.18,
            width=2.95,
            height=1.18,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).next_to(size_box, DOWN, buff=0.14)
        callout_title = Text(
            "What to read",
            font_size=15,
            color="#0F172A",
        ).move_to(callout_box.get_top() + DOWN * 0.28)
        callout_text = Text(
            "Right = larger enrollment\nHigher = higher rate",
            font_size=13,
            color="#334155",
            line_spacing=0.82,
        ).move_to(callout_box.get_center() + DOWN * 0.06)

        intro_box = RoundedRectangle(
            corner_radius=0.16,
            width=4.9,
            height=0.78,
            stroke_color="#CBD5E1",
            stroke_width=1.8,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).move_to(axes.c2p(x_range_min + 0.48, y_range_max * 0.96))
        intro_text = Text(
            "Watch these higher-rate states first: " + ", ".join(focal_states[:5]),
            font_size=16,
            color="#0F172A",
        ).scale_to_fit_width(intro_box.width - 0.24).move_to(intro_box.get_center())
        intro_focus_states = {state for state in focal_states[:5] if state in bubble_map}

        checkpoint_box = RoundedRectangle(
            corner_radius=0.16,
            width=4.2,
            height=0.72,
            stroke_color="#CBD5E1",
            stroke_width=1.8,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).move_to(axes.c2p(x_range_min + 0.46, y_range_max * 0.83))
        checkpoint_text = Text(
            "",
            font_size=15,
            color="#0F172A",
        ).move_to(checkpoint_box.get_center())

        narrative_box = RoundedRectangle(
            corner_radius=0.16,
            width=4.45,
            height=0.86,
            stroke_color="#CBD5E1",
            stroke_width=1.8,
            fill_color="#FFFFFF",
            fill_opacity=0.96,
        ).move_to(axes.c2p(x_range_min + 0.62, y_range_max * 0.72))
        narrative_text = Text(
            "",
            font_size=15,
            color="#0F172A",
            line_spacing=0.82,
        ).move_to(narrative_box.get_center())

        self.play(
            FadeIn(title, shift=DOWN * 0.2),
            FadeIn(subtitle, shift=DOWN * 0.2),
            run_time=2.4,
        )
        self.play(
            Create(axes),
            FadeIn(x_label),
            FadeIn(y_label),
            FadeIn(log_note),
            FadeIn(x_ticks),
            run_time=2.8,
        )
        self.play(
            FadeIn(note_box),
            FadeIn(note_text),
            FadeIn(color_key),
            FadeIn(size_box),
            FadeIn(size_legend),
            FadeIn(callout_box),
            FadeIn(callout_title),
            FadeIn(callout_text),
            run_time=2.2,
        )
        self.play(FadeIn(year_text), FadeIn(bubble_mobs), FadeIn(label_mobs), run_time=2.4)
        self.play(FadeIn(intro_box), FadeIn(intro_text, shift=UP * 0.08), run_time=1.2)
        highlight_states.clear()
        highlight_states.update(intro_focus_states)
        self.play(highlight_strength.animate.set_value(1.0), run_time=0.7)
        self.wait(1.0)
        self.play(highlight_strength.animate.set_value(0.55), run_time=0.6)
        self.play(FadeOut(intro_box), FadeOut(intro_text), run_time=1.1)

        phase_1_end = float(min(year_min + (year_max - year_min) * 0.35, year_max))
        phase_2_end = float(min(year_min + (year_max - year_min) * 0.7, year_max))

        self.play(year_tracker.animate.set_value(phase_1_end), run_time=12, rate_func=linear)
        checkpoint_text.become(
            Text(
                "The watchlist states stay elevated as time advances.",
                font_size=15,
                color="#0F172A",
            ).move_to(checkpoint_box.get_center())
        )
        narrative_text.become(
            Text(
                "Follow them across the chart:\nstill higher-rate than most peers.",
                font_size=15,
                color="#0F172A",
                line_spacing=0.82,
            ).move_to(narrative_box.get_center())
        )
        first_pause_states = {state for state in intro_focus_states if state in bubble_map}
        highlight_states.clear()
        highlight_states.update(first_pause_states)
        self.play(FadeIn(VGroup()), run_time=0.1)
        self.play(
            FadeIn(checkpoint_box),
            FadeIn(checkpoint_text, shift=UP * 0.05),
            FadeIn(narrative_box),
            FadeIn(narrative_text, shift=UP * 0.05),
            highlight_strength.animate.set_value(1.0),
            run_time=0.9,
        )
        self.wait(1.6)
        self.play(
            FadeOut(checkpoint_box),
            FadeOut(checkpoint_text),
            FadeOut(narrative_box),
            FadeOut(narrative_text),
            highlight_strength.animate.set_value(0.0),
            run_time=0.8,
        )
        highlight_states.clear()
        self.play(year_tracker.animate.set_value(phase_2_end), run_time=10, rate_func=linear)
        checkpoint_text.become(
            Text(
                "Rate differences widen as the years advance.",
                font_size=15,
                color="#0F172A",
            ).move_to(checkpoint_box.get_center())
        )
        narrative_text.become(
            Text(
                "Higher-rate states begin to separate\nfrom the rest of the field.",
                font_size=15,
                color="#0F172A",
                line_spacing=0.82,
            ).move_to(narrative_box.get_center())
        )
        second_pause_states = {
            state
            for state in top_rate_states_by_year.get(int(round(phase_2_end)), set())
            if state in bubble_map
        }
        highlight_states.clear()
        highlight_states.update(second_pause_states)
        self.play(FadeIn(VGroup()), run_time=0.1)
        self.play(
            FadeIn(checkpoint_box),
            FadeIn(checkpoint_text, shift=UP * 0.05),
            FadeIn(narrative_box),
            FadeIn(narrative_text, shift=UP * 0.05),
            highlight_strength.animate.set_value(1.0),
            run_time=0.9,
        )
        self.wait(1.6)
        self.play(
            FadeOut(checkpoint_box),
            FadeOut(checkpoint_text),
            FadeOut(narrative_box),
            FadeOut(narrative_text),
            highlight_strength.animate.set_value(0.0),
            run_time=0.8,
        )
        highlight_states.clear()
        self.play(year_tracker.animate.set_value(float(year_max)), run_time=10, rate_func=linear)
        self.wait(2.0)

        followup_box = RoundedRectangle(
            corner_radius=0.16,
            width=5.1,
            height=0.9,
            stroke_color="#CBD5E1",
            stroke_width=1.8,
            fill_color="#FFFFFF",
            fill_opacity=0.97,
        ).move_to(axes.c2p(x_range_min + 0.75, y_range_max * 0.88))
        followup_text = Text(
            "Follow-up: those same watchlist states still stand out near the top.",
            font_size=15,
            color="#0F172A",
        ).scale_to_fit_width(followup_box.width - 0.28).move_to(followup_box.get_center())
        highlight_states.clear()
        highlight_states.update(intro_focus_states)
        self.play(
            FadeIn(followup_box),
            FadeIn(followup_text, shift=UP * 0.05),
            highlight_strength.animate.set_value(1.0),
            run_time=1.0,
        )
        self.wait(1.8)
        self.play(
            FadeOut(followup_box),
            FadeOut(followup_text),
            highlight_strength.animate.set_value(0.0),
            run_time=0.8,
        )
        highlight_states.clear()

        final_banner = RoundedRectangle(
            corner_radius=0.18,
            width=4.25,
            height=0.92,
            stroke_color="#1D4ED8",
            stroke_width=2.2,
            fill_color="#EFF6FF",
            fill_opacity=0.98,
        ).to_edge(LEFT, buff=0.45).shift(DOWN * 2.75)
        final_text = Text(
            "Rate = height\nIncidents = size",
            font_size=18,
            color="#0F172A",
        ).scale_to_fit_width(final_banner.width - 0.3).move_to(final_banner.get_center())

        final_pause_states = {state for state in focal_states if state in bubble_map}
        highlight_states.clear()
        highlight_states.update(final_pause_states)
        self.play(FadeIn(VGroup()), run_time=0.1)
        self.play(
            highlight_strength.animate.set_value(1.0),
            run_time=0.9,
        )
        self.wait(1.2)
        self.play(
            FadeIn(final_banner),
            FadeIn(final_text, shift=UP * 0.1),
            run_time=1.8,
        )
        self.wait(5.0)
        self.play(highlight_strength.animate.set_value(0.0), run_time=0.9)
        highlight_states.clear()
        self.wait(4.5)
