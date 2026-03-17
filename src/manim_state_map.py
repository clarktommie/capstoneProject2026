from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from manim import *

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from manim_gapminder import build_state_year_panel


STATE_TILE_POSITIONS = {
    "WA": (0, 0),
    "MT": (2, 0),
    "ND": (4, 0),
    "MN": (5, 0),
    "WI": (6, 0),
    "MI": (7, 0),
    "VT": (10, 0),
    "NH": (11, 0),
    "ME": (12, 0),
    "OR": (0, 1),
    "ID": (1, 1),
    "WY": (2, 1),
    "SD": (4, 1),
    "IA": (5, 1),
    "IL": (6, 1),
    "IN": (7, 1),
    "OH": (8, 1),
    "PA": (9, 1),
    "NY": (10, 1),
    "MA": (11, 1),
    "RI": (12, 1),
    "CA": (0, 2),
    "NV": (1, 2),
    "UT": (2, 2),
    "CO": (3, 2),
    "NE": (4, 2),
    "MO": (5, 2),
    "KY": (7, 2),
    "WV": (8, 2),
    "VA": (9, 2),
    "NJ": (10, 2),
    "CT": (11, 2),
    "AZ": (1, 3),
    "NM": (2, 3),
    "KS": (4, 3),
    "AR": (5, 3),
    "TN": (7, 3),
    "NC": (9, 3),
    "SC": (10, 3),
    "OK": (3, 4),
    "LA": (5, 4),
    "MS": (6, 4),
    "AL": (7, 4),
    "GA": (8, 4),
    "DC": (9, 4),
    "TX": (3, 5),
    "FL": (9, 5),
    "AK": (0, 6),
    "HI": (2, 6),
    "MD": (10, 4),
    "DE": (11, 3),
}


class StateRiskMap(Scene):
    def construct(self):
        self.camera.background_color = "#F8FAFC"

        panel, states, year_min, year_max = build_state_year_panel()
        available_states = [state for state in states if state in STATE_TILE_POSITIONS]
        panel = panel.loc[panel["State"].isin(available_states)].copy()

        latest_year = int(panel["Year"].max())
        latest_snapshot = panel.loc[panel["Year"] == latest_year].copy()
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
                numerator = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x_vals, y_vals))
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
        high_rate_state = (
            latest_with_slope.sort_values("incident_rate_per_100k", ascending=False).iloc[0]["State"]
        )
        fast_rising_state = (
            latest_with_slope.sort_values("recent_slope", ascending=False).iloc[0]["State"]
        )
        stable_candidates = latest_with_slope.loc[
            latest_with_slope["total_students"] >= latest_with_slope["total_students"].median()
        ].copy()
        stable_state = (
            stable_candidates.assign(stability_gap=lambda df: df["recent_slope"].abs())
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
            mid_candidates.sort_values(["recent_slope", "incident_count"], ascending=False).iloc[0]["State"]
            if not mid_candidates.empty
            else latest_with_slope.sort_values(["recent_slope", "incident_count"], ascending=False).iloc[1]["State"]
        )

        focal_states: list[str] = []
        for state in [
            large_high_count_state,
            high_rate_state,
            fast_rising_state,
            stable_state,
            mid_sized_state,
        ]:
            if state not in focal_states:
                focal_states.append(state)

        records = {
            state: {
                int(row.Year): {
                    "rate": float(row.incident_rate_per_100k),
                    "incidents": float(row.incident_count),
                    "students": float(row.total_students),
                }
                for row in panel.loc[panel["State"] == state].itertuples()
            }
            for state in available_states
        }

        rate_low = float(panel["incident_rate_per_100k"].quantile(0.15))
        rate_mid = float(panel["incident_rate_per_100k"].quantile(0.55))
        rate_high = float(panel["incident_rate_per_100k"].quantile(0.9))

        year_tracker = ValueTracker(float(year_min))
        highlight_states: set[str] = set()
        highlight_strength = ValueTracker(0.0)

        title = Text(
            "School Shooting Risk by State",
            font_size=34,
            color="#0F172A",
        ).to_edge(UP)
        subtitle = Text(
            "Tile map of incidents per 100k students, using the same state-year panel as the bubble animation",
            font_size=16,
            color="#475569",
        ).next_to(title, DOWN, buff=0.14)
        subtitle.scale_to_fit_width(config.frame_width - 1.0)

        year_backdrop = Text(
            str(year_min),
            font_size=184,
            color="#CBD5E1",
        ).set_opacity(0.26)
        year_backdrop.move_to(LEFT * 1.2 + DOWN * 0.2)
        year_backdrop.add_updater(
            lambda mob: mob.become(
                Text(
                    str(int(round(year_tracker.get_value()))),
                    font_size=184,
                    color="#CBD5E1",
                )
                .set_opacity(0.26)
                .move_to(LEFT * 1.2 + DOWN * 0.2)
            )
        )

        map_group = VGroup()
        tile_group = VGroup()
        label_group = VGroup()
        tile_lookup: dict[str, Square] = {}

        tile_size = 0.48
        x_offset = -3.7
        y_offset = 2.0

        def tile_center(state: str):
            col, row = STATE_TILE_POSITIONS[state]
            return RIGHT * (x_offset + col * 0.58) + UP * (y_offset - row * 0.58)

        def interpolate_rate(state: str, year_value: float) -> float:
            lower_year = max(year_min, min(int(math.floor(year_value)), year_max))
            upper_year = max(year_min, min(int(math.ceil(year_value)), year_max))
            lower = records[state][lower_year]["rate"]
            upper = records[state][upper_year]["rate"]
            if upper_year == lower_year:
                return lower
            alpha = (year_value - lower_year) / (upper_year - lower_year)
            return lower + (upper - lower) * alpha

        def rate_color(rate_value: float, highlight_value: float, state: str):
            base = interpolate_color(
                interpolate_color(ManimColor("#DBEAFE"), ManimColor("#FDE68A"), inverse_interpolate(rate_low, rate_mid, rate_value)),
                ManimColor("#DC2626"),
                inverse_interpolate(rate_mid, rate_high, rate_value),
            )
            if highlight_value > 0 and state not in highlight_states:
                return interpolate_color(base, ManimColor("#CBD5E1"), 0.68 * highlight_value)
            if highlight_value > 0 and state in highlight_states:
                return interpolate_color(base, ManimColor("#B91C1C"), 0.5 * highlight_value)
            return base

        for state in available_states:
            tile = Square(
                side_length=tile_size,
                stroke_color=WHITE,
                stroke_width=1.8,
                fill_color="#DBEAFE",
                fill_opacity=0.96,
            ).move_to(tile_center(state))

            def tile_updater(mob: Square, state_name: str = state) -> Square:
                rate_value = interpolate_rate(state_name, year_tracker.get_value())
                highlight_value = highlight_strength.get_value()
                scale_factor = 1.0
                opacity = 0.94
                stroke_width = 1.8
                if highlight_value > 0:
                    if state_name in highlight_states:
                        scale_factor = 1.0 + 0.08 * highlight_value
                        opacity = 0.98
                        stroke_width = 2.6
                    else:
                        opacity = 0.28
                        stroke_width = 1.2
                mob.become(
                    Square(
                        side_length=tile_size * scale_factor,
                        stroke_color=WHITE,
                        stroke_width=stroke_width,
                        fill_color=rate_color(rate_value, highlight_value, state_name),
                        fill_opacity=opacity,
                    ).move_to(tile_center(state_name))
                )
                return mob

            tile.add_updater(tile_updater)
            tile_group.add(tile)
            tile_lookup[state] = tile

            label = Text(state, font_size=14, color="#0F172A").move_to(tile.get_center())

            def label_updater(mob: Text, state_name: str = state) -> Text:
                highlight_value = highlight_strength.get_value()
                mob.move_to(tile_lookup[state_name].get_center())
                if highlight_value > 0:
                    mob.set_opacity(1.0 if state_name in highlight_states else 0.35)
                else:
                    mob.set_opacity(0.88)
                return mob

            label.add_updater(label_updater)
            label_group.add(label)

        map_group.add(tile_group, label_group).shift(DOWN * 0.45 + LEFT * 0.15)

        legend = self.build_legend(rate_low, rate_mid, rate_high).to_edge(RIGHT, buff=0.45).shift(UP * 1.7)
        read_box = self.read_box().next_to(legend, DOWN, buff=0.18)
        story_box = self.story_box(focal_states, latest_with_slope).next_to(read_box, DOWN, buff=0.18)

        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(subtitle, shift=DOWN * 0.15), run_time=1.8)
        self.play(FadeIn(year_backdrop), run_time=1.1)
        self.play(FadeIn(map_group, shift=UP * 0.2), FadeIn(legend), FadeIn(read_box), FadeIn(story_box), run_time=2.4)
        self.wait(1.2)

        phase_1_end = float(min(year_min + (year_max - year_min) * 0.35, year_max))
        phase_2_end = float(min(year_min + (year_max - year_min) * 0.7, year_max))

        self.play(year_tracker.animate.set_value(phase_1_end), run_time=10, rate_func=linear)

        large_states = set(
            latest_with_slope.sort_values("total_students", ascending=False).head(3)["State"].tolist()
        )
        self.play(highlight_strength.animate.set_value(1.0), run_time=0.7)
        highlight_states.clear()
        highlight_states.update(large_states)
        self.play(FadeIn(VGroup()), run_time=0.1)
        checkpoint_1 = self.checkpoint_message(
            "Large states anchor the map,\nbut risk is still color, not size."
        )
        self.play(FadeIn(checkpoint_1), run_time=0.7)
        self.wait(2.2)
        self.play(FadeOut(checkpoint_1), run_time=0.6)
        self.play(highlight_strength.animate.set_value(0.0), run_time=0.7)
        highlight_states.clear()

        self.play(year_tracker.animate.set_value(phase_2_end), run_time=9, rate_func=linear)

        high_rate_states = set(
            latest_with_slope.sort_values("incident_rate_per_100k", ascending=False).head(4)["State"].tolist()
        )
        self.play(highlight_strength.animate.set_value(1.0), run_time=0.7)
        highlight_states.clear()
        highlight_states.update(high_rate_states)
        self.play(FadeIn(VGroup()), run_time=0.1)
        checkpoint_2 = self.checkpoint_message(
            "Dark red states have the highest\nincidents per 100k students."
        )
        self.play(FadeIn(checkpoint_2), run_time=0.7)
        self.wait(2.2)
        self.play(FadeOut(checkpoint_2), run_time=0.6)
        self.play(highlight_strength.animate.set_value(0.0), run_time=0.7)
        highlight_states.clear()

        self.play(year_tracker.animate.set_value(float(year_max)), run_time=9, rate_func=linear)

        highlight_states.update(focal_states[:5])
        self.play(highlight_strength.animate.set_value(1.0), run_time=0.8)
        final_box = self.final_message().to_edge(DOWN, buff=0.3)
        self.play(FadeIn(final_box, shift=UP * 0.12), run_time=1.2)
        self.wait(4.0)

    def build_legend(self, rate_low: float, rate_mid: float, rate_high: float) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.18,
            width=3.4,
            height=1.75,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.97,
        )
        title = Text("Risk Scale", font_size=20, color="#0F172A").move_to(frame.get_top() + DOWN * 0.25)

        colors = ["#DBEAFE", "#FDE68A", "#DC2626"]
        values = [rate_low, rate_mid, rate_high]
        labels = VGroup()
        swatches = VGroup()
        for idx, (color, value) in enumerate(zip(colors, values)):
            swatch = RoundedRectangle(
                corner_radius=0.08,
                width=0.42,
                height=0.24,
                stroke_width=0,
                fill_color=color,
                fill_opacity=1,
            ).move_to(frame.get_left() + RIGHT * 0.55 + DOWN * 0.15 + DOWN * idx * 0.34)
            label = Text(
                f"{value:.2f} per 100k",
                font_size=15,
                color="#334155",
            ).next_to(swatch, RIGHT, buff=0.15, aligned_edge=DOWN)
            swatches.add(swatch)
            labels.add(label)
        return VGroup(frame, title, swatches, labels)

    def read_box(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.18,
            width=3.4,
            height=1.35,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.97,
        )
        title = Text("How To Read", font_size=18, color="#0F172A").move_to(frame.get_top() + DOWN * 0.23)
        body = Text(
            "Each tile is a state.\nColor shows incidents per 100k students.",
            font_size=15,
            color="#334155",
            line_spacing=0.82,
        ).move_to(frame.get_center() + DOWN * 0.12)
        return VGroup(frame, title, body)

    def story_box(self, focal_states: list[str], latest_with_slope: pd.DataFrame) -> VGroup:
        labels = []
        if focal_states:
            labels.append(f"High count: {focal_states[0]}")
        if len(focal_states) > 1:
            labels.append(f"High rate: {focal_states[1]}")
        if len(focal_states) > 2:
            labels.append(f"Rising: {focal_states[2]}")
        if len(focal_states) > 3:
            labels.append(f"Stable: {focal_states[3]}")
        if len(focal_states) > 4:
            labels.append(f"Mid-size: {focal_states[4]}")
        frame = RoundedRectangle(
            corner_radius=0.18,
            width=3.4,
            height=2.05,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.97,
        )
        title = Text("Story States", font_size=18, color="#0F172A").move_to(frame.get_top() + DOWN * 0.24)
        body = Text(
            "\n".join(labels[:5]),
            font_size=15,
            color="#334155",
            line_spacing=0.8,
        ).move_to(frame.get_center() + DOWN * 0.08)
        return VGroup(frame, title, body)

    def checkpoint_message(self, text: str) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.18,
            width=5.8,
            height=0.92,
            stroke_color="#1D4ED8",
            stroke_width=2.2,
            fill_color="#EFF6FF",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0.34)
        label = Text(
            text,
            font_size=17,
            color="#0F172A",
            line_spacing=0.82,
        ).scale_to_fit_width(frame.width - 0.28).move_to(frame.get_center())
        return VGroup(frame, label)

    def final_message(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.18,
            width=8.8,
            height=0.92,
            stroke_color="#1D4ED8",
            stroke_width=2.2,
            fill_color="#EFF6FF",
            fill_opacity=0.98,
        )
        label = Text(
            "The map separates where risk is concentrated from where exposure is simply large.",
            font_size=17,
            color="#0F172A",
        ).scale_to_fit_width(frame.width - 0.3).move_to(frame.get_center())
        return VGroup(frame, label)
