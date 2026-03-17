from pathlib import Path

from manim import *


def load_bias_points() -> list[tuple[str, str]]:
    project_root = Path(__file__).resolve().parents[1]
    bias_file = project_root / "bias_analysis.txt"
    if not bias_file.exists():
        return [
            ("Population normalization", "incident_rate_per_100k = incident_count / total_students * 100000"),
            ("Full state-year panel", "MultiIndex.from_product(...) and reindex(full_index)"),
            ("Zero-incident preservation", 'merged["incident_count"] = merged["incident_count"].fillna(0)'),
            ("Enrollment interpolation", 'panel["total_students"] = ... interpolate(limit_direction="both")'),
            ("Reliable year restriction", "filtering to Year >= 1987"),
            ("Duplicate control", 'drop_duplicates(["State", "Year"])'),
        ]

    points: list[tuple[str, str]] = []
    current_label = ""
    current_behavior = ""
    with bias_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("- "):
                if current_label and current_behavior:
                    points.append((current_label, current_behavior))
                current_label = line[2:].strip()
                current_behavior = ""
            elif not current_behavior:
                current_behavior = line
        if current_label and current_behavior:
            points.append((current_label, current_behavior))
    return points


class BiasBeforeAfterStory(Scene):
    def construct(self):
        self.camera.background_color = "#F8FAFC"
        bias_points = load_bias_points()
        intro_time = 2.4
        panel_time = 2.2
        text_time = 1.5
        swap_time = 1.2
        spotlight_in = 0.9
        spotlight_hold = 3.6
        spotlight_out = 0.7
        beat_hold = 1.8
        final_hold = 5.0

        title = Text(
            "Before and After Bias Control",
            font_size=36,
            color="#0F172A",
        ).to_edge(UP)
        subtitle = Text(
            "How the project made state-year comparisons more fair",
            font_size=18,
            color="#475569",
        ).next_to(title, DOWN, buff=0.14)

        left_header = Text("Before", font_size=28, color="#991B1B")
        right_header = Text("After", font_size=28, color="#166534")
        left_header.move_to(LEFT * 3.2 + UP * 2.4)
        right_header.move_to(RIGHT * 3.2 + UP * 2.4)

        left_panel = RoundedRectangle(
            corner_radius=0.24,
            width=5.7,
            height=5.2,
            stroke_color="#FCA5A5",
            stroke_width=3,
            fill_color="#FEF2F2",
            fill_opacity=0.98,
        ).move_to(LEFT * 3.2 + DOWN * 0.15)
        right_panel = RoundedRectangle(
            corner_radius=0.24,
            width=5.7,
            height=5.2,
            stroke_color="#86EFAC",
            stroke_width=3,
            fill_color="#F0FDF4",
            fill_opacity=0.98,
        ).move_to(RIGHT * 3.2 + DOWN * 0.15)

        connector = Arrow(
            left_panel.get_right() + RIGHT * 0.18,
            right_panel.get_left() + LEFT * 0.18,
            color="#1D4ED8",
            stroke_width=6,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(subtitle, shift=DOWN * 0.15), run_time=intro_time)
        self.play(FadeIn(left_panel), FadeIn(right_panel), Write(left_header), Write(right_header), run_time=intro_time)
        self.play(GrowArrow(connector), run_time=1.1)

        raw_chart = self.before_counts_chart().move_to(left_panel.get_center() + UP * 0.95)
        fair_chart = self.after_rate_chart().move_to(right_panel.get_center() + UP * 0.95)

        raw_panel_text = Text(
            "Raw incident counts make\nlarge states dominate",
            font_size=18,
            color="#7F1D1D",
            line_spacing=0.82,
        ).move_to(left_panel.get_center() + DOWN * 1.55)
        fair_panel_text = Text(
            "Rates per 100k adjust for\nenrollment differences",
            font_size=18,
            color="#166534",
            line_spacing=0.82,
        ).move_to(right_panel.get_center() + DOWN * 1.55)

        self.play(FadeIn(raw_chart, shift=UP * 0.15), FadeIn(fair_chart, shift=UP * 0.15), run_time=panel_time)
        self.play(FadeIn(raw_panel_text), FadeIn(fair_panel_text), run_time=text_time)
        self.wait(beat_hold)

        spotlight = self.make_spotlight_message(
            "The first correction was scale:\nincident_rate_per_100k instead of raw counts."
        )
        self.play(FadeIn(spotlight), run_time=spotlight_in)
        self.wait(spotlight_hold)
        self.play(FadeOut(spotlight), run_time=spotlight_out)

        panel_before = self.unbalanced_panel().move_to(left_panel.get_center() + UP * 0.55)
        panel_after = self.balanced_panel().move_to(right_panel.get_center() + UP * 0.55)
        panel_before_text = Text(
            "Missing years and gaps\nskew the picture",
            font_size=18,
            color="#7F1D1D",
            line_spacing=0.82,
        ).move_to(left_panel.get_center() + DOWN * 1.85)
        panel_after_text = Text(
            "Full state-year panel,\nzero-incident years kept",
            font_size=18,
            color="#166534",
            line_spacing=0.82,
        ).move_to(right_panel.get_center() + DOWN * 1.85)

        self.play(
            FadeOut(raw_chart),
            FadeOut(fair_chart),
            FadeOut(raw_panel_text),
            FadeOut(fair_panel_text),
            run_time=swap_time,
        )
        self.play(FadeIn(panel_before, shift=UP * 0.15), FadeIn(panel_after, shift=UP * 0.15), run_time=panel_time)
        self.play(FadeIn(panel_before_text), FadeIn(panel_after_text), run_time=text_time)
        self.wait(beat_hold)

        spotlight = self.make_spotlight_message(
            "The second correction was coverage:\na complete panel plus fillna(0) for no-incident years."
        )
        self.play(FadeIn(spotlight), run_time=spotlight_in)
        self.wait(spotlight_hold)
        self.play(FadeOut(spotlight), run_time=spotlight_out)

        clean_before = self.messy_pipeline().move_to(left_panel.get_center() + UP * 0.45)
        clean_after = self.cleaned_pipeline().move_to(right_panel.get_center() + UP * 0.45)
        clean_before_text = Text(
            "Inconsistent keys, duplicates,\nand weak year coverage",
            font_size=18,
            color="#7F1D1D",
            line_spacing=0.82,
        ).move_to(left_panel.get_center() + DOWN * 1.9)
        clean_after_text = Text(
            "Uppercase keys, state mapping,\ndeduping, interpolation, 1987+",
            font_size=18,
            color="#166534",
            line_spacing=0.82,
        ).move_to(right_panel.get_center() + DOWN * 1.9)

        self.play(
            FadeOut(panel_before),
            FadeOut(panel_after),
            FadeOut(panel_before_text),
            FadeOut(panel_after_text),
            run_time=swap_time,
        )
        self.play(FadeIn(clean_before, shift=UP * 0.15), FadeIn(clean_after, shift=UP * 0.15), run_time=panel_time)
        self.play(FadeIn(clean_before_text), FadeIn(clean_after_text), run_time=text_time)
        self.wait(beat_hold)

        spotlight = self.make_spotlight_message(
            "The third correction was comparability:\nclean keys, deduped rows, interpolation, and a reliable time window."
        )
        self.play(FadeIn(spotlight), run_time=spotlight_in)
        self.wait(spotlight_hold)
        self.play(FadeOut(spotlight), run_time=spotlight_out)

        model_before = self.naive_model().move_to(left_panel.get_center() + UP * 0.55)
        model_after = self.controlled_model().move_to(right_panel.get_center() + UP * 0.55)
        model_before_text = Text(
            "Raw count model\nmisses exposure and panel effects",
            font_size=18,
            color="#7F1D1D",
            line_spacing=0.82,
        ).move_to(left_panel.get_center() + DOWN * 1.9)
        model_after_text = Text(
            "Offset + state effects + year effects\nmake the comparison fairer",
            font_size=18,
            color="#166534",
            line_spacing=0.82,
        ).move_to(right_panel.get_center() + DOWN * 1.9)

        self.play(
            FadeOut(clean_before),
            FadeOut(clean_after),
            FadeOut(clean_before_text),
            FadeOut(clean_after_text),
            run_time=swap_time,
        )
        self.play(FadeIn(model_before, shift=UP * 0.15), FadeIn(model_after, shift=UP * 0.15), run_time=panel_time)
        self.play(FadeIn(model_before_text), FadeIn(model_after_text), run_time=text_time)
        self.wait(beat_hold)

        spotlight = self.make_spotlight_message(
            "The last correction was modeling:\nenrollment offset plus state and year controls."
        )
        self.play(FadeIn(spotlight), run_time=spotlight_in)
        self.wait(spotlight_hold)
        self.play(FadeOut(spotlight), run_time=spotlight_out)

        fixes_card = self.bias_fix_card(bias_points).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(fixes_card, shift=UP * 0.18), run_time=1.6)
        self.wait(final_hold)

    def before_counts_chart(self) -> VGroup:
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 16, 4],
            x_length=4.0,
            y_length=2.3,
            axis_config={"color": "#94A3B8", "stroke_width": 2},
            tips=False,
        )
        labels = VGroup(
            Text("Small", font_size=14, color="#475569").next_to(axes.c2p(0.75, 0), DOWN, buff=0.12),
            Text("Mid", font_size=14, color="#475569").next_to(axes.c2p(1.5, 0), DOWN, buff=0.12),
            Text("Large", font_size=14, color="#475569").next_to(axes.c2p(2.25, 0), DOWN, buff=0.12),
        )
        bars = VGroup()
        values = [3, 6, 14]
        colors = ["#FCA5A5", "#F87171", "#DC2626"]
        for idx, value in enumerate(values):
            bar = Rectangle(
                width=0.52,
                height=(value / 16) * axes.y_length,
                fill_color=colors[idx],
                fill_opacity=0.95,
                stroke_width=0,
            )
            bar.align_to(axes.c2p(idx + 0.55, 0), DOWN)
            bar.move_to(axes.c2p(idx + 0.65, value / 2))
            bars.add(bar)
        top_note = Text("Counts alone", font_size=16, color="#7F1D1D").next_to(axes, UP, buff=0.15)
        return VGroup(axes, labels, bars, top_note)

    def after_rate_chart(self) -> VGroup:
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 6, 1.5],
            x_length=4.0,
            y_length=2.3,
            axis_config={"color": "#94A3B8", "stroke_width": 2},
            tips=False,
        )
        labels = VGroup(
            Text("Small", font_size=14, color="#475569").next_to(axes.c2p(0.75, 0), DOWN, buff=0.12),
            Text("Mid", font_size=14, color="#475569").next_to(axes.c2p(1.5, 0), DOWN, buff=0.12),
            Text("Large", font_size=14, color="#475569").next_to(axes.c2p(2.25, 0), DOWN, buff=0.12),
        )
        bars = VGroup()
        values = [4.8, 4.1, 3.9]
        colors = ["#4ADE80", "#22C55E", "#16A34A"]
        for idx, value in enumerate(values):
            bar = Rectangle(
                width=0.52,
                height=(value / 6) * axes.y_length,
                fill_color=colors[idx],
                fill_opacity=0.95,
                stroke_width=0,
            )
            bar.align_to(axes.c2p(idx + 0.55, 0), DOWN)
            bar.move_to(axes.c2p(idx + 0.65, value / 2))
            bars.add(bar)
        top_note = Text("Rates per 100k", font_size=16, color="#166534").next_to(axes, UP, buff=0.15)
        return VGroup(axes, labels, bars, top_note)

    def unbalanced_panel(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.16,
            width=4.4,
            height=2.55,
            stroke_color="#FCA5A5",
            stroke_width=2,
            fill_color="#FFF1F2",
            fill_opacity=0.95,
        )
        grid = VGroup()
        for row in range(4):
            for col in range(6):
                cell = Square(side_length=0.38, stroke_color="#CBD5E1", stroke_width=1.2)
                cell.move_to(frame.get_center() + LEFT * 1.0 + RIGHT * col * 0.44 + UP * 0.62 - DOWN * row * 0.46)
                grid.add(cell)
        missing_indices = {1, 4, 8, 11, 17, 20}
        for idx, cell in enumerate(grid):
            if idx in missing_indices:
                mark = Cross(cell, stroke_color="#DC2626", stroke_width=4)
                grid.add(mark)
        row_labels = VGroup(
            Text("CA", font_size=16, color="#475569"),
            Text("TX", font_size=16, color="#475569"),
            Text("FL", font_size=16, color="#475569"),
            Text("OH", font_size=16, color="#475569"),
        )
        for idx, label in enumerate(row_labels):
            label.move_to(frame.get_left() + RIGHT * 0.35 + UP * 0.62 - DOWN * idx * 0.46)
        year_note = Text("State-year rows", font_size=15, color="#7F1D1D").next_to(frame, UP, buff=0.12)
        return VGroup(frame, grid, row_labels, year_note)

    def balanced_panel(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.16,
            width=4.4,
            height=2.55,
            stroke_color="#86EFAC",
            stroke_width=2,
            fill_color="#F0FDF4",
            fill_opacity=0.95,
        )
        grid = VGroup()
        for row in range(4):
            for col in range(6):
                cell = Square(
                    side_length=0.38,
                    stroke_color="#BBF7D0",
                    stroke_width=1.2,
                    fill_color="#4ADE80",
                    fill_opacity=0.55,
                )
                cell.move_to(frame.get_center() + LEFT * 1.0 + RIGHT * col * 0.44 + UP * 0.62 - DOWN * row * 0.46)
                grid.add(cell)
        zero_cells = [2, 9, 14]
        zero_labels = VGroup()
        for idx in zero_cells:
            zero = Text("0", font_size=14, color="#166534").move_to(grid[idx].get_center())
            zero_labels.add(zero)
        row_labels = VGroup(
            Text("CA", font_size=16, color="#475569"),
            Text("TX", font_size=16, color="#475569"),
            Text("FL", font_size=16, color="#475569"),
            Text("OH", font_size=16, color="#475569"),
        )
        for idx, label in enumerate(row_labels):
            label.move_to(frame.get_left() + RIGHT * 0.35 + UP * 0.62 - DOWN * idx * 0.46)
        year_note = Text("Complete panel", font_size=15, color="#166534").next_to(frame, UP, buff=0.12)
        return VGroup(frame, grid, zero_labels, row_labels, year_note)

    def messy_pipeline(self) -> VGroup:
        chips = VGroup(
            self.pipeline_chip("california", "#FECACA"),
            self.pipeline_chip("CA", "#FCA5A5"),
            self.pipeline_chip("2019 twice", "#FCA5A5"),
            self.pipeline_chip("1980s gaps", "#FECACA"),
        ).arrange(DOWN, buff=0.2)
        brace = Brace(chips, LEFT, color="#DC2626")
        label = Text("Merge risk", font_size=16, color="#7F1D1D").next_to(brace, LEFT, buff=0.12)
        return VGroup(chips, brace, label)

    def cleaned_pipeline(self) -> VGroup:
        chips = VGroup(
            self.pipeline_chip("STATE_ABBREV", "#BBF7D0"),
            self.pipeline_chip("drop_duplicates", "#86EFAC"),
            self.pipeline_chip("interpolate", "#86EFAC"),
            self.pipeline_chip("Year >= 1987", "#BBF7D0"),
        ).arrange(DOWN, buff=0.2)
        brace = Brace(chips, LEFT, color="#16A34A")
        label = Text("Comparable panel", font_size=16, color="#166534").next_to(brace, LEFT, buff=0.12)
        return VGroup(chips, brace, label)

    def naive_model(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.16,
            width=4.45,
            height=2.55,
            stroke_color="#FCA5A5",
            stroke_width=2,
            fill_color="#FFF1F2",
            fill_opacity=0.95,
        )
        eq = Text("incidents ~ policy", font_size=24, color="#7F1D1D").move_to(frame.get_center() + UP * 0.45)
        issues = VGroup(
            Text("No enrollment offset", font_size=16, color="#991B1B"),
            Text("No state control", font_size=16, color="#991B1B"),
            Text("No year control", font_size=16, color="#991B1B"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(frame.get_center() + DOWN * 0.45)
        return VGroup(frame, eq, issues)

    def controlled_model(self) -> VGroup:
        frame = RoundedRectangle(
            corner_radius=0.16,
            width=4.45,
            height=2.55,
            stroke_color="#86EFAC",
            stroke_width=2,
            fill_color="#F0FDF4",
            fill_opacity=0.95,
        )
        eq = Text("policy + offset + C(State) + C(Year)", font_size=18, color="#166534")
        eq.scale_to_fit_width(frame.width - 0.35).move_to(frame.get_center() + UP * 0.45)
        wins = VGroup(
            Text("Risk relative to size", font_size=16, color="#166534"),
            Text("Persistent state differences controlled", font_size=16, color="#166534"),
            Text("Common year shocks controlled", font_size=16, color="#166534"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(frame.get_center() + DOWN * 0.45)
        return VGroup(frame, eq, wins)

    def pipeline_chip(self, text: str, fill_color: str) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.14,
            width=2.95,
            height=0.48,
            stroke_color=WHITE,
            stroke_width=1.5,
            fill_color=fill_color,
            fill_opacity=0.98,
        )
        label = Text(text, font_size=16, color="#0F172A").move_to(box.get_center())
        return VGroup(box, label)

    def make_spotlight_message(self, text: str) -> VGroup:
        panel = RoundedRectangle(
            corner_radius=0.18,
            width=5.8,
            height=0.95,
            stroke_color="#1D4ED8",
            stroke_width=2.4,
            fill_color="#EFF6FF",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0.38)
        label = Text(
            text,
            font_size=17,
            color="#0F172A",
            line_spacing=0.85,
        ).scale_to_fit_width(panel.width - 0.3).move_to(panel.get_center())
        return VGroup(panel, label)

    def bias_fix_card(self, bias_points: list[tuple[str, str]]) -> VGroup:
        selected = bias_points[:6]
        card = RoundedRectangle(
            corner_radius=0.18,
            width=12.4,
            height=2.0,
            stroke_color="#BFDBFE",
            stroke_width=2,
            fill_color="#FFFFFF",
            fill_opacity=0.98,
        )
        heading = Text("Implemented fixes in code", font_size=20, color="#1E3A8A")
        heading.move_to(card.get_top() + DOWN * 0.24).align_to(card, LEFT).shift(RIGHT * 0.28)
        items = VGroup()
        for label, behavior in selected:
            item = Text(
                f"{label}: {behavior}",
                font_size=12,
                color="#334155",
            ).scale_to_fit_width(card.width - 0.55)
            items.add(item)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        items.next_to(heading, DOWN, aligned_edge=LEFT, buff=0.12)
        return VGroup(card, heading, items)
