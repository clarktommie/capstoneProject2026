from manim import *


class VisualizationOutputsStory(Scene):
    def construct(self):
        self.camera.background_color = "#F8FAFC"

        title = Text(
            "Visualization and Outputs Story",
            font_size=34,
            color="#0F172A",
        ).to_edge(UP)
        subtitle = Text(
            "Built from the notebook ETL, QA, and enrollment review sections",
            font_size=16,
            color="#475569",
        ).next_to(title, DOWN, buff=0.15)
        subtitle.scale_to_fit_width(config.frame_width - 1.0)

        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle, shift=DOWN * 0.2), run_time=2.2)
        self.wait(1.2)

        step_y = 1.4
        steps = VGroup(
            self.process_box("Raw Sources", "Shooting data\nNCES files", "#E2E8F0"),
            self.process_box("Extract", "Load exports\nValidate schema", "#DBEAFE"),
            self.process_box("Transform", "Clean fields\nHarmonize keys", "#DCFCE7"),
            self.process_box("Load", "Supabase tables\nDerived views", "#FEF3C7"),
            self.process_box("Notebook Outputs", "QA checks\nPlots\nModel-ready panel", "#F3E8FF"),
        ).arrange(RIGHT, buff=0.26).scale(0.76).move_to(UP * step_y)

        arrows = VGroup()
        for left_box, right_box in zip(steps[:-1], steps[1:]):
            arrow = Arrow(
                start=left_box.get_right() + RIGHT * 0.05,
                end=right_box.get_left() + LEFT * 0.05,
                buff=0,
                stroke_width=5,
                max_tip_length_to_length_ratio=0.16,
                color="#64748B",
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[FadeIn(box, shift=UP * 0.2) for box in steps], lag_ratio=0.18), run_time=4.0)
        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.18), run_time=2.8)
        self.wait(1.6)

        qa_title = Text("Notebook QA outputs", font_size=26, color="#0F172A")
        qa_title.next_to(steps, DOWN, buff=0.6).align_to(steps, LEFT)

        qa_cards = VGroup(
            self.qa_card("Row Counts", "Compare core\ntables", "#E0F2FE"),
            self.qa_card("Data Types", "Categorical\nNumeric\nDatetime", "#ECFCCB"),
            self.qa_card("Missing Values", "Highlight top\nmissing columns", "#FEF3C7"),
            self.qa_card("Duplicates", "Check exact\nduplicates", "#FCE7F3"),
        ).arrange(RIGHT, buff=0.2).scale(0.82)
        qa_cards.next_to(qa_title, DOWN, aligned_edge=LEFT, buff=0.3)

        self.play(Write(qa_title), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.15) for card in qa_cards], lag_ratio=0.22), run_time=3.8)
        self.wait(1.8)

        anomaly_panel = RoundedRectangle(
            corner_radius=0.22,
            width=12.2,
            height=2.7,
            stroke_color="#B45309",
            stroke_width=3,
            fill_color="#FFF7ED",
            fill_opacity=0.98,
        ).move_to(DOWN * 2.0)

        anomaly_title = Text(
            "Enrollment anomaly review",
            font_size=25,
            color="#9A3412",
        ).move_to(anomaly_panel.get_top() + DOWN * 0.35)

        left_note = VGroup(
            Text("Issue", font_size=18, color="#9A3412"),
            Text("1.31M drop in 2019", font_size=20, color="#0F172A"),
            Text("Did not fit the prior trend", font_size=15, color="#475569"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(anomaly_panel.get_left() + RIGHT * 2.2 + UP * 0.02)

        mid_note = VGroup(
            Text("Diagnosis", font_size=18, color="#9A3412"),
            Text("School counts shifted by era", font_size=20, color="#0F172A"),
            Text("Reporting universe changed", font_size=15, color="#475569"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(DOWN * 2.08)

        right_note = VGroup(
            Text("Action", font_size=18, color="#9A3412"),
            Text("Rebuild with one template", font_size=20, color="#0F172A"),
            Text("Restore comparability", font_size=15, color="#475569"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to(anomaly_panel.get_right() + LEFT * 2.25 + UP * 0.02)

        anomaly_arrows = VGroup(
            Arrow(
                start=left_note.get_right() + RIGHT * 0.1,
                end=mid_note.get_left() + LEFT * 0.1,
                buff=0,
                stroke_width=4,
                color="#C2410C",
            ),
            Arrow(
                start=mid_note.get_right() + RIGHT * 0.1,
                end=right_note.get_left() + LEFT * 0.1,
                buff=0,
                stroke_width=4,
                color="#C2410C",
            ),
        )

        self.play(FadeIn(anomaly_panel), Write(anomaly_title), run_time=1.5)
        self.play(FadeIn(left_note, shift=UP * 0.15), run_time=1.2)
        self.wait(0.8)
        self.play(GrowArrow(anomaly_arrows[0]), FadeIn(mid_note, shift=UP * 0.15), run_time=1.4)
        self.wait(0.8)
        self.play(GrowArrow(anomaly_arrows[1]), FadeIn(right_note, shift=UP * 0.15), run_time=1.4)
        self.wait(1.8)

        final_banner = RoundedRectangle(
            corner_radius=0.2,
            width=12.2,
            height=1.0,
            stroke_color="#1D4ED8",
            stroke_width=2.5,
            fill_color="#EFF6FF",
            fill_opacity=0.98,
        ).to_edge(DOWN, buff=0.3)
        final_text = Text(
            "Notebook takeaway: clear outputs depend on audited inputs, comparable enrollment history, and a consistent 1987+ panel.",
            font_size=17,
            color="#0F172A",
        ).scale_to_fit_width(final_banner.width - 0.4).move_to(final_banner.get_center())

        scene_group = VGroup(steps, arrows, qa_title, qa_cards, anomaly_panel, anomaly_title, left_note, mid_note, right_note, anomaly_arrows)
        self.play(scene_group.animate.shift(UP * 0.35).scale(0.96), run_time=1.8)
        self.play(FadeIn(final_banner), FadeIn(final_text, shift=UP * 0.1), run_time=1.6)
        self.wait(3.2)

    def process_box(self, heading, body, fill_color):
        box = RoundedRectangle(
            corner_radius=0.18,
            width=2.15,
            height=1.55,
            stroke_color="#64748B",
            stroke_width=2.5,
            fill_color=fill_color,
            fill_opacity=0.98,
        )
        title = Text(heading, font_size=18, color="#0F172A").move_to(box.get_top() + DOWN * 0.28)
        desc = Text(body, font_size=13, color="#334155", line_spacing=0.82)
        desc.move_to(box.get_center() + DOWN * 0.1)
        return VGroup(box, title, desc)

    def qa_card(self, heading, body, fill_color):
        card = RoundedRectangle(
            corner_radius=0.16,
            width=2.65,
            height=1.6,
            stroke_color="#CBD5E1",
            stroke_width=2,
            fill_color=fill_color,
            fill_opacity=0.98,
        )
        title = Text(heading, font_size=17, color="#0F172A").move_to(card.get_top() + DOWN * 0.25)
        desc = Text(body, font_size=13, color="#334155", line_spacing=0.82).move_to(card.get_center() + DOWN * 0.06)
        return VGroup(card, title, desc)
