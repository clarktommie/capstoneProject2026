from pathlib import Path
from textwrap import wrap

from manim import *
from manim import config

config.format = "mp4"
config.pixel_format = "yuv420p"
config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720
config.codec = "libx264"
config.movie_file_extension = ".mp4"
config.transparent = False


class ProjectModelSummaryStory(Scene):
    BG = "#F6F3EC"
    PAPER = "#FFFDF8"
    INK = "#1F2933"
    MUTED = "#52606D"
    LINE = "#D9D2C3"
    ACCENT = "#8D3B2F"
    ACCENT_SOFT = "#F2E3DE"
    GOLD = "#D39B38"
    GREEN = "#2F855A"
    BLUE = "#2B6CB0"
    DIM = "#C9C1B3"

    def construct(self):
        self.camera.background_color = self.BG
        project_root = Path(__file__).resolve().parents[1]
        motion = lambda seconds: seconds * 2.0
        hold = lambda seconds: seconds * 11.0
        overview_text = (
            "We use state-by-year data from 1987 forward to compare incident rates, "
            "total victims, and harm per incident. The clearest pattern is the national "
            "rise in recent years, while most single-policy effects stay small or unclear."
        )
        overview_chip_text = "State-by-year data since 1987; the late-year national trend is clearest."

        self.add(self.make_backdrop())

        title = Text(
            "Project Model Summary",
            font_size=40,
            color=self.ACCENT,
            weight=BOLD,
        ).move_to(UP * 1.05)

        overview = self.text_panel(
            overview_text,
            width=10.3,
            font_size=22,
            panel_height=1.95,
            width_chars=62,
        ).next_to(title, DOWN, buff=0.38)

        self.play(FadeIn(title, shift=UP * 0.25), run_time=motion(1.3))
        self.wait(hold(0.3))
        self.play(FadeIn(overview, shift=DOWN * 0.18), run_time=motion(1.5))
        self.wait(hold(0.8))

        title_target = title.copy().scale(0.72).to_edge(UP, buff=0.35)
        overview_header = self.text_panel(
            overview_chip_text,
            width=5.95,
            font_size=12.5,
            panel_height=0.68,
            width_chars=74,
        ).next_to(title_target, DOWN, buff=0.12).shift(LEFT * 0.95)
        overview_header[0].set_stroke(self.LINE, opacity=0.35)
        overview_header[0].set_fill(self.PAPER, opacity=0.92)
        overview_header[1].set_opacity(0.24)

        phase_tag = self.phase_tag("Models").to_corner(UL, buff=0.4)
        self.play(
            AnimationGroup(
                Transform(title, title_target),
                Transform(overview, overview_header),
                FadeIn(phase_tag, shift=RIGHT * 0.15),
                lag_ratio=0.0,
            ),
            run_time=motion(1.2),
        )

        model_specs = [
            {
                "label": "Model 1",
                "title": "Incident count",
                "target": "Negative binomial",
                "detail": "Used for incident frequency. Question: do policy changes line up with more or fewer incidents?",
                "accent": "#E8D6CE",
            },
            {
                "label": "Model 2",
                "title": "Total victims",
                "target": "Negative binomial",
                "detail": "Used for overall harm. Question: do policy changes line up with more or fewer total victims?",
                "accent": "#E6DCC8",
            },
            {
                "label": "Model 3",
                "title": "Victims per incident",
                "target": "Gamma model",
                "detail": "Used for severity once an incident occurs. Question: do policies line up with higher or lower harm per incident?",
                "accent": "#D9E5DA",
            },
            {
                "label": "Check model",
                "title": "Clustered count check",
                "target": "Poisson clustered",
                "detail": "Used as a stricter comparison. Question: do the broad time and policy patterns point in the same direction?",
                "accent": "#DDE6F2",
            },
        ]

        model_cards = VGroup(
            *[
                self.model_card(
                    spec["label"],
                    spec["title"],
                    spec["target"],
                    spec["accent"],
                )
                for spec in model_specs
            ]
        )
        model_cards.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.32)).move_to(DOWN * 0.3)

        self.play(
            LaggedStart(*[FadeIn(card, shift=UP * 0.18) for card in model_cards], lag_ratio=0.12),
            run_time=motion(2.1),
        )

        for idx, spec in enumerate(model_specs):
            detail = self.detail_strip(spec["detail"]).to_edge(DOWN, buff=0.5)
            focus = SurroundingRectangle(
                model_cards[idx],
                buff=0.12,
                corner_radius=0.16,
                stroke_color=self.GOLD,
                stroke_width=3,
            )
            others = [card.animate.set_opacity(0.26) for card_idx, card in enumerate(model_cards) if card_idx != idx]
            self.play(*others, FadeIn(detail, shift=UP * 0.12), FadeIn(focus), run_time=motion(0.7))
            self.play(Indicate(model_cards[idx], color=self.GOLD, scale_factor=1.03), run_time=motion(1.05))
            self.wait(hold(0.9))
            restore = [card.animate.set_opacity(1.0) for card_idx, card in enumerate(model_cards) if card_idx != idx]
            self.play(*restore, FadeOut(detail, shift=DOWN * 0.08), FadeOut(focus), run_time=motion(0.55))

        model_strip = VGroup(
            self.context_chip("Incidents", model_specs[0]["accent"]),
            self.context_chip("Victims", model_specs[1]["accent"]),
            self.context_chip("Per incident", model_specs[2]["accent"]),
            self.context_chip("Check", model_specs[3]["accent"]),
        )
        model_strip.arrange(RIGHT, buff=0.1)
        model_strip.to_corner(UR, buff=0.42).shift(DOWN * 0.78)

        findings_tag = self.phase_tag("Key Findings").move_to(phase_tag)
        self.play(
            FadeTransform(model_cards, model_strip),
            Transform(phase_tag, findings_tag),
            run_time=motion(1.1),
        )
        model_cards = model_strip

        finding_specs = [
            {
                "eyebrow": "Finding 1",
                "headline": "Later years drive the clearest rise",
                "body": "Incident rates climb most sharply from 2018 through 2024.",
                "accent": self.ACCENT,
                "emphasis": True,
            },
            {
                "eyebrow": "Finding 2",
                "headline": "Total victims follow the same late-year pattern",
                "body": "The surge is not just about more incidents. The harm totals rise late as well.",
                "accent": self.GOLD,
                "emphasis": False,
            },
            {
                "eyebrow": "Finding 3",
                "headline": "Most single-policy estimates stay unclear",
                "body": "In the main count and total-victim models, most policy terms do not separate cleanly from zero.",
                "accent": self.BLUE,
                "emphasis": False,
            },
            {
                "eyebrow": "Finding 4",
                "headline": "One severity signal stands out",
                "body": "The K-12 setting policy is the clearest policy-linked increase in victims per incident. Background checks remain suggestive, not firm.",
                "accent": self.GREEN,
                "emphasis": False,
            },
            {
                "eyebrow": "Finding 5",
                "headline": "Time pattern beats short-run policy pattern",
                "body": "Across the summary, the national time trend is stronger than most individual policy effects.",
                "accent": self.ACCENT,
                "emphasis": True,
            },
        ]

        progress = self.finding_progress(len(finding_specs)).to_edge(DOWN, buff=0.45)
        self.activate_progress(progress, 0)

        finding_card = self.finding_card(**finding_specs[0]).move_to(DOWN * 0.2)

        self.play(FadeIn(finding_card, shift=UP * 0.18), FadeIn(progress, shift=UP * 0.1), run_time=motion(1.0))
        self.play(Indicate(progress[0], color=self.GOLD, scale_factor=1.12), run_time=motion(0.65))
        self.play(Indicate(finding_card, color=self.GOLD, scale_factor=1.03), run_time=motion(1.0))
        self.wait(hold(1.1))

        for idx, spec in enumerate(finding_specs[1:], start=1):
            next_card = self.finding_card(**spec).move_to(finding_card)
            progress_state = self.progress_state(progress, idx)
            self.play(
                Transform(finding_card, next_card),
                *progress_state,
                run_time=motion(1.0),
            )
            if spec["emphasis"]:
                self.play(Indicate(finding_card, color=self.GOLD, scale_factor=1.03), run_time=motion(1.0))
            self.wait(hold(1.0))

        visuals_tag = self.phase_tag("Visual Evidence").move_to(phase_tag)
        self.play(
            Transform(phase_tag, visuals_tag),
            run_time=motion(0.8),
        )

        visual_specs = [
            {
                "eyebrow": "Visual 1",
                "headline": "The national rate turns sharply upward late",
                "body": "This is the report's clearest visual pattern. The right side of the chart carries the story.",
                "focus_note": "Focus on the late-year surge.",
                "thumb": "Rate",
                "path": project_root / "outputs" / "project_model_summary_assets" / "incidents_per_100k_students.png",
                "focus": (0.26, 0.58, 0.86, 0.47),
            },
            {
                "eyebrow": "Visual 2",
                "headline": "Enrollment sets the denominator for fair comparison",
                "body": "The models are comparing rates against student exposure, not just raw counts.",
                "focus_note": "This is the population base.",
                "thumb": "Enroll",
                "path": project_root / "outputs" / "project_model_summary_assets" / "national_enrollment_trend.png",
                "focus": (0.72, 0.42, 0.52, 0.63),
            },
            {
                "eyebrow": "Visual 3",
                "headline": "Incident-count policy terms cluster near the center",
                "body": "Most policy estimates stay close to the no-clear-change line while the time effects stand out more.",
                "focus_note": "Most terms stay near zero.",
                "thumb": "Count",
                "path": project_root / "images" / "model_twfe.png",
                "focus": (0.23, 0.72, 0.36, 0.47),
            },
            {
                "eyebrow": "Visual 4",
                "headline": "The total-victim model repeats the same hierarchy",
                "body": "Again, later-year movement is easier to read than most individual policy coefficients.",
                "focus_note": "The late-period pattern persists.",
                "thumb": "Victims",
                "path": project_root / "images" / "model_severity_twfe.png",
                "focus": (0.60, 0.18, 0.50, 0.80),
            },
            {
                "eyebrow": "Visual 5",
                "headline": "Severity has the clearest policy-specific signal",
                "body": "This is where the K-12 setting policy stands out as the most visible policy-linked increase in harm per incident.",
                "focus_note": "One policy signal breaks away.",
                "thumb": "Severity",
                "path": project_root / "images" / "model_incident_severity.png",
                "focus": (0.58, 0.19, 0.46, 0.81),
            },
        ]

        thumb_rail = RoundedRectangle(
            corner_radius=0.22,
            width=11.0,
            height=1.32,
            stroke_color=self.LINE,
            stroke_width=1.5,
            fill_color=self.PAPER,
            fill_opacity=0.68,
        ).to_edge(DOWN, buff=0.18)

        active_caption = self.visual_caption(
            visual_specs[0]["eyebrow"],
            visual_specs[0]["headline"],
            visual_specs[0]["body"],
            visual_specs[0]["focus_note"],
        ).to_edge(LEFT, buff=0.45).shift(DOWN * 0.08)
        active_panel, active_image = self.image_panel(visual_specs[0]["path"])
        active_panel.to_edge(RIGHT, buff=0.55).shift(DOWN * 0.02)
        active_focus = self.relative_focus_box(active_image, *visual_specs[0]["focus"])

        self.play(
            FadeOut(finding_card, shift=UP * 0.1),
            FadeOut(progress, shift=DOWN * 0.1),
            FadeIn(thumb_rail, shift=UP * 0.08),
            FadeIn(active_caption, shift=RIGHT * 0.15),
            FadeIn(active_panel, shift=LEFT * 0.2),
            run_time=motion(1.15),
        )
        self.play(Create(active_focus), run_time=motion(0.9))
        self.play(Indicate(active_focus, color=self.GOLD, scale_factor=1.02), run_time=motion(1.0))
        self.wait(hold(1.4))

        thumbnails = Group()
        thumb_labels = VGroup()

        for idx, spec in enumerate(visual_specs[1:], start=1):
            thumb_scale = 0.78 / active_panel.height
            thumb_target = self.thumbnail_position(len(thumbnails))
            thumb_label = Text(
                visual_specs[idx - 1]["thumb"],
                font_size=12,
                color=self.MUTED,
            )
            thumb_label.move_to(thumb_target + DOWN * 0.47)

            next_caption = self.visual_caption(
                spec["eyebrow"],
                spec["headline"],
                spec["body"],
                spec["focus_note"],
            ).move_to(active_caption)
            next_panel, next_image = self.image_panel(spec["path"])
            next_panel.move_to(active_panel)
            next_panel.shift(RIGHT * 0.35)
            next_focus = self.relative_focus_box(next_image, *spec["focus"])

            self.play(
                FadeOut(active_focus),
                active_panel.animate.scale(thumb_scale).move_to(thumb_target).set_opacity(0.28),
                FadeIn(thumb_label, shift=UP * 0.05),
                Transform(active_caption, next_caption),
                FadeIn(next_panel, shift=LEFT * 0.18),
                run_time=motion(1.3),
            )

            thumbnails.add(active_panel)
            thumb_labels.add(thumb_label)
            active_panel = next_panel
            active_caption = active_caption
            active_image = next_image
            active_focus = next_focus

            self.play(Create(active_focus), run_time=motion(0.85))
            self.play(Indicate(active_focus, color=self.GOLD, scale_factor=1.02), run_time=motion(1.0))
            self.wait(hold(1.3))

        final_note = self.takeaway_banner(
            "Final takeaway",
            "Recent-year time effects dominate the story more than most single-policy terms.",
        ).move_to(ORIGIN + DOWN * 0.08)

        outro_group = Group(active_panel, active_caption, active_focus, thumbnails, thumb_labels, thumb_rail)
        self.play(
            outro_group.animate.set_opacity(0.16).scale(0.94),
            phase_tag.animate.set_opacity(0.28),
            title.animate.set_opacity(0.35),
            run_time=motion(1.15),
        )
        self.play(FadeIn(final_note, shift=UP * 0.18), run_time=motion(1.2))
        self.play(Indicate(final_note, color=self.GOLD, scale_factor=1.03), run_time=motion(1.2))
        self.wait(hold(3.0))

    def make_backdrop(self):
        left_glow = Circle(
            radius=2.5,
            stroke_width=0,
            fill_color=self.ACCENT_SOFT,
            fill_opacity=0.42,
        ).move_to(LEFT * 6.0 + UP * 3.1)
        right_glow = Circle(
            radius=2.0,
            stroke_width=0,
            fill_color="#E9DED0",
            fill_opacity=0.46,
        ).move_to(RIGHT * 5.8 + DOWN * 2.8)
        base_band = Rectangle(
            width=14.5,
            height=2.2,
            stroke_width=0,
            fill_color=self.PAPER,
            fill_opacity=0.24,
        ).shift(DOWN * 2.75)
        return VGroup(left_glow, right_glow, base_band)

    def phase_tag(self, text):
        label = Text(text, font_size=18, color=self.ACCENT)
        pill = RoundedRectangle(
            corner_radius=0.18,
            width=label.width + 0.5,
            height=0.44,
            stroke_color=self.ACCENT,
            stroke_width=1.2,
            fill_color=self.ACCENT_SOFT,
            fill_opacity=0.94,
        )
        label.move_to(pill)
        return VGroup(pill, label)

    def wrapped_paragraph(self, text, width_chars, font_size, color, width):
        para = Paragraph(
            *wrap(text, width_chars),
            alignment="left",
            font_size=font_size,
            color=color,
            line_spacing=0.9,
        )
        para.scale_to_fit_width(width)
        return para

    def text_panel(self, text, width, font_size, panel_height, width_chars=62):
        panel = RoundedRectangle(
            corner_radius=0.24,
            width=width,
            height=panel_height,
            stroke_color=self.LINE,
            stroke_width=1.8,
            fill_color=self.PAPER,
            fill_opacity=0.97,
        )
        para = self.wrapped_paragraph(
            text,
            width_chars=width_chars,
            font_size=font_size,
            color=self.MUTED,
            width=width - 0.7,
        ).move_to(panel)
        para.align_to(panel.get_left() + RIGHT * 0.35, LEFT)
        return VGroup(panel, para)

    def model_card(self, label, title, target, fill_color):
        card = RoundedRectangle(
            corner_radius=0.18,
            width=3.1,
            height=1.9,
            stroke_color=self.LINE,
            stroke_width=1.8,
            fill_color=fill_color,
            fill_opacity=0.97,
        )
        label_text = Text(label, font_size=14, color=self.MUTED).move_to(card.get_top() + DOWN * 0.22)
        title_text = self.wrapped_paragraph(
            title,
            width_chars=19,
            font_size=20,
            color=self.INK,
            width=2.45,
        ).move_to(card.get_center() + UP * 0.14)
        target_tag = RoundedRectangle(
            corner_radius=0.14,
            width=2.15,
            height=0.38,
            stroke_color=self.LINE,
            stroke_width=1.0,
            fill_color=WHITE,
            fill_opacity=0.84,
        ).move_to(card.get_bottom() + UP * 0.28)
        target_text = Text(target, font_size=14, color=self.ACCENT).move_to(target_tag)
        return VGroup(card, label_text, title_text, target_tag, target_text)

    def context_chip(self, text, fill_color):
        label = Text(text, font_size=12, color=self.MUTED)
        chip = RoundedRectangle(
            corner_radius=0.12,
            width=label.width + 0.42,
            height=0.36,
            stroke_color=self.LINE,
            stroke_width=1.0,
            fill_color=fill_color,
            fill_opacity=0.55,
        )
        label.move_to(chip)
        label.set_opacity(0.55)
        return VGroup(chip, label)

    def detail_strip(self, text):
        strip = RoundedRectangle(
            corner_radius=0.2,
            width=11.2,
            height=0.88,
            stroke_color=self.ACCENT,
            stroke_width=1.6,
            fill_color=self.PAPER,
            fill_opacity=0.98,
        )
        label = self.wrapped_paragraph(
            text,
            width_chars=92,
            font_size=18,
            color=self.INK,
            width=10.5,
        ).move_to(strip)
        return VGroup(strip, label)

    def finding_progress(self, count):
        bars = VGroup()
        for _ in range(count):
            bar = RoundedRectangle(
                corner_radius=0.09,
                width=1.25,
                height=0.16,
                stroke_width=0,
                fill_color=self.DIM,
                fill_opacity=0.55,
            )
            bars.add(bar)
        bars.arrange(RIGHT, buff=0.14)
        return bars

    def activate_progress(self, progress, active_idx):
        for idx, bar in enumerate(progress):
            if idx == active_idx:
                bar.set_fill(self.ACCENT, opacity=0.95)
            else:
                bar.set_fill(self.DIM, opacity=0.45)

    def progress_state(self, progress, active_idx):
        animations = []
        for idx, bar in enumerate(progress):
            if idx == active_idx:
                animations.append(bar.animate.set_fill(self.ACCENT, opacity=0.95))
            else:
                animations.append(bar.animate.set_fill(self.DIM, opacity=0.45))
        return animations

    def finding_card(self, eyebrow, headline, body, accent, emphasis):
        brow = Text(eyebrow.upper(), font_size=15, color=accent)
        title = self.wrapped_paragraph(
            headline,
            width_chars=36,
            font_size=27 if emphasis else 23,
            color=accent if emphasis else self.INK,
            width=7.2,
        )
        body_text = self.wrapped_paragraph(
            body,
            width_chars=48,
            font_size=18.5,
            color=self.MUTED,
            width=7.2,
        )
        content = VGroup(brow, title, body_text).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.18,
        )
        card = RoundedRectangle(
            corner_radius=0.24,
            width=8.25,
            height=max(2.58, content.height + 0.62),
            stroke_color=accent,
            stroke_width=2.2,
            fill_color=self.PAPER,
            fill_opacity=0.98,
        )
        content.move_to(card.get_center())
        content.align_to(card.get_left() + RIGHT * 0.34, LEFT)
        content.shift(UP * 0.05)
        return VGroup(card, brow, title, body_text)

    def visual_caption(self, eyebrow, headline, body, focus_note):
        brow = Text(eyebrow.upper(), font_size=13, color=self.ACCENT)
        title = self.wrapped_paragraph(
            headline,
            width_chars=24,
            font_size=21,
            color=self.INK,
            width=3.6,
        )
        body_text = self.wrapped_paragraph(
            body,
            width_chars=31,
            font_size=15.5,
            color=self.MUTED,
            width=3.6,
        )
        note = RoundedRectangle(
            corner_radius=0.14,
            width=3.2,
            height=0.38,
            stroke_color=self.GOLD,
            stroke_width=1.2,
            fill_color="#FFF7E6",
            fill_opacity=0.95,
        )
        note_text = Text(focus_note, font_size=13, color=self.ACCENT)
        note_text.scale_to_fit_width(note.width - 0.2).move_to(note)
        note_group = VGroup(note, note_text)

        content = VGroup(brow, title, body_text, note_group).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.18,
        )
        panel = RoundedRectangle(
            corner_radius=0.24,
            width=4.5,
            height=max(3.1, content.height + 0.58),
            stroke_color=self.LINE,
            stroke_width=1.6,
            fill_color=self.PAPER,
            fill_opacity=0.98,
        )
        content.move_to(panel.get_center())
        content.align_to(panel.get_left() + RIGHT * 0.28, LEFT)
        content.shift(UP * 0.02)
        return VGroup(panel, brow, title, body_text, note_group)

    def image_panel(self, path):
        image = ImageMobject(str(path))
        image.scale_to_fit_width(5.6)
        image.scale_to_fit_height(4.55)
        frame = RoundedRectangle(
            corner_radius=0.16,
            width=image.width + 0.22,
            height=image.height + 0.22,
            stroke_color=self.LINE,
            stroke_width=1.8,
            fill_color=WHITE,
            fill_opacity=1.0,
        )
        image.move_to(frame.get_center())
        return Group(frame, image), image

    def relative_focus_box(self, image, width_ratio, height_ratio, x_ratio, y_ratio):
        width = image.width * width_ratio
        height = image.height * height_ratio
        x = image.get_left()[0] + image.width * x_ratio
        y = image.get_bottom()[1] + image.height * y_ratio
        box = RoundedRectangle(
            corner_radius=0.1,
            width=width,
            height=height,
            stroke_color=self.GOLD,
            stroke_width=3.2,
            fill_opacity=0,
        ).move_to(RIGHT * x + UP * y)
        glow = box.copy().set_stroke(width=9, opacity=0.18)
        return VGroup(glow, box)

    def thumbnail_position(self, index):
        start_x = -4.2
        spacing = 2.1
        return RIGHT * (start_x + index * spacing) + DOWN * 3.08

    def takeaway_banner(self, eyebrow, message):
        panel = RoundedRectangle(
            corner_radius=0.26,
            width=9.9,
            height=2.25,
            stroke_color=self.ACCENT,
            stroke_width=2.2,
            fill_color=self.PAPER,
            fill_opacity=0.99,
        )
        brow = Text(eyebrow.upper(), font_size=16, color=self.ACCENT).move_to(panel.get_top() + DOWN * 0.27)
        headline = self.wrapped_paragraph(
            message,
            width_chars=52,
            font_size=30,
            color=self.ACCENT,
            width=8.5,
        ).move_to(panel.get_center())
        return VGroup(panel, brow, headline)
