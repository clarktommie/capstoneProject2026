from manim import *


class PolicyBurdenOverlap(Scene):
    def construct(self):
        self.camera.background_color = "#F7F4EA"

        title = Text(
            "Where Risk Burden Meets Policy Coverage",
            font_size=34,
            color="#14213D",
        ).to_edge(UP)
        subtitle = Text(
            "Illustrative stakeholder view using recent high-burden states and dense policy-coverage states",
            font_size=16,
            color="#4A5568",
        ).scale_to_fit_width(config.frame_width - 1.2).next_to(title, DOWN, buff=0.18)

        left_circle = Ellipse(
            width=4.8,
            height=5.6,
            color="#235789",
            fill_color="#4D9DE0",
            fill_opacity=0.48,
            stroke_width=8,
        ).shift(LEFT * 1.6 + DOWN * 0.2)
        right_circle = Ellipse(
            width=4.8,
            height=5.6,
            color="#C05621",
            fill_color="#F6AD55",
            fill_opacity=0.48,
            stroke_width=8,
        ).shift(RIGHT * 1.6 + DOWN * 0.2)

        left_label = Text(
            "High Recent\nBurden States",
            font_size=24,
            color="#0F172A",
            line_spacing=0.8,
        ).move_to(left_circle.get_center() + LEFT * 1.0 + UP * 1.75)
        right_label = Text(
            "Dense Policy\nCoverage States",
            font_size=24,
            color="#0F172A",
            line_spacing=0.8,
        ).move_to(right_circle.get_center() + RIGHT * 1.0 + UP * 1.75)

        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle, shift=DOWN * 0.2))
        self.play(Create(left_circle), Create(right_circle))
        self.play(Write(left_label), Write(right_label))

        burden_only = ["TX", "FL", "OH", "GA"]
        overlap = ["CA", "IL", "NY"]
        policy_only = ["MA", "NJ", "CT", "HI"]

        burden_stack = self.make_state_stack(
            burden_only, left_circle.get_center() + LEFT * 1.3 + DOWN * 0.3, "#173F5F"
        )
        overlap_stack = self.make_state_stack(overlap, DOWN * 0.55, "#2D6A4F")
        policy_stack = self.make_state_stack(
            policy_only, right_circle.get_center() + RIGHT * 1.3 + DOWN * 0.3, "#9C4221"
        )

        burden_caption = Text(
            "Pressure points for intervention",
            font_size=17,
            color="#173F5F",
        ).next_to(burden_stack, DOWN, buff=0.35)
        overlap_caption = Text(
            "Priority comparison group",
            font_size=17,
            color="#2D6A4F",
        ).next_to(overlap_stack, DOWN, buff=0.35)
        policy_caption = Text(
            "Lower burden despite dense laws",
            font_size=17,
            color="#9C4221",
        ).next_to(policy_stack, DOWN, buff=0.35)

        self.play(
            LaggedStart(
                FadeIn(burden_stack, shift=RIGHT * 0.3),
                FadeIn(overlap_stack, shift=UP * 0.3),
                FadeIn(policy_stack, shift=LEFT * 0.3),
                lag_ratio=0.2,
            )
        )
        self.play(
            FadeIn(burden_caption),
            FadeIn(overlap_caption),
            FadeIn(policy_caption),
        )
        self.wait(0.8)

        focus_title = Text(
            "Boolean views for stakeholder framing",
            font_size=27,
            color="#14213D",
        ).to_edge(UP)

        venn_group = VGroup(
            left_circle,
            right_circle,
            left_label,
            right_label,
            burden_stack,
            overlap_stack,
            policy_stack,
            burden_caption,
            overlap_caption,
            policy_caption,
        )
        self.play(
            FadeOut(subtitle),
            Transform(title, focus_title),
            venn_group.animate.scale(0.66).to_edge(LEFT, buff=0.35).shift(DOWN * 0.3),
        )

        base_left = left_circle.copy()
        base_right = right_circle.copy()

        intersection_shape = Intersection(
            base_left, base_right, color="#2D6A4F", fill_color="#80ED99", fill_opacity=0.9
        ).scale(0.3).move_to(RIGHT * 4.0 + UP * 2.0)
        union_shape = Union(
            base_left, base_right, color="#D97706", fill_color="#FDE68A", fill_opacity=0.9
        ).scale(0.3).move_to(RIGHT * 4.0 + UP * 0.55)
        difference_shape = Difference(
            base_left, base_right, color="#1D4ED8", fill_color="#93C5FD", fill_opacity=0.9
        ).scale(0.3).move_to(RIGHT * 2.55 + DOWN * 1.0)
        exclusion_shape = Exclusion(
            base_left, base_right, color="#7C3AED", fill_color="#D8B4FE", fill_opacity=0.9
        ).scale(0.3).move_to(RIGHT * 5.45 + DOWN * 1.0)

        intersection_label = Text("Intersection", font_size=18, color="#0F172A").next_to(
            intersection_shape, UP, buff=0.18
        )
        union_label = Text("Union", font_size=18, color="#0F172A").next_to(
            union_shape, UP, buff=0.18
        )
        difference_label = Text("Difference", font_size=18, color="#0F172A").next_to(
            difference_shape, UP, buff=0.18
        )
        exclusion_label = Text("Exclusion", font_size=18, color="#0F172A").next_to(
            exclusion_shape, UP, buff=0.18
        )

        intersection_text = Text(
            "CA, IL, NY\nShared burden\n+ policy density",
            font_size=14,
            color="#334155",
            line_spacing=0.75,
        ).next_to(intersection_shape, DOWN, buff=0.12)
        union_text = Text(
            "All states needing\nattention or comparison",
            font_size=14,
            color="#334155",
            line_spacing=0.75,
        ).next_to(union_shape, DOWN, buff=0.12)
        difference_text = Text(
            "TX, FL, OH, GA\nBurden without\nmatching density",
            font_size=14,
            color="#334155",
            line_spacing=0.75,
        ).next_to(difference_shape, DOWN, buff=0.12)
        exclusion_text = Text(
            "Non-overlap highlights\ncontrast cases",
            font_size=14,
            color="#334155",
            line_spacing=0.75,
        ).next_to(exclusion_shape, DOWN, buff=0.12)

        self.play(FadeIn(intersection_shape, scale=0.85), Write(intersection_label))
        self.play(FadeIn(intersection_text, shift=UP * 0.1))

        self.play(FadeIn(union_shape, scale=0.85), Write(union_label))
        self.play(FadeIn(union_text, shift=UP * 0.1))

        self.play(FadeIn(difference_shape, scale=0.85), Write(difference_label))
        self.play(FadeIn(difference_text, shift=UP * 0.1))

        self.play(FadeIn(exclusion_shape, scale=0.85), Write(exclusion_label))
        self.play(FadeIn(exclusion_text, shift=UP * 0.1))

        takeaway = Text(
            "Stakeholder message: the overlap shows where high recent incident burden persists even under dense policy environments.",
            font_size=18,
            color="#14213D",
        ).scale_to_fit_width(config.frame_width - 1.0).to_edge(DOWN)
        self.play(FadeIn(takeaway, shift=UP * 0.2))
        self.wait(2)

    def make_state_stack(self, states, center_point, color):
        chips = VGroup()
        for idx, state in enumerate(states):
            chip = RoundedRectangle(
                corner_radius=0.12,
                width=1.0,
                height=0.48,
                fill_color=color,
                fill_opacity=0.95,
                stroke_color=WHITE,
                stroke_width=2,
            )
            label = Text(state, font_size=20, color=WHITE).move_to(chip.get_center())
            group = VGroup(chip, label)
            row = idx % 2
            col = idx // 2
            group.move_to(center_point + RIGHT * (col * 1.1) + DOWN * (row * 0.65))
            chips.add(group)
        return chips
