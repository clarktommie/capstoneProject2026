#!/usr/bin/env python3
"""Build a slide-ready data and findings section for the capstone presentation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation_data_section"

INCIDENT_RATE_CHART = ROOT / "outputs" / "project_data_overview_assets" / "incidents_per_100k_students.png"
ENROLLMENT_CHART = ROOT / "outputs" / "project_data_overview_assets" / "national_enrollment_trend.png"
POLICY_SE_CHART = ROOT / "outputs" / "project_model_summary_assets" / "poisson_clustered_policy_coefficients.png"
ENTITY_RELATIONSHIP = ROOT / "images" / "entity_relationship.png"
ETL_OVERVIEW = ROOT / "images" / "Updated_ETL_Overview.png"

WIDTH = 1920
HEIGHT = 1080

COLORS = {
    "bg": "#f4efe6",
    "paper": "#fffdf9",
    "paper_alt": "#f8f2e8",
    "ink": "#22313b",
    "muted": "#60707c",
    "line": "#d6ccbf",
    "accent": "#934730",
    "accent_soft": "#f1ddd3",
    "teal": "#557f86",
    "teal_soft": "#dbe9eb",
    "gold": "#c69332",
    "gold_soft": "#f5ead3",
    "green": "#5d7d59",
    "green_soft": "#e5efe6",
    "red": "#a64c38",
    "red_soft": "#f5e1dc",
    "blue": "#496f9a",
    "blue_soft": "#dde8f5",
    "shadow": "#cfc3b2",
}

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
SERIF = FONT_DIR / "DejaVuSerif.ttf"
SERIF_BOLD = FONT_DIR / "DejaVuSerif-Bold.ttf"

TABLE_COUNTS = [
    ("Incident", 3136, COLORS["accent"]),
    ("Shooter", 3542, COLORS["gold"]),
    ("Victim", 8370, COLORS["teal"]),
    ("Weapon", 3168, COLORS["gold"]),
    ("Enrollment", 2548, COLORS["teal"]),
]

DC_RATE_SERIES = [
    ("2019", 3.90),
    ("2022", 11.18),
    ("2023", 16.36),
    ("2024", 6.72),
    ("2025", 4.03),
]


def load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        path = SERIF_BOLD if bold else SERIF
    else:
        path = SANS_BOLD if bold else SANS
    return ImageFont.truetype(str(path), size)


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def wrap_lines(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    blocks = text.split("\n")
    lines: list[str] = []
    for block in blocks:
        words = block.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if measure(draw, trial, font)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    max_size: int,
    min_size: int,
    bold: bool = False,
    serif: bool = False,
    line_gap: int = 6,
) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    for size in range(max_size, min_size - 1, -2):
        font = load_font(size, bold=bold, serif=serif)
        lines = wrap_lines(draw, text, font, width)
        line_height = measure(draw, "Ag", font)[1]
        total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        if total_height <= height:
            return font, lines, line_height
    font = load_font(min_size, bold=bold, serif=serif)
    lines = wrap_lines(draw, text, font, width)
    line_height = measure(draw, "Ag", font)[1]
    return font, lines, line_height


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    max_size: int,
    min_size: int,
    bold: bool = False,
    serif: bool = False,
    line_gap: int = 6,
) -> None:
    font, lines, line_height = fit_text(
        draw,
        text,
        box,
        max_size=max_size,
        min_size=min_size,
        bold=bold,
        serif=serif,
        line_gap=line_gap,
    )
    y = box[1]
    for line in lines:
        draw.text((box[0], y), line, font=font, fill=fill)
        y += line_height + line_gap


def make_canvas() -> Image.Image:
    base = Image.new("RGBA", (WIDTH, HEIGHT), rgba(COLORS["bg"]))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((-180, -180, 520, 420), fill=rgba(COLORS["accent_soft"], 170))
    draw.ellipse((1450, -140, 2080, 520), fill=rgba(COLORS["teal_soft"], 150))
    draw.rectangle((0, HEIGHT - 150, WIDTH, HEIGHT), fill=rgba("#efe8dc", 160))
    return Image.alpha_composite(base, overlay)


def add_panel(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str | None = None,
    outline: str | None = None,
    radius: int = 30,
    shadow_blur: int = 16,
    shadow_offset: tuple[int, int] = (0, 12),
) -> None:
    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    dx, dy = shadow_offset
    sdraw.rounded_rectangle(
        (x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        radius=radius,
        fill=rgba(COLORS["shadow"], 150),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
    base.alpha_composite(shadow)

    panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    pdraw.rounded_rectangle(
        box,
        radius=radius,
        fill=rgba(fill or COLORS["paper"]),
        outline=rgba(outline or COLORS["line"]),
        width=2,
    )
    base.alpha_composite(panel)


def draw_badge(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    font = load_font(23, bold=True)
    text_w, text_h = measure(draw, text, font)
    x, y = xy
    draw.rounded_rectangle(
        (x, y, x + text_w + 38, y + text_h + 20),
        radius=24,
        fill=COLORS["accent_soft"],
        outline=COLORS["accent"],
        width=1,
    )
    draw.text((x + 19, y + 9), text, font=font, fill=COLORS["accent"])


def draw_header(base: Image.Image, badge: str, title: str, subtitle: str) -> None:
    draw = ImageDraw.Draw(base)
    draw_badge(draw, badge, (72, 54))
    draw.text((72, 138), title, font=load_font(64, bold=True, serif=True), fill=COLORS["accent"])
    draw_text_block(
        draw,
        subtitle,
        (72, 225, 1520, 304),
        fill=COLORS["muted"],
        max_size=28,
        min_size=24,
        line_gap=8,
    )
    draw.line((72, 312, WIDTH - 72, 312), fill=COLORS["line"], width=2)


def draw_card(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    body: str,
    accent: str,
    fill: str | None = None,
    title_size: int = 28,
    body_max: int = 24,
    body_min: int = 18,
    body_top: int = 102,
) -> None:
    add_panel(base, box, fill=fill or COLORS["paper"], outline=accent, radius=28, shadow_blur=12)
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 18, y0 + 18, x0 + 96, y0 + 42), radius=12, fill=accent)
    draw.text((x0 + 24, y0 + 58), title, font=load_font(title_size, bold=True), fill=COLORS["ink"])
    draw_text_block(
        draw,
        body,
        (x0 + 24, y0 + body_top, x1 - 24, y1 - 24),
        fill=COLORS["muted"],
        max_size=body_max,
        min_size=body_min,
        line_gap=5,
    )


def draw_bullet_panel(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    bullets: list[str],
    accent: str,
    fill: str | None = None,
    bullet_max: int = 18,
    bullet_min: int = 15,
) -> None:
    add_panel(base, box, fill=fill or COLORS["paper"], outline=COLORS["line"], radius=28, shadow_blur=10)
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.text((x0 + 24, y0 + 22), title, font=load_font(28, bold=True, serif=True), fill=COLORS["ink"])
    top = y0 + 72
    bottom = y1 - 18
    row_height = max(26, (bottom - top) // max(1, len(bullets)))
    bullet_font = load_font(19, bold=True)
    for bullet in bullets:
        draw.text((x0 + 26, top - 1), "\u2022", font=bullet_font, fill=accent)
        draw_text_block(
            draw,
            bullet,
            (x0 + 56, top, x1 - 24, top + row_height - 8),
            fill=COLORS["muted"],
            max_size=bullet_max,
            min_size=bullet_min,
            line_gap=4,
        )
        top += row_height


def draw_chip(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    fill: str,
    outline: str,
    text_fill: str | None = None,
) -> int:
    font = load_font(22, bold=True)
    text_w, text_h = measure(draw, text, font)
    x, y = xy
    width = text_w + 32
    draw.rounded_rectangle((x, y, x + width, y + text_h + 16), radius=20, fill=fill, outline=outline, width=2)
    draw.text((x + 16, y + 7), text, font=font, fill=text_fill or COLORS["ink"])
    return width


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 18 * direction, ey - 10), (ex - 18 * direction, ey + 10)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 10, ey - 18 * direction), (ex + 10, ey - 18 * direction)]
    draw.polygon(points, fill=color)


def draw_stat_box(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    value: str,
    label: str,
    accent: str,
    fill: str,
    value_size: int = 32,
    label_size: int = 17,
) -> None:
    add_panel(base, box, fill=fill, outline=accent, radius=24, shadow_blur=8, shadow_offset=(0, 5))
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.text((x0 + 18, y0 + 16), value, font=load_font(value_size, bold=True), fill=accent)
    draw_text_block(
        draw,
        label,
        (x0 + 18, y0 + 60, x1 - 18, y1 - 16),
        fill=COLORS["muted"],
        max_size=label_size,
        min_size=max(13, label_size - 4),
        line_gap=4,
    )


def fit_image(path: Path, target: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    image.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", target, rgba("#ffffff"))
    offset = ((target[0] - image.width) // 2, (target[1] - image.height) // 2)
    canvas.alpha_composite(image, offset)
    return canvas


def draw_bar_chart(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle: str,
    series: list[tuple[str, float]],
    accent: str,
) -> None:
    add_panel(base, box, fill=COLORS["paper"], outline=COLORS["line"], radius=30, shadow_blur=10)
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.text((x0 + 28, y0 + 22), title, font=load_font(30, bold=True, serif=True), fill=COLORS["ink"])
    draw.text((x0 + 28, y0 + 64), subtitle, font=load_font(19), fill=COLORS["muted"])

    chart_left = x0 + 72
    chart_right = x1 - 52
    chart_top = y0 + 148
    chart_bottom = y1 - 86
    chart_height = chart_bottom - chart_top

    max_value = max(value for _, value in series)
    step = 5
    grid_top = ((int(max_value) // step) + 1) * step
    if grid_top < max_value:
        grid_top += step
    grid_top = max(grid_top, step)

    for tick in range(0, grid_top + step, step):
        y = chart_bottom - int((tick / grid_top) * chart_height)
        draw.line((chart_left, y, chart_right, y), fill="#e8dfd2", width=2)
        label = f"{tick:g}"
        draw.text((chart_left - 42, y - 11), label, font=load_font(16), fill=COLORS["muted"])

    count = len(series)
    gap = 28
    total_gap = gap * (count - 1)
    bar_width = (chart_right - chart_left - total_gap) // count

    for idx, (label, value) in enumerate(series):
        bar_x0 = chart_left + idx * (bar_width + gap)
        bar_x1 = bar_x0 + bar_width
        bar_height = int((value / grid_top) * chart_height)
        bar_y0 = chart_bottom - bar_height
        color = COLORS["accent"] if idx == 2 else accent
        draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, chart_bottom), radius=18, fill=color)
        value_text = f"{value:.2f}"
        value_font = load_font(18, bold=True)
        value_w, _ = measure(draw, value_text, value_font)
        draw.text((bar_x0 + (bar_width - value_w) / 2, bar_y0 - 28), value_text, font=value_font, fill=color)
        label_font = load_font(18, bold=True)
        label_w, _ = measure(draw, label, label_font)
        draw.text((bar_x0 + (bar_width - label_w) / 2, chart_bottom + 18), label, font=label_font, fill=COLORS["ink"])

    draw.text((chart_left, y1 - 46), "Incident rate per 100,000 students", font=load_font(18, bold=True), fill=COLORS["accent"])


def draw_line_chart(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle: str,
    series: list[tuple[str, float]],
    accent: str,
) -> None:
    add_panel(base, box, fill=COLORS["paper"], outline=COLORS["line"], radius=30, shadow_blur=10)
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    draw.text((x0 + 28, y0 + 22), title, font=load_font(30, bold=True, serif=True), fill=COLORS["ink"])
    draw.text((x0 + 28, y0 + 64), subtitle, font=load_font(19), fill=COLORS["muted"])

    chart_left = x0 + 76
    chart_right = x1 - 44
    chart_top = y0 + 148
    chart_bottom = y1 - 96
    chart_height = chart_bottom - chart_top
    chart_width = chart_right - chart_left

    max_value = max(value for _, value in series)
    grid_top = max(20.0, ((int(max_value) // 5) + 1) * 5.0)
    steps = int(grid_top // 5)
    for tick in range(steps + 1):
        value = tick * 5
        y = chart_bottom - int((value / grid_top) * chart_height)
        draw.line((chart_left, y, chart_right, y), fill="#e8dfd2", width=2)
        draw.text((chart_left - 40, y - 11), f"{value:g}", font=load_font(16), fill=COLORS["muted"])

    point_count = len(series)
    xs: list[int] = []
    for idx in range(point_count):
        if point_count == 1:
            x = chart_left + chart_width // 2
        else:
            x = chart_left + int(idx * chart_width / (point_count - 1))
        xs.append(x)

    points: list[tuple[int, int]] = []
    for x, (label, value) in zip(xs, series, strict=True):
        y = chart_bottom - int((value / grid_top) * chart_height)
        points.append((x, y))
        draw.text((x - 18, chart_bottom + 16), label, font=load_font(18, bold=True), fill=COLORS["ink"])

    for idx in range(len(points) - 1):
        draw.line((points[idx], points[idx + 1]), fill=accent, width=5)

    for idx, ((x, y), (label, value)) in enumerate(zip(points, series, strict=True)):
        color = COLORS["accent"] if value == max_value else accent
        radius = 11 if value == max_value else 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#ffffff", width=3)
        value_font = load_font(18, bold=True)
        value_w, _ = measure(draw, f"{value:.2f}", value_font)
        draw.text((x - value_w / 2, y - 34), f"{value:.2f}", font=value_font, fill=color)

    peak_x, peak_y = points[max(range(len(series)), key=lambda i: series[i][1])]
    draw.rounded_rectangle((peak_x - 84, chart_top - 6, peak_x + 84, chart_top + 28), radius=16, fill=COLORS["accent_soft"])
    draw.text((peak_x - 65, chart_top + 2), "2023 peak", font=load_font(18, bold=True), fill=COLORS["accent"])
    draw.text((chart_left, y1 - 52), "Selected annual D.C. incident rates per 100,000 students", font=load_font(18, bold=True), fill=COLORS["accent"])


def build_slide_1() -> Path:
    slide = make_canvas()
    draw_header(
        slide,
        "Data Structure",
        "Core Data Relationships",
        "The project combines event-level detail with state-year context. Incidents anchor the relational tables, while enrollment and policy joins create the panel used for risk analysis.",
    )

    add_panel(slide, (72, 354, 1305, 938), fill=COLORS["paper"], outline=COLORS["line"], radius=34)
    relationship = fit_image(ENTITY_RELATIONSHIP, (1180, 530))
    slide.alpha_composite(relationship, (96, 386))

    draw = ImageDraw.Draw(slide)
    draw_bullet_panel(
        slide,
        (1340, 354, 1848, 618),
        title="Structure of the data",
        bullets=[
            "The incident table is the central event record for the project.",
            "Shooter, victim, and weapon records add incident-level detail through Incident_ID.",
            "Enrollment and policy tables attach state-year context for normalized comparisons.",
        ],
        accent=COLORS["accent"],
        fill=COLORS["paper"],
        bullet_max=18,
        bullet_min=15,
    )
    draw_stat_box(
        slide,
        (1340, 650, 1585, 808),
        value="3,136",
        label="Incident rows in the anchor table.",
        accent=COLORS["accent"],
        fill=COLORS["paper_alt"],
    )
    draw_stat_box(
        slide,
        (1603, 650, 1848, 808),
        value="2,548",
        label="Enrollment state-year rows used for exposure.",
        accent=COLORS["teal"],
        fill="#f8fbfc",
    )
    add_panel(slide, (1340, 834, 1848, 938), fill="#fffaf4", outline=COLORS["gold"], radius=28, shadow_blur=12)
    draw.rounded_rectangle((1360, 852, 1438, 876), radius=12, fill=COLORS["gold"])
    draw.text((1364, 890), "Analytic consequence", font=load_font(19, bold=True), fill=COLORS["ink"])
    draw_text_block(
        draw,
        "State-year joins turn event records into a normalized risk panel.",
        (1364, 916, 1824, 934),
        fill=COLORS["muted"],
        max_size=12,
        min_size=11,
        line_gap=2,
    )

    ribbon_box = (72, 968, 1848, 1044)
    add_panel(slide, ribbon_box, fill=COLORS["paper_alt"], outline=COLORS["line"], radius=22, shadow_blur=8, shadow_offset=(0, 5))
    draw.text((100, 986), "Key rate used throughout the project: incident_count / total_students x 100,000", font=load_font(27, bold=True), fill=COLORS["accent"])

    output = OUT_DIR / "01_data_structure_overview.png"
    slide.convert("RGB").save(output, quality=95)
    return output


def build_slide_2() -> Path:
    slide = make_canvas()
    draw_header(
        slide,
        "Data Credibility",
        "Analytic Pipeline And Quality Controls",
        "The repo does substantial cleaning before modeling. It standardizes keys, rebuilds the enrollment series for comparability, preserves zero-incident years, and constructs a cleaner state-year panel.",
    )

    draw = ImageDraw.Draw(slide)
    add_panel(slide, (72, 354, 1185, 926), fill=COLORS["paper"], outline=COLORS["line"], radius=34)
    pipeline = fit_image(ETL_OVERVIEW, (1060, 500))
    slide.alpha_composite(pipeline, (98, 390))

    draw_bullet_panel(
        slide,
        (1225, 354, 1848, 604),
        title="What strengthens the panel",
        bullets=[
            "State and year keys are harmonized across incident, enrollment, and policy sources.",
            "The enrollment break around 2019 is treated as a comparability issue and rebuilt into a consistent series.",
            "Zero-incident years remain in the panel, preventing low-incident states from disappearing.",
        ],
        accent=COLORS["accent"],
        fill=COLORS["paper"],
        bullet_max=18,
        bullet_min=15,
    )
    draw_bullet_panel(
        slide,
        (1225, 636, 1848, 926),
        title="Analytical implications",
        bullets=[
            "The main modeling window is 1987-2025, not the full historical record, because comparability matters more than raw span.",
            "Incidents are analyzed as risk per student, not raw volume, using enrollment-normalized rates and log offsets.",
            "The outputs support descriptive, panel-based inference rather than a causal policy claim.",
        ],
        accent=COLORS["teal"],
        fill="#f8fbfc",
        bullet_max=17,
        bullet_min=14,
    )
    draw_stat_box(
        slide,
        (72, 954, 400, 1056),
        value="1966-2025",
        label="Overall incident history available in the repo outputs.",
        accent=COLORS["accent"],
        fill=COLORS["paper_alt"],
        value_size=30,
        label_size=15,
    )
    draw_stat_box(
        slide,
        (430, 954, 758, 1056),
        value="1987-2025",
        label="Core analytic window used for the cleaned state-year panel.",
        accent=COLORS["teal"],
        fill="#f8fbfc",
        value_size=30,
        label_size=15,
    )
    draw_stat_box(
        slide,
        (788, 954, 1116, 1056),
        value="51",
        label="States in the panel, including D.C.",
        accent=COLORS["gold"],
        fill="#fffaf4",
        value_size=36,
        label_size=15,
    )
    coverage_box = (1146, 946, 1848, 1056)
    add_panel(slide, coverage_box, fill="#fffaf4", outline=COLORS["gold"], radius=24, shadow_blur=8, shadow_offset=(0, 4))
    draw.rounded_rectangle((1166, 964, 1248, 990), radius=14, fill=COLORS["gold"])
    draw.text((1166, 1002), "Best-covered fields: date, place, and incident count.", font=load_font(14, bold=True), fill=COLORS["ink"])

    output = OUT_DIR / "02_coverage_and_quality.png"
    slide.convert("RGB").save(output, quality=95)
    return output


def build_slide_3() -> Path:
    slide = make_canvas()
    draw_header(
        slide,
        "Modeling Framework",
        "The Project Uses Multiple Model Types",
        "The modeling work separates state-year incident risk from incident-level severity. Count models explain whether incidents happen more often; severity models explain how harmful incidents are once they occur.",
    )

    draw = ImageDraw.Draw(slide)
    add_panel(slide, (72, 354, 1030, 932), fill=COLORS["paper"], outline=COLORS["line"], radius=34)
    draw.text((102, 384), "Count-risk models", font=load_font(31, bold=True, serif=True), fill=COLORS["ink"])
    draw.text((102, 428), "These models operate on the state-year panel.", font=load_font(20), fill=COLORS["muted"])

    risk_cards = [
        ((102, 476, 998, 610), "Negative Binomial Baseline", "Initial count model for the zero-heavy, right-skewed incident_count outcome.", COLORS["gold"], "#fffaf4"),
        ((102, 632, 998, 766), "Two-Way Fixed Effects Risk Model", "Adds state effects, year effects, and a log enrollment offset so the comparison is incident risk rather than raw volume.", COLORS["teal"], "#f8fbfc"),
        ((102, 788, 998, 922), "Preferred Final Count Specification", "README reports NB2 alpha near zero with unstable convergence, so Poisson with state-clustered errors becomes the final count model.", COLORS["accent"], COLORS["paper_alt"]),
    ]
    for box, title, body, accent, fill in risk_cards:
        draw_card(slide, box, title=title, body=body, accent=accent, fill=fill, title_size=23, body_max=15, body_min=13, body_top=84)

    add_panel(slide, (1068, 354, 1848, 646), fill=COLORS["paper"], outline=COLORS["line"], radius=34)
    draw.text((1098, 384), "Clustered inference matters", font=load_font(31, bold=True, serif=True), fill=COLORS["ink"])
    chart = fit_image(POLICY_SE_CHART, (720, 210))
    slide.alpha_composite(chart, (1098, 414))

    draw_card(
        slide,
        (1068, 680, 1848, 922),
        title="Severity models",
        body="A second Negative Binomial model tracks total victims. An incident-level Gamma GLM models victims per incident; that is where the clearest policy-linked severity signal appears.",
        accent=COLORS["teal"],
        fill="#f8fbfc",
        title_size=26,
        body_max=17,
        body_min=14,
        body_top=88,
    )

    ribbon_box = (72, 952, 1848, 1044)
    add_panel(slide, ribbon_box, fill=COLORS["paper_alt"], outline=COLORS["line"], radius=22, shadow_blur=8, shadow_offset=(0, 5))
    draw_text_block(
        draw,
        "Interpretation: the project compares model families, but the strongest recurring story is still the broader time trend rather than a large set of stable policy effects.",
        (100, 972, 1818, 1032),
        fill=COLORS["accent"],
        max_size=19,
        min_size=16,
        bold=True,
        line_gap=4,
    )

    output = OUT_DIR / "03_models_used.png"
    slide.convert("RGB").save(output, quality=95)
    return output


def build_slide_4() -> Path:
    slide = make_canvas()
    draw_header(
        slide,
        "Project Findings",
        "Headline Findings From The Full Project",
        "Across notebooks, scripts, and frozen outputs, the strongest supported pattern is a late-year rise in national incident risk. Most single-policy effects stay smaller and less consistent than the broader time pattern.",
    )

    draw = ImageDraw.Draw(slide)
    add_panel(slide, (72, 344, 1065, 950), fill=COLORS["paper"], outline=COLORS["line"], radius=34)
    draw.text((104, 374), "National incident risk", font=load_font(31, bold=True, serif=True), fill=COLORS["ink"])
    chart = fit_image(INCIDENT_RATE_CHART, (920, 450))
    slide.alpha_composite(chart, (108, 424))
    draw_text_block(
        draw,
        "Peak observed national rate in the repo summaries: 0.71 incidents per 100,000 students in 2023.",
        (104, 892, 1034, 938),
        fill=COLORS["accent"],
        max_size=22,
        min_size=18,
        bold=True,
        line_gap=4,
    )

    draw_stat_box(
        slide,
        (1100, 344, 1328, 500),
        value="0.71",
        label="Peak national rate in the available series, reached in 2023.",
        accent=COLORS["accent"],
        fill=COLORS["paper_alt"],
    )
    draw_stat_box(
        slide,
        (1354, 344, 1582, 500),
        value="2018-2024",
        label="Window where the national rise is visually and descriptively strongest.",
        accent=COLORS["teal"],
        fill="#f8fbfc",
        value_size=28,
    )
    draw_stat_box(
        slide,
        (1608, 344, 1836, 500),
        value="0.30",
        label="Latest 2025 rate reported in the Plotly insight summary.",
        accent=COLORS["gold"],
        fill="#fffaf4",
    )

    draw_bullet_panel(
        slide,
        (1100, 536, 1836, 708),
        title="Main analytical result",
        bullets=[
            "The main incident-count and total-victim models do not show many clean single-policy effects.",
            "The time pattern is stronger and more stable than most individual policy coefficients.",
            "The repo supports a stronger trend story than a strong single-policy story.",
        ],
        accent=COLORS["teal"],
        fill=COLORS["paper"],
        bullet_max=17,
        bullet_min=14,
    )

    draw_bullet_panel(
        slide,
        (1100, 730, 1836, 888),
        title="Recent monitoring lens",
        bullets=[
            "In the repo's burden and volatility view, CA, TX, OH, NY, and IL are the leading recent monitoring states.",
            "Fast-rising recent trajectories in the Plotly insight summary are FL, NE, and WA.",
        ],
        accent=COLORS["gold"],
        fill="#fffaf4",
        bullet_max=17,
        bullet_min=14,
    )

    add_panel(slide, (1100, 910, 1836, 972), fill=COLORS["paper_alt"], outline=COLORS["line"], radius=22, shadow_blur=8, shadow_offset=(0, 4))
    draw_text_block(
        draw,
        "Severity model note: K-12 settings is the clearest positive victims-per-incident signal in the project.",
        (1126, 920, 1806, 964),
        fill=COLORS["accent"],
        max_size=19,
        min_size=16,
        bold=True,
        line_gap=3,
    )

    output = OUT_DIR / "04_project_findings_snapshot.png"
    slide.convert("RGB").save(output, quality=95)
    return output


def build_slide_5() -> Path:
    slide = make_canvas()
    draw_header(
        slide,
        "D.C. Findings",
        "D.C. Is A Rate Outlier, But Not A Severity Outlier",
        "The project treats D.C. as a meaningful special case. Rate-based outputs show sharp spikes and a large positive state effect, while the incident-level severity model does not identify D.C. as unusually severe once an incident occurs.",
    )
    draw = ImageDraw.Draw(slide)

    draw_line_chart(
        slide,
        (72, 354, 980, 938),
        title="D.C. rate pattern",
        subtitle="Selected annual rates pulled directly from the notebooks. The sharpest spike appears in 2023.",
        series=DC_RATE_SERIES,
        accent=COLORS["teal"],
    )

    draw_bullet_panel(
        slide,
        (1020, 354, 1848, 560),
        title="Risk-based evidence",
        bullets=[
            "D.C. reaches 16.36 incidents per 100,000 students in 2023, with another large spike in 2022.",
            "The risk models show a large positive D.C. state effect, around +2.86, with strong statistical significance.",
            "The SMR notebook also flags D.C. as an extreme outlier with SMR = 10.50.",
        ],
        accent=COLORS["accent"],
        fill=COLORS["paper"],
        bullet_max=18,
        bullet_min=15,
    )
    draw_card(
        slide,
        (1020, 590, 1848, 742),
        title="Severity result",
        body="In the incident-level Gamma model for victims per incident, the D.C. coefficient is negative and not statistically significant. That means D.C. stands out more on incident risk than on conditional severity.",
        accent=COLORS["teal"],
        fill="#f8fbfc",
        title_size=26,
        body_max=18,
        body_min=15,
        body_top=86,
    )
    draw_card(
        slide,
        (1020, 774, 1848, 938),
        title="Robustness and interpretation",
        body="The no-D.C. robustness check leaves the overall policy story broadly intact. The main shift is that report_stolen_lost_law moves to conventional significance, while the larger national time-trend story remains unchanged.",
        accent=COLORS["gold"],
        fill="#fffaf4",
        title_size=26,
        body_max=18,
        body_min=15,
        body_top=86,
    )

    ribbon_box = (72, 960, 1848, 1044)
    add_panel(slide, ribbon_box, fill=COLORS["paper_alt"], outline=COLORS["line"], radius=22, shadow_blur=8, shadow_offset=(0, 5))
    draw_text_block(
        draw,
        "Bottom line: D.C. is a strong rate-based outlier in the project, but it is not the driver of the full-project conclusions and it is not a severity outlier in the Gamma model.",
        (100, 974, 1818, 1034),
        fill=COLORS["accent"],
        max_size=18,
        min_size=15,
        bold=True,
        line_gap=4,
    )

    output = OUT_DIR / "05_dc_findings_snapshot.png"
    slide.convert("RGB").save(output, quality=95)
    return output


def build_pdf(images: list[Path]) -> Path:
    pdf_path = OUT_DIR / "presentation_data_section_slides.pdf"
    pil_images = [Image.open(path).convert("RGB") for path in images]
    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
    return pdf_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_slide_1(),
        build_slide_2(),
        build_slide_3(),
        build_slide_4(),
        build_slide_5(),
    ]
    Image.open(outputs[3]).convert("RGB").save(OUT_DIR / "03_key_findings.png")
    build_pdf(outputs)
    print("Created:")
    for output in outputs:
        print(output.relative_to(ROOT))
    print((OUT_DIR / "presentation_data_section_slides.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
