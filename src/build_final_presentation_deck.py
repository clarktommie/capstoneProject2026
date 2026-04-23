#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from supabase import create_client


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation_data_section" / "final_presentation_deck_assets"
PDF_PATH = ROOT / "presentation_data_section" / "final_presentation_deck.pdf"

WIDTH = 1920
HEIGHT = 1080

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
SERIF_BOLD = FONT_DIR / "DejaVuSerif-Bold.ttf"

COLORS = {
    "bg": "#f6f1e8",
    "panel": "#fffdf9",
    "panel_alt": "#f7f2ea",
    "ink": "#23313d",
    "muted": "#5e7180",
    "accent": "#9b4f34",
    "teal": "#4f7c88",
    "gold": "#c8942c",
    "blue": "#295c7a",
    "red": "#b04f3d",
    "line": "#d7ccbd",
    "shadow": "#cfc4b4",
    "grid": "#e7ddd0",
}

COUNT_MODEL_DC = {"coef": 2.8558, "low": 1.716, "high": 3.996}
SEVERITY_MODEL_DC = {"coef": -0.3264, "low": -1.338, "high": 0.686}
SEVERITY_K12 = {"coef": 0.6781, "low": 0.302, "high": 1.054}
NB2_ALPHA = 0.03

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


def load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        return ImageFont.truetype(str(SERIF_BOLD), size)
    return ImageFont.truetype(str(SANS_BOLD if bold else SANS), size)


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
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


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    size: int,
    bold: bool = False,
    serif: bool = False,
    line_gap: int = 6,
) -> None:
    font = load_font(size, bold=bold, serif=serif)
    lines = wrap_text(draw, text, font, box[2] - box[0])
    line_h = measure(draw, "Ag", font)[1]
    y = box[1]
    for line in lines:
        draw.text((box[0], y), line, font=font, fill=fill)
        y += line_h + line_gap


def make_canvas() -> Image.Image:
    base = Image.new("RGBA", (WIDTH, HEIGHT), rgba(COLORS["bg"]))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((-150, -140, 520, 460), fill=rgba("#eedfd0", 200))
    draw.ellipse((1450, -120, 2080, 520), fill=rgba("#dde9eb", 165))
    draw.rectangle((0, HEIGHT - 120, WIDTH, HEIGHT), fill=rgba("#f1eadd", 170))
    return Image.alpha_composite(base, overlay)


def add_panel(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str = None,
    outline: str = None,
    radius: int = 28,
) -> None:
    fill = fill or COLORS["panel"]
    outline = outline or COLORS["line"]
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    x0, y0, x1, y1 = box
    sdraw.rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius=radius, fill=rgba(COLORS["shadow"], 120))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    base.alpha_composite(shadow)
    panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    pdraw.rounded_rectangle(box, radius=radius, fill=rgba(fill), outline=rgba(outline), width=2)
    base.alpha_composite(panel)


def draw_badge(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    font = load_font(24, bold=True)
    w, h = measure(draw, text, font)
    draw.rounded_rectangle((x, y, x + w + 42, y + h + 18), radius=22, fill="#f0dfd1", outline=COLORS["accent"], width=1)
    draw.text((x + 20, y + 7), text, font=font, fill=COLORS["accent"])


def draw_header(slide: Image.Image, badge: str, title: str, subtitle: str) -> None:
    draw = ImageDraw.Draw(slide)
    draw_badge(draw, badge, 70, 54)
    title_box = (72, 138, 1848, 228)
    for size in range(62, 38, -2):
        font = load_font(size, serif=True)
        lines = wrap_text(draw, title, font, title_box[2] - title_box[0])
        line_h = measure(draw, "Ag", font)[1]
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * 8
        if len(lines) <= 2 and total_h <= (title_box[3] - title_box[1]):
            y = title_box[1]
            for line in lines:
                draw.text((title_box[0], y), line, font=font, fill=COLORS["accent"])
                y += line_h + 8
            break
    draw_wrapped_text(draw, subtitle, (72, 244, 1610, 320), fill=COLORS["muted"], size=27, line_gap=8)
    draw.line((72, 312, WIDTH - 72, 312), fill=COLORS["line"], width=2)


def card(slide: Image.Image, box: tuple[int, int, int, int], title: str, body: str, accent: str) -> None:
    add_panel(slide, box, fill=COLORS["panel"], outline=accent)
    draw = ImageDraw.Draw(slide)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 18, y0 + 18, x0 + 94, y0 + 42), radius=12, fill=accent)
    draw.text((x0 + 24, y0 + 58), title, font=load_font(28, bold=True), fill=COLORS["ink"])
    draw_wrapped_text(draw, body, (x0 + 24, y0 + 102, x1 - 24, y1 - 24), fill=COLORS["muted"], size=22)


def bullet_panel(slide: Image.Image, box: tuple[int, int, int, int], title: str, bullets: list[str], accent: str) -> None:
    add_panel(slide, box, fill=COLORS["panel"])
    draw = ImageDraw.Draw(slide)
    x0, y0, x1, y1 = box
    draw.text((x0 + 24, y0 + 22), title, font=load_font(30, serif=True), fill=COLORS["ink"])
    top = y0 + 80
    row_h = max(36, (y1 - top - 18) // len(bullets))
    bullet_font = load_font(22, bold=True)
    for bullet in bullets:
        draw.text((x0 + 24, top - 2), "•", font=bullet_font, fill=accent)
        draw_wrapped_text(draw, bullet, (x0 + 58, top, x1 - 24, top + row_h - 6), fill=COLORS["muted"], size=18, line_gap=3)
        top += row_h


def stat_box(slide: Image.Image, box: tuple[int, int, int, int], value: str, label: str, accent: str, value_size: int = 33) -> None:
    add_panel(slide, box, fill=COLORS["panel_alt"], outline=accent, radius=24)
    draw = ImageDraw.Draw(slide)
    x0, y0, x1, y1 = box
    draw.text((x0 + 18, y0 + 16), value, font=load_font(value_size, bold=True), fill=accent)
    draw_wrapped_text(draw, label, (x0 + 18, y0 + 62, x1 - 18, y1 - 14), fill=COLORS["muted"], size=15, line_gap=3)


def save_plot(fig, path: Path, *, width: float = 8.5, height: float = 4.6) -> None:
    fig.set_size_inches(width, height)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fit_image(path: Path, target: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    image.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", target, rgba("#ffffff"))
    offset = ((target[0] - image.width) // 2, (target[1] - image.height) // 2)
    canvas.alpha_composite(image, offset)
    return canvas


def fetch_all_rows(client, table_name: str, page_size: int = 1000) -> pd.DataFrame:
    start = 0
    rows: list[dict] = []
    while True:
        chunk = client.table(table_name).select("*").range(start, start + page_size - 1).execute().data or []
        if not chunk:
            break
        rows.extend(chunk)
        start += page_size
    return pd.DataFrame(rows)


def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    load_dotenv(ROOT / ".env", override=True)
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    national = fetch_all_rows(client, "incident_rate_per_100k")
    national = national.rename(columns={"incidents_per_100k_students": "risk_per_100k"})
    national["year"] = pd.to_numeric(national["year"], errors="coerce").astype(int)
    national["risk_per_100k"] = pd.to_numeric(national["risk_per_100k"], errors="coerce")
    national["incident_count"] = pd.to_numeric(national["incident_count"], errors="coerce")
    national["national_enrollment"] = pd.to_numeric(national["national_enrollment"], errors="coerce")
    national = national.sort_values("year").reset_index(drop=True)

    incident = fetch_all_rows(client, "incident")
    enrollment = fetch_all_rows(client, "enrollment_state_year_mat")

    incident["State"] = incident["State"].astype(str).str.upper().str.strip()
    incident["Year"] = pd.to_numeric(incident["Year"], errors="coerce")
    incident = incident.dropna(subset=["State", "Year"])
    incident["Year"] = incident["Year"].astype(int)
    incident_counts = (
        incident.groupby(["State", "Year"], as_index=False).size().rename(columns={"size": "incident_count"})
    )

    enrollment["state"] = enrollment["state"].astype(str).str.upper().str.strip()
    enrollment["State"] = enrollment["state"].map(STATE_NAME_TO_ABBREV)
    enrollment["Year"] = pd.to_numeric(enrollment["year"], errors="coerce")
    enrollment["total_students"] = pd.to_numeric(enrollment["total_students"], errors="coerce")
    enrollment = (
        enrollment.dropna(subset=["State", "Year", "total_students"])
        .sort_values("total_students", ascending=False)
        .drop_duplicates(["State", "Year"])
        .loc[:, ["State", "Year", "total_students"]]
        .copy()
    )
    enrollment["Year"] = enrollment["Year"].astype(int)

    panel = enrollment.merge(incident_counts, on=["State", "Year"], how="left")
    panel["incident_count"] = panel["incident_count"].fillna(0.0)
    panel["risk_per_100k"] = np.where(
        panel["total_students"] > 0,
        panel["incident_count"] / panel["total_students"] * 100000.0,
        np.nan,
    )
    panel = panel.sort_values(["Year", "State"]).reset_index(drop=True)

    national.to_csv(OUT_DIR / "national_series.csv", index=False)
    panel.to_csv(OUT_DIR / "state_panel.csv", index=False)
    return national, panel


def build_charts(national: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    peak = national.loc[national["risk_per_100k"].idxmax()]
    latest = national.iloc[-1]

    fig, ax = plt.subplots()
    ax.plot(national["year"], national["risk_per_100k"], color=COLORS["blue"], linewidth=3)
    ax.axvspan(2018, 2025.5, color="#f3dfd7", alpha=0.75)
    ax.axvline(2018, color=COLORS["red"], linestyle="--", linewidth=1.8)
    ax.scatter([peak["year"], latest["year"]], [peak["risk_per_100k"], latest["risk_per_100k"]], color=COLORS["accent"], s=42, zorder=3)
    ax.annotate("2023 peak\n0.71", (peak["year"], peak["risk_per_100k"]), xytext=(peak["year"] - 5, peak["risk_per_100k"] - 0.16), arrowprops=dict(arrowstyle="-", color=COLORS["accent"]), color=COLORS["accent"], fontsize=10, fontweight="bold")
    ax.annotate("2025\n0.30", (latest["year"], latest["risk_per_100k"]), xytext=(latest["year"] - 4, latest["risk_per_100k"] + 0.09), arrowprops=dict(arrowstyle="-", color=COLORS["accent"]), color=COLORS["accent"], fontsize=10, fontweight="bold")
    ax.set_title("National Risk Enters A Higher Regime After 2018", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents per 100k students")
    ax.grid(True, axis="y", color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    charts["core_insight"] = OUT_DIR / "chart_core_insight.png"
    save_plot(fig, charts["core_insight"])

    roll = national["risk_per_100k"].rolling(3, min_periods=1).mean()
    fig, ax = plt.subplots()
    ax.fill_between(national["year"], national["risk_per_100k"], color="#dbe9ee", alpha=0.9)
    ax.plot(national["year"], national["risk_per_100k"], color=COLORS["teal"], linewidth=2.4, label="Annual risk")
    ax.plot(national["year"], roll, color=COLORS["accent"], linewidth=2.6, label="3-year average")
    ax.set_title("The National Rise Is Sustained, Not A Single-Year Blip", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents per 100k students")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    charts["national_pattern"] = OUT_DIR / "chart_national_pattern.png"
    save_plot(fig, charts["national_pattern"])

    dc = panel.loc[panel["State"] == "DC", ["Year", "incident_count", "total_students", "risk_per_100k"]].copy()
    dc = dc[dc["Year"].between(2018, 2025)].sort_values("Year")
    others = (
        panel.loc[(panel["State"] != "DC") & panel["Year"].between(2018, 2025)]
        .groupby("Year")["risk_per_100k"]
        .median()
        .reset_index(name="median_other_states")
    )
    top_2023 = (
        panel.loc[panel["Year"] == 2023, ["State", "risk_per_100k"]]
        .sort_values("risk_per_100k", ascending=False)
        .head(8)
        .sort_values("risk_per_100k", ascending=True)
    )

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(dc["Year"], dc["risk_per_100k"], color=COLORS["accent"], linewidth=3, marker="o", label="D.C.")
    axes[0].plot(others["Year"], others["median_other_states"], color=COLORS["blue"], linewidth=2.5, linestyle="--", label="Median other states")
    axes[0].set_title("D.C. vs Median Of Other States", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Risk per 100k")
    axes[0].grid(True, axis="y", color=COLORS["grid"])
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = [COLORS["accent"] if state == "DC" else COLORS["teal"] for state in top_2023["State"]]
    axes[1].barh(top_2023["State"], top_2023["risk_per_100k"], color=colors)
    axes[1].set_title("2023 Rates Show Why D.C. Separates", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Risk per 100k")
    axes[1].grid(True, axis="x", color=COLORS["grid"])
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    charts["dc_vs_others"] = OUT_DIR / "chart_dc_vs_others.png"
    save_plot(fig, charts["dc_vs_others"], width=11.0, height=4.8)

    fig, ax = plt.subplots()
    names = ["Count: D.C.\nPoisson FE", "Severity: D.C.\nGamma", "Severity:\nk12_settings_law"]
    coefs = [COUNT_MODEL_DC["coef"], SEVERITY_MODEL_DC["coef"], SEVERITY_K12["coef"]]
    lows = [COUNT_MODEL_DC["coef"] - COUNT_MODEL_DC["low"], SEVERITY_MODEL_DC["coef"] - SEVERITY_MODEL_DC["low"], SEVERITY_K12["coef"] - SEVERITY_K12["low"]]
    highs = [COUNT_MODEL_DC["high"] - COUNT_MODEL_DC["coef"], SEVERITY_MODEL_DC["high"] - SEVERITY_MODEL_DC["coef"], SEVERITY_K12["high"] - SEVERITY_K12["coef"]]
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color="#7f8c93", linewidth=1.5)
    ax.errorbar(coefs, y, xerr=[lows, highs], fmt="o", color=COLORS["accent"], ecolor=COLORS["accent"], elinewidth=2.5, capsize=5, markersize=8)
    ax.set_yticks(y, names)
    ax.set_xlabel("Coefficient estimate with 95% CI")
    ax.set_title("Count And Severity Models Tell Different Stories", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", color=COLORS["grid"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    charts["model_results"] = OUT_DIR / "chart_model_results.png"
    save_plot(fig, charts["model_results"], width=8.8, height=4.6)

    return charts


def slide_title(national: pd.DataFrame, chart_path: Path) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Title",
        "Rising National Risk Is The Main Finding",
        "This deck uses only the project’s saved data, model outputs, and existing analytic conclusions. The strongest supported result is a sharp post-2018 increase in national incident risk.",
    )
    add_panel(slide, (72, 360, 1200, 940))
    slide.alpha_composite(fit_image(chart_path, (1080, 520)), (96, 392))
    stat_box(slide, (1240, 360, 1848, 540), "0.71", "Peak national rate in the project tables, reached in 2023.", COLORS["accent"], value_size=44)
    card(
        slide,
        (1240, 578, 1848, 940),
        "Main finding",
        "National incident risk rises sharply after 2018 and remains far above most earlier years in the available 1987-2025 series.",
        COLORS["teal"],
    )
    return slide


def slide_core_insight(chart_path: Path) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Core Insight",
        "National Risk Enters A Higher Post-2018 Regime",
        "The clearest project-wide result is not a single coefficient. It is the visible shift in national risk after 2018, peaking in 2023 and staying elevated in later years.",
    )
    add_panel(slide, (72, 360, 1848, 948))
    slide.alpha_composite(fit_image(chart_path, (1700, 500)), (110, 392))
    draw = ImageDraw.Draw(slide)
    draw_wrapped_text(draw, "The project tables report 0.709825 in 2023 and 0.297758 in 2025.", (118, 902, 840, 940), fill=COLORS["accent"], size=24, bold=True)
    return slide


def slide_national_pattern(chart_path: Path, national: pd.DataFrame) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "National Pattern",
        "The Increase Is Sustained, Not Just A One-Year Spike",
        "The project normalizes incidents by enrollment and explicitly repairs denominator comparability issues. That makes the late-year rise a stronger signal than raw incident counts alone.",
    )
    add_panel(slide, (72, 354, 1180, 936))
    slide.alpha_composite(fit_image(chart_path, (1050, 500)), (104, 392))
    stat_box(slide, (1220, 354, 1435, 520), "1987-2025", "Core analytic window used for the cleaned national series.", COLORS["teal"], value_size=28)
    stat_box(slide, (1460, 354, 1680, 520), "-0.35%", "Latest 2025 enrollment YoY change in the Plotly insight summary.", COLORS["gold"], value_size=28)
    stat_box(slide, (1705, 354, 1848, 520), "39", "Annual observations in the national series.", COLORS["accent"], value_size=28)
    bullet_panel(
        slide,
        (1220, 556, 1848, 936),
        "What this proves",
        [
            "The national rise survives population normalization because the rate is built on total students, not raw counts.",
            "The project explicitly flags and repairs the 2019 enrollment comparability break before interpreting trend movement.",
            "The strongest story in the repo is the broader time pattern, not a broad set of stable policy effects.",
        ],
        COLORS["teal"],
    )
    return slide


def slide_dc_vs_others(chart_path: Path, panel: pd.DataFrame) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "D.C. vs Other States",
        "D.C. Looks Extreme Because Short Bursts Hit A Much Smaller Student Base",
        "D.C. is a genuine rate outlier in the project data, but the reason is visible once incident counts and student exposure are considered together.",
    )
    add_panel(slide, (72, 354, 1240, 936))
    slide.alpha_composite(fit_image(chart_path, (1110, 500)), (102, 392))
    dc_2023 = panel.loc[(panel["State"] == "DC") & (panel["Year"] == 2023)].iloc[0]
    stat_box(slide, (1280, 354, 1488, 520), f"{dc_2023['incident_count']:.0f}", "D.C. incidents in 2023.", COLORS["accent"], value_size=40)
    stat_box(slide, (1512, 354, 1730, 520), f"{int(dc_2023['total_students']):,}", "D.C. student base in 2023.", COLORS["teal"], value_size=32)
    stat_box(slide, (1752, 354, 1848, 520), f"{dc_2023['risk_per_100k']:.2f}", "D.C. risk in 2023.", COLORS["gold"], value_size=30)
    bullet_panel(
        slide,
        (1280, 556, 1848, 936),
        "Why D.C. separates",
        [
            "The notebooks report D.C. at 11.175213 in 2022 and 16.356796 in 2023 before falling to 4.025333 in 2025.",
            "The 2023 spike is 12 incidents over 73,364 students, so the denominator is small enough to magnify a short burst into a very high rate.",
            "That makes D.C. a special case in rate space without overturning the broader national trend.",
        ],
        COLORS["accent"],
    )
    return slide


def slide_model_results(chart_path: Path) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Model Results",
        "The Final Count Model Flags D.C., But The Severity Model Does Not",
        "The project’s final count specification is Poisson with state and year fixed effects, a log enrollment offset, and state-clustered standard errors. The severity model is a Gamma GLM on victims per incident.",
    )
    add_panel(slide, (72, 354, 1100, 936))
    slide.alpha_composite(fit_image(chart_path, (980, 500)), (98, 392))
    bullet_panel(
        slide,
        (1140, 354, 1848, 640),
        "Model facts from the repo",
        [
            f"README reports NB2 alpha near zero at about {NB2_ALPHA:.2f}, statistically indistinguishable from zero, with unstable convergence.",
            "That is why the final count model is Poisson with clustered state-level standard errors rather than Negative Binomial.",
            "The count model reports a large positive D.C. effect, while the severity model’s D.C. term crosses zero.",
        ],
        COLORS["teal"],
    )
    stat_box(slide, (1140, 682, 1364, 844), "2.8558", "Count model: D.C. coefficient with 95% CI [1.716, 3.996].", COLORS["accent"], value_size=34)
    stat_box(slide, (1384, 682, 1608, 844), "-0.3264", "Severity model: D.C. coefficient with 95% CI [-1.338, 0.686].", COLORS["blue"], value_size=30)
    stat_box(slide, (1628, 682, 1848, 844), "0.6781", "Severity model: k12_settings_law with 95% CI [0.302, 1.054].", COLORS["gold"], value_size=32)
    add_panel(slide, (1140, 876, 1848, 936), fill=COLORS["panel_alt"])
    draw = ImageDraw.Draw(slide)
    draw_wrapped_text(draw, "Interpretation: D.C. is a strong rate effect in the count model, but not a conditional-severity outlier once an incident occurs.", (1164, 892, 1820, 930), fill=COLORS["accent"], size=18, bold=True)
    return slide


def slide_interpretation() -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Interpretation",
        "The Repo Supports A Trend Story First, A D.C. Outlier Story Second, And A Limited Policy Story Third",
        "The cleaned panel, the final model choice, and the D.C. special-case results line up. The strongest result is rising national risk over time rather than a broad causal policy story.",
    )
    card(slide, (72, 370, 596, 902), "Data", "The project builds a cleaned 1987-2025 panel, preserves zero-incident years, and normalizes incidents by student exposure.", COLORS["teal"])
    card(slide, (698, 370, 1222, 902), "Models", "The final count model is Poisson FE with clustered standard errors; the severity model is a Gamma GLM and yields a different D.C. conclusion.", COLORS["accent"])
    card(slide, (1324, 370, 1848, 902), "D.C.", "D.C. is a rate outlier because short bursts operate over a smaller denominator, but it is not a significant severity outlier in the Gamma model.", COLORS["gold"])
    add_panel(slide, (72, 938, 1848, 1018), fill=COLORS["panel_alt"])
    draw = ImageDraw.Draw(slide)
    draw.text((108, 962), "Bottom line: the project’s strongest evidence points to a rising national risk regime, with D.C. as an important but non-decisive special case.", font=load_font(24, bold=True), fill=COLORS["accent"])
    return slide


def slide_conclusion() -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Conclusion",
        "Rising National Risk Is The Cleanest Final Takeaway",
        "The full project is most persuasive when read as a cleaned, enrollment-normalized national risk story with D.C. treated as a distinct but not decisive outlier.",
    )
    stat_box(slide, (72, 380, 560, 620), "0.71 in 2023", "Peak national risk reported in the project tables.", COLORS["accent"], value_size=42)
    stat_box(slide, (596, 380, 1084, 620), "D.C. rate outlier", "Large count-model state effect, but no significant severity effect in the Gamma model.", COLORS["teal"], value_size=34)
    stat_box(slide, (1120, 380, 1608, 620), "Policy signals are limited", "The broader time pattern is stronger and more stable than most individual policy coefficients.", COLORS["gold"], value_size=30)
    card(slide, (1640, 380, 1848, 620), "Final read", "Monitor the national trend first.", COLORS["blue"])
    bullet_panel(
        slide,
        (72, 674, 1848, 952),
        "Closing takeaway",
        [
            "National incident risk rises sharply after 2018 and reaches its peak in 2023.",
            "D.C. appears different because a small student base magnifies short bursts into extreme per-capita rates.",
            "The final count model supports a strong D.C. risk effect, but the severity model does not.",
            "The strongest evidence in the repo supports monitoring the time trend more than claiming broad uniform policy effects.",
        ],
        COLORS["accent"],
    )
    return slide


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    national, panel = load_project_data()
    charts = build_charts(national, panel)

    slides = [
        slide_title(national, charts["core_insight"]),
        slide_core_insight(charts["core_insight"]),
        slide_national_pattern(charts["national_pattern"], national),
        slide_dc_vs_others(charts["dc_vs_others"], panel),
        slide_model_results(charts["model_results"]),
        slide_interpretation(),
        slide_conclusion(),
    ]

    slide_paths: list[Path] = []
    for idx, slide in enumerate(slides, start=1):
        path = OUT_DIR / f"slide_{idx:02d}.png"
        slide.convert("RGB").save(path, quality=95)
        slide_paths.append(path)

    pdf_images = [Image.open(path).convert("RGB") for path in slide_paths]
    pdf_images[0].save(PDF_PATH, save_all=True, append_images=pdf_images[1:])
    print(PDF_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
