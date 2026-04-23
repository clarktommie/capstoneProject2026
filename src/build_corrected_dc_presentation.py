#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "presentation_data_section" / "final_presentation_deck_assets" / "state_panel.csv"
OUT_DIR = ROOT / "presentation_data_section" / "corrected_dc_deck_assets"
PDF_PATH = ROOT / "presentation_data_section" / "corrected_dc_deck.pdf"
NOTES_PATH = ROOT / "presentation_data_section" / "corrected_dc_slide_notes.md"
FACT_PATH = ROOT / "presentation_data_section" / "corrected_dc_fact_check.txt"
COMPARE_PATH = ROOT / "presentation_data_section" / "corrected_dc_national_comparison.csv"

WIDTH = 1920
HEIGHT = 1080

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
SERIF_BOLD = FONT_DIR / "DejaVuSerif-Bold.ttf"

COLORS = {
    "bg": "#f6f1e8",
    "panel": "#fffdf9",
    "panel_alt": "#f4eee4",
    "ink": "#21303a",
    "muted": "#5f6f7c",
    "accent": "#9f4e31",
    "teal": "#457b83",
    "blue": "#2d5f7c",
    "gold": "#ba8a2d",
    "green": "#617a53",
    "line": "#d8cdbd",
    "grid": "#e8ddd0",
    "shadow": "#d3c6b4",
}

MODEL_RESULTS = {
    "poisson_dc": {"coef": 2.8576018223168043, "low": 2.5893380489716953, "high": 3.1258655956619132},
    "poisson_k12": {"coef": -0.42560609489330703, "low": -0.5752408654115099, "high": -0.2759713243751042},
    "gamma_dc": {"coef": 0.011029422103197559, "low": -1.118295515869158, "high": 1.140354360075553},
    "gamma_k12": {"coef": 0.47901164235497057, "low": 0.22768550520815384, "high": 0.7303377795017874},
    "nb2_alpha_note": "README reports NB2 alpha ≈ 0.03 and unstable convergence, so Poisson with clustered SE is the repo's preferred count model.",
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
    line_height = measure(draw, "Ag", font)[1]
    y = box[1]
    for line in lines:
        draw.text((box[0], y), line, font=font, fill=fill)
        y += line_height + line_gap


def make_canvas() -> Image.Image:
    base = Image.new("RGBA", (WIDTH, HEIGHT), rgba(COLORS["bg"]))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((-180, -150, 560, 420), fill=rgba("#eedecf", 190))
    draw.ellipse((1460, -120, 2100, 480), fill=rgba("#dbe9ea", 150))
    draw.rectangle((0, HEIGHT - 140, WIDTH, HEIGHT), fill=rgba("#efe7da", 150))
    return Image.alpha_composite(base, overlay)


def add_panel(
    base: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str | None = None,
    outline: str | None = None,
    radius: int = 28,
) -> None:
    fill = fill or COLORS["panel"]
    outline = outline or COLORS["line"]
    x0, y0, x1, y1 = box
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius=radius, fill=rgba(COLORS["shadow"], 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    base.alpha_composite(shadow)
    panel = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)
    pdraw.rounded_rectangle(box, radius=radius, fill=rgba(fill), outline=rgba(outline), width=2)
    base.alpha_composite(panel)


def draw_badge(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    font = load_font(22, bold=True)
    w, h = measure(draw, text, font)
    draw.rounded_rectangle((x, y, x + w + 38, y + h + 18), radius=22, fill="#f0dfd1", outline=COLORS["accent"], width=1)
    draw.text((x + 19, y + 8), text, font=font, fill=COLORS["accent"])


def draw_header(slide: Image.Image, badge: str, title: str, subtitle: str) -> None:
    draw = ImageDraw.Draw(slide)
    title_box = (72, 132, 1820, 240)
    for size in range(58, 40, -2):
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
    draw_wrapped_text(draw, subtitle, (72, 240, 1650, 310), fill=COLORS["muted"], size=27, line_gap=8)
    draw.line((72, 312, WIDTH - 72, 312), fill=COLORS["line"], width=2)


def bullet_panel(slide: Image.Image, box: tuple[int, int, int, int], bullets: list[str], accent: str) -> None:
    add_panel(slide, box)
    draw = ImageDraw.Draw(slide)
    x0, y0, x1, y1 = box
    top = y0 + 30
    row_h = (y1 - y0 - 48) // len(bullets)
    for bullet in bullets:
        draw.text((x0 + 24, top - 6), "•", font=load_font(26, bold=True), fill=accent)
        draw_wrapped_text(draw, bullet, (x0 + 56, top, x1 - 24, top + row_h - 8), fill=COLORS["ink"], size=20, line_gap=4)
        top += row_h


def stat_box(slide: Image.Image, box: tuple[int, int, int, int], value: str, label: str, accent: str, value_size: int = 34) -> None:
    add_panel(slide, box, fill=COLORS["panel_alt"], outline=accent, radius=24)
    draw = ImageDraw.Draw(slide)
    x0, y0, x1, y1 = box
    draw.text((x0 + 18, y0 + 16), value, font=load_font(value_size, bold=True), fill=accent)
    draw_wrapped_text(draw, label, (x0 + 18, y0 + 60, x1 - 18, y1 - 16), fill=COLORS["muted"], size=15, line_gap=3)


def fit_image(path: Path, target: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    image.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", target, rgba("#ffffff"))
    offset = ((target[0] - image.width) // 2, (target[1] - image.height) // 2)
    canvas.alpha_composite(image, offset)
    return canvas


def save_plot(fig, path: Path, *, width: float, height: float) -> None:
    fig.set_size_inches(width, height)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_series(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["Year"] = panel["Year"].astype(int)
    panel["incident_count"] = pd.to_numeric(panel["incident_count"], errors="coerce")
    panel["total_students"] = pd.to_numeric(panel["total_students"], errors="coerce")
    panel["risk_per_100k"] = np.where(
        panel["total_students"] > 0,
        panel["incident_count"] / panel["total_students"] * 100000.0,
        np.nan,
    )

    def aggregate(df: pd.DataFrame, name: str) -> pd.DataFrame:
        out = df.groupby("Year", as_index=False)[["incident_count", "total_students"]].sum()
        out[f"risk_{name}"] = out["incident_count"] / out["total_students"] * 100000.0
        return out.rename(
            columns={
                "incident_count": f"incident_count_{name}",
                "total_students": f"total_students_{name}",
            }
        )

    with_dc = aggregate(panel, "with_dc")
    without_dc = aggregate(panel.loc[panel["State"] != "DC"], "without_dc")
    compare = with_dc.merge(without_dc, on="Year")
    compare["abs_diff"] = compare["risk_with_dc"] - compare["risk_without_dc"]
    compare["pct_diff_vs_without"] = compare["abs_diff"] / compare["risk_without_dc"] * 100.0
    return panel, compare


def build_charts(panel: pd.DataFrame, compare: pd.DataFrame) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Path] = {}

    states_2023 = (
        panel.loc[panel["Year"] == 2023, ["State", "risk_per_100k", "incident_count", "total_students"]]
        .sort_values("risk_per_100k", ascending=False)
        .head(8)
        .sort_values("risk_per_100k", ascending=True)
    )
    fig, ax = plt.subplots()
    colors = [COLORS["accent"] if state == "DC" else COLORS["teal"] for state in states_2023["State"]]
    ax.barh(states_2023["State"], states_2023["risk_per_100k"], color=colors)
    ax.set_title("2023 State Rates: D.C. Sits Far Above Every Other State", fontsize=16, fontweight="bold")
    ax.set_xlabel("Risk per 100k students")
    ax.grid(True, axis="x", color=COLORS["grid"])
    ax.spines[["top", "right"]].set_visible(False)
    dc_row = states_2023.loc[states_2023["State"] == "DC"].iloc[0]
    top_non_dc = states_2023.loc[states_2023["State"] != "DC"].iloc[-1]
    dc_pos = states_2023.reset_index(drop=True).index[states_2023.reset_index(drop=True)["State"] == "DC"][0]
    ax.text(
        dc_row["risk_per_100k"] * 0.60,
        dc_pos,
        f"D.C. {dc_row['risk_per_100k']:.2f}\nvs next highest {top_non_dc['State']} {top_non_dc['risk_per_100k']:.2f}",
        fontsize=10,
        color="white",
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": COLORS["accent"], "ec": COLORS["accent"]},
    )
    charts["state_gap"] = OUT_DIR / "chart_state_gap_2023.png"
    save_plot(fig, charts["state_gap"], width=8.8, height=4.9)

    fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax = axes[0]
    ax.plot(compare["Year"], compare["risk_with_dc"], color=COLORS["accent"], linewidth=2.8, label="Including D.C.")
    ax.plot(compare["Year"], compare["risk_without_dc"], color=COLORS["blue"], linewidth=2.8, linestyle="--", label="Excluding D.C.")
    ax.fill_between(compare["Year"], compare["risk_with_dc"], compare["risk_without_dc"], color="#efd7cd", alpha=0.55, label="D.C. lift")
    ax.axvspan(2018, 2025.5, color="#f3dfd7", alpha=0.45)
    ax.axvline(2018, color=COLORS["gold"], linestyle=":", linewidth=1.6)
    ax.set_title("Removing D.C. Lowers The Recent National Rate", fontsize=16, fontweight="bold")
    ax.set_ylabel("Incidents per 100k students")
    ax.grid(True, axis="y", color=COLORS["grid"])
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    axes[1].bar(compare["Year"], compare["abs_diff"], color=COLORS["accent"], alpha=0.8)
    axes[1].axvline(2018, color=COLORS["gold"], linestyle=":", linewidth=1.6)
    axes[1].set_ylabel("Lift")
    axes[1].set_xlabel("Year")
    axes[1].grid(True, axis="y", color=COLORS["grid"])
    axes[1].spines[["top", "right"]].set_visible(False)
    charts["core"] = OUT_DIR / "chart_core_insight.png"
    save_plot(fig, charts["core"], width=8.8, height=6.0)

    recent = compare.loc[compare["Year"].between(2018, 2025)].copy()
    fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax = axes[0]
    ax.plot(recent["Year"], recent["risk_with_dc"], color=COLORS["accent"], linewidth=3, marker="o", label="Including D.C.")
    ax.plot(recent["Year"], recent["risk_without_dc"], color=COLORS["blue"], linewidth=3, marker="o", linestyle="--", label="Excluding D.C.")
    ax.fill_between(recent["Year"], recent["risk_with_dc"], recent["risk_without_dc"], color="#efd7cd", alpha=0.55)
    peak_with = recent.loc[recent["risk_with_dc"].idxmax()]
    peak_without = recent.loc[recent["risk_without_dc"].idxmax()]
    latest = recent.loc[recent["Year"] == 2025].iloc[0]
    ax.scatter([peak_with["Year"], peak_without["Year"], latest["Year"]], [peak_with["risk_with_dc"], peak_without["risk_without_dc"], latest["risk_with_dc"]], color=COLORS["gold"], s=42, zorder=3)
    ax.annotate(f"Peak with D.C.\n{int(peak_with['Year'])}: {peak_with['risk_with_dc']:.3f}", (peak_with["Year"], peak_with["risk_with_dc"]), xytext=(2018.1, peak_with["risk_with_dc"] + 0.03), fontsize=10, color=COLORS["accent"])
    ax.annotate(f"Peak without D.C.\n{int(peak_without['Year'])}: {peak_without['risk_without_dc']:.3f}", (peak_without["Year"], peak_without["risk_without_dc"]), xytext=(2018.1, peak_without["risk_without_dc"] - 0.10), fontsize=10, color=COLORS["blue"])
    ax.annotate(f"2025\nwith {latest['risk_with_dc']:.3f}\nwithout {latest['risk_without_dc']:.3f}", (latest["Year"], latest["risk_with_dc"]), xytext=(2023.2, 0.36), fontsize=10, color=COLORS["ink"])
    ax.set_title("2023 Still Peaks Without D.C., But At A Lower Level", fontsize=16, fontweight="bold")
    ax.set_ylabel("Incidents per 100k students")
    ax.grid(True, axis="y", color=COLORS["grid"])
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    axes[1].bar(recent["Year"], recent["pct_diff_vs_without"], color=COLORS["accent"], alpha=0.82)
    axes[1].set_ylabel("% lift")
    axes[1].set_xlabel("Year")
    axes[1].grid(True, axis="y", color=COLORS["grid"])
    axes[1].spines[["top", "right"]].set_visible(False)
    charts["comparison"] = OUT_DIR / "chart_with_vs_without_dc.png"
    save_plot(fig, charts["comparison"], width=8.8, height=6.0)

    state_change = panel.copy()
    baseline = state_change.loc[state_change["Year"] < 2018].groupby("State", as_index=False)["risk_per_100k"].mean().rename(columns={"risk_per_100k": "pre_2018_mean"})
    recent_peak = state_change.loc[state_change["Year"] >= 2018].groupby("State", as_index=False)["risk_per_100k"].max().rename(columns={"risk_per_100k": "post_2018_peak"})
    state_change = baseline.merge(recent_peak, on="State", how="inner")
    state_change["increase"] = state_change["post_2018_peak"] - state_change["pre_2018_mean"]
    top_states = state_change.sort_values("increase", ascending=False).head(5)["State"].tolist()
    if "DC" not in top_states:
        top_states = ["DC"] + [state for state in top_states if state != "DC"][:4]

    focus = panel.loc[panel["State"].isin(top_states)].copy()
    fig, ax = plt.subplots()
    palette = {
        "DC": COLORS["blue"],
        top_states[1]: "#ff7f0e",
        top_states[2]: "#2ca02c",
        top_states[3]: "#d62728",
        top_states[4]: "#9467bd",
    }
    for state in top_states:
        state_df = focus.loc[focus["State"] == state].sort_values("Year")
        ax.plot(
            state_df["Year"],
            state_df["risk_per_100k"],
            color=palette[state],
            linewidth=2.2 if state == "DC" else 1.9,
            label=state,
        )
    ax.axvline(2018, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Top States By Post-2018 Increase In Risk per 100k", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Risk per 100,000 students")
    ax.grid(True, axis="y", color=COLORS["grid"])
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        2021.2,
        15.4,
        "D.C. peak in 2023:\n16.36 per 100k",
        fontsize=10,
        color=COLORS["blue"],
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#edf3f6", "ec": COLORS["blue"]},
    )
    charts["dc_outlier"] = OUT_DIR / "chart_dc_outlier.png"
    save_plot(fig, charts["dc_outlier"], width=10.4, height=4.8)

    fig, ax = plt.subplots()
    labels = [
        "Poisson count\nD.C. state effect",
        "Poisson count\nK-12 settings law",
        "Gamma severity\nD.C. state effect",
        "Gamma severity\nK-12 settings law",
    ]
    coefs = [
        MODEL_RESULTS["poisson_dc"]["coef"],
        MODEL_RESULTS["poisson_k12"]["coef"],
        MODEL_RESULTS["gamma_dc"]["coef"],
        MODEL_RESULTS["gamma_k12"]["coef"],
    ]
    lows = [
        MODEL_RESULTS["poisson_dc"]["coef"] - MODEL_RESULTS["poisson_dc"]["low"],
        MODEL_RESULTS["poisson_k12"]["coef"] - MODEL_RESULTS["poisson_k12"]["low"],
        MODEL_RESULTS["gamma_dc"]["coef"] - MODEL_RESULTS["gamma_dc"]["low"],
        MODEL_RESULTS["gamma_k12"]["coef"] - MODEL_RESULTS["gamma_k12"]["low"],
    ]
    highs = [
        MODEL_RESULTS["poisson_dc"]["high"] - MODEL_RESULTS["poisson_dc"]["coef"],
        MODEL_RESULTS["poisson_k12"]["high"] - MODEL_RESULTS["poisson_k12"]["coef"],
        MODEL_RESULTS["gamma_dc"]["high"] - MODEL_RESULTS["gamma_dc"]["coef"],
        MODEL_RESULTS["gamma_k12"]["high"] - MODEL_RESULTS["gamma_k12"]["coef"],
    ]
    y = np.arange(len(labels))[::-1]
    colors = [COLORS["accent"], COLORS["teal"], COLORS["gold"], COLORS["green"]]
    ax.axvline(0, color="#7f8c93", linewidth=1.5)
    for idx, (coef, lo, hi, color) in enumerate(zip(coefs, lows, highs, colors, strict=True)):
        ax.errorbar(coef, y[idx], xerr=[[lo], [hi]], fmt="o", color=color, ecolor=color, elinewidth=2.5, capsize=5, markersize=8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Coefficient estimate with 95% CI")
    ax.set_title("Count And Severity Models Do Not Tell The Same D.C. Story", fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", color=COLORS["grid"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    charts["models"] = OUT_DIR / "chart_model_results.png"
    save_plot(fig, charts["models"], width=9.0, height=4.8)

    return charts


def paste_chart(slide: Image.Image, box: tuple[int, int, int, int], chart_path: Path) -> None:
    add_panel(slide, box)
    chart = fit_image(chart_path, (box[2] - box[0] - 24, box[3] - box[1] - 24))
    slide.alpha_composite(chart, (box[0] + 12, box[1] + 12))


def slide_1(compare: pd.DataFrame, charts: dict[str, Path]) -> Image.Image:
    slide = make_canvas()
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    row2025 = compare.loc[compare["Year"] == 2025].iloc[0]
    draw_header(
        slide,
        "Slide 1",
        "D.C. Is The Clear State-Level Outlier And It Raises The Recent National Rate",
        "School Shootings Capstone Project",
    )
    add_panel(slide, (72, 350, 1848, 900))
    stat_box(slide, (120, 420, 500, 560), "16.36", "D.C. risk per 100k in 2023", COLORS["accent"], value_size=42)
    stat_box(slide, (560, 420, 940, 560), "3.96", "Next-highest state risk per 100k in 2023", COLORS["teal"], value_size=42)
    stat_box(slide, (1000, 420, 1380, 560), f"{row2023['abs_diff']:.3f}", "D.C. lift to the 2023 national rate", COLORS["blue"], value_size=42)
    stat_box(slide, (1440, 420, 1820, 560), f"{row2023['pct_diff_vs_without']:.2f}%", "Percent lift to the 2023 national rate", COLORS["gold"], value_size=40)
    bullets = [
        "D.C. is not close to the rest of the state distribution.",
        f"In 2023, D.C. was 16.36 per 100k; the next-highest state was 3.96.",
        f"At the national level, removing D.C. lowers 2023 from {row2023['risk_with_dc']:.3f} to {row2023['risk_without_dc']:.3f}, and 2025 from {row2025['risk_with_dc']:.3f} to {row2025['risk_without_dc']:.3f}.",
    ]
    bullet_panel(slide, (120, 610, 760, 860), bullets, COLORS["accent"])
    paste_chart(slide, (860, 610, 1800, 860), charts["state_gap"])
    return slide


def slide_2(compare: pd.DataFrame, charts: dict[str, Path]) -> Image.Image:
    slide = make_canvas()
    post = compare.loc[compare["Year"] >= 2018]
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    draw_header(
        slide,
        "Slide 2",
        "D.C. Sits Far Above Every Other State In 2023",
        "State-level comparison from the state-year panel used in the notebook.",
    )
    bullets = [
        "D.C. is the highest-risk state in the panel by a wide margin in 2023.",
        "That outlier status comes from a small enrollment denominator combined with a burst in incidents.",
        f"This is why D.C. lifts the national series even though it is only one jurisdiction.",
    ]
    bullet_panel(slide, (72, 350, 720, 884), bullets, COLORS["teal"])
    paste_chart(slide, (760, 350, 1848, 884), charts["state_gap"])
    return slide


def slide_3(compare: pd.DataFrame, charts: dict[str, Path]) -> Image.Image:
    slide = make_canvas()
    row2025 = compare.loc[compare["Year"] == 2025].iloc[0]
    peak_with = compare.loc[compare["risk_with_dc"].idxmax()]
    peak_without = compare.loc[compare["risk_without_dc"].idxmax()]
    draw_header(
        slide,
        "Slide 3",
        "Removing D.C. Lowers The National Rate, But 2023 Still Remains The Peak Year",
        "National series including and excluding D.C., shown directly.",
    )
    bullets = [
        f"Peak with D.C.: {int(peak_with['Year'])}, {peak_with['risk_with_dc']:.3f} per 100k. Peak without D.C.: {int(peak_without['Year'])}, {peak_without['risk_without_dc']:.3f} per 100k.",
        f"2025 with D.C.: {row2025['risk_with_dc']:.3f}. 2025 without D.C.: {row2025['risk_without_dc']:.3f}.",
        f"In 2023, D.C. changes the national rate by {compare.loc[compare['Year'] == 2023, 'abs_diff'].iloc[0]:.3f} per 100k, or {compare.loc[compare['Year'] == 2023, 'pct_diff_vs_without'].iloc[0]:.2f}%.",
        f"Post-2018 average risk falls from {compare.loc[compare['Year'] >= 2018, 'risk_with_dc'].mean():.3f} to {compare.loc[compare['Year'] >= 2018, 'risk_without_dc'].mean():.3f} when D.C. is removed.",
    ]
    bullet_panel(slide, (72, 350, 720, 884), bullets, COLORS["blue"])
    paste_chart(slide, (760, 350, 1848, 884), charts["comparison"])
    return slide


def slide_4(panel: pd.DataFrame, compare: pd.DataFrame, charts: dict[str, Path]) -> Image.Image:
    slide = make_canvas()
    dc2023 = panel.loc[(panel["State"] == "DC") & (panel["Year"] == 2023)].iloc[0]
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    draw_header(
        slide,
        "Slide 4",
        "D.C. Looks Different Because A Small Denominator Magnifies Short Bursts",
        "The outlier story is real, but the national trend does not depend on it.",
    )
    bullets = [
        f"In 2023, D.C. had {int(dc2023['incident_count'])} incidents across {int(dc2023['total_students']):,} students, producing {dc2023['risk_per_100k']:.2f} per 100k.",
        f"D.C. accounts for 3.43% of 2023 incidents but only 0.16% of students in the panel, which is why its rate spikes so sharply.",
        "The corrected visual now places D.C. against the other highest-increase states over time, matching the intended notebook-style comparison.",
        f"D.C.'s 2023 rate is {dc2023['risk_per_100k'] / row2023['risk_without_dc']:.1f}x the national rate excluding D.C., so it is a genuine outlier even though it does not create the national peak year.",
    ]
    bullet_panel(slide, (72, 350, 720, 884), bullets, COLORS["gold"])
    paste_chart(slide, (760, 350, 1848, 884), charts["dc_outlier"])
    return slide


def slide_5(charts: dict[str, Path]) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Slide 5",
        "Repo Model Outputs Show A D.C. Count Effect, Not A D.C. Severity Effect",
        "These are the notebook-backed model estimates, reconciled with the corrected D.C. sensitivity check.",
    )
    bullets = [
        MODEL_RESULTS["nb2_alpha_note"],
        f"Preferred Poisson count model: `C(State)[T.DC] = {MODEL_RESULTS['poisson_dc']['coef']:.3f}` with 95% CI [{MODEL_RESULTS['poisson_dc']['low']:.3f}, {MODEL_RESULTS['poisson_dc']['high']:.3f}].",
        f"Gamma severity model: `C(State)[T.DC] = {MODEL_RESULTS['gamma_dc']['coef']:.3f}` with 95% CI [{MODEL_RESULTS['gamma_dc']['low']:.3f}, {MODEL_RESULTS['gamma_dc']['high']:.3f}], which crosses zero.",
        f"Gamma severity model: `k12_settings_law = {MODEL_RESULTS['gamma_k12']['coef']:.3f}` with 95% CI [{MODEL_RESULTS['gamma_k12']['low']:.3f}, {MODEL_RESULTS['gamma_k12']['high']:.3f}].",
    ]
    bullet_panel(slide, (72, 350, 760, 900), bullets, COLORS["accent"])
    paste_chart(slide, (810, 350, 1848, 900), charts["models"])
    return slide


def slide_6(compare: pd.DataFrame) -> Image.Image:
    slide = make_canvas()
    draw_header(
        slide,
        "Slide 6",
        "What The Corrected Reading Supports And What It Does Not",
        "The data trend, the D.C. comparison, and the model outputs point to a narrower and more defensible story.",
    )
    add_panel(slide, (72, 360, 620, 890), fill="#f9f3ec", outline=COLORS["accent"])
    add_panel(slide, (686, 360, 1234, 890), fill="#eef5f6", outline=COLORS["teal"])
    add_panel(slide, (1300, 360, 1848, 890), fill="#f6f1e8", outline=COLORS["gold"])
    draw = ImageDraw.Draw(slide)
    draw.text((104, 392), "Supported", font=load_font(34, serif=True), fill=COLORS["accent"])
    draw_wrapped_text(
        draw,
        "The national rise after 2018 remains visible and peaks in 2023 even when D.C. is excluded.",
        (104, 452, 588, 620),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw_wrapped_text(
        draw,
        "D.C. is a real rate outlier in the count data because its student base is small relative to short bursts in incidents.",
        (104, 620, 588, 820),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw.text((718, 392), "Not Supported", font=load_font(34, serif=True), fill=COLORS["teal"])
    draw_wrapped_text(
        draw,
        "It is not supported to say D.C. alone creates the national post-2018 trend.",
        (718, 452, 1202, 620),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw_wrapped_text(
        draw,
        "It is not supported to treat D.C. as a severity outlier after conditioning on incidents.",
        (718, 620, 1202, 820),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw.text((1332, 392), "Interpretation", font=load_font(34, serif=True), fill=COLORS["gold"])
    draw_wrapped_text(
        draw,
        "Trend first: the strongest empirical pattern is the national rate increase.",
        (1332, 452, 1816, 580),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw_wrapped_text(
        draw,
        "D.C. second: it sharpens the level but does not change the peak year or remove the trend when excluded.",
        (1332, 600, 1816, 760),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    draw_wrapped_text(
        draw,
        "Model third: count and severity models should be discussed separately.",
        (1332, 780, 1816, 860),
        fill=COLORS["ink"],
        size=24,
        line_gap=6,
    )
    return slide


def slide_7(compare: pd.DataFrame) -> Image.Image:
    slide = make_canvas()
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    draw_header(
        slide,
        "Slide 7",
        "Final Corrected Takeaway: D.C. Amplifies The National Pattern But Does Not Drive It",
        "The corrected conclusion is narrower, data-checked, and presentation-ready.",
    )
    add_panel(slide, (120, 380, 1798, 860), fill="#fffaf4", outline=COLORS["accent"], radius=36)
    draw = ImageDraw.Draw(slide)
    draw_wrapped_text(
        draw,
        f"The state-year panel shows a national risk peak in 2023 with D.C. ({row2023['risk_with_dc']:.3f}) and without D.C. ({row2023['risk_without_dc']:.3f}). D.C. remains an outlier because 12 incidents over 73,364 students produce a very high local rate, but removing D.C. changes the 2023 national rate by only {row2023['abs_diff']:.3f} per 100k ({row2023['pct_diff_vs_without']:.2f}%). The count model reinforces D.C. as a rate outlier; the severity model does not. The strongest supported claim is the broader post-2018 rise in national risk, not a D.C.-only story.",
        (176, 456, 1740, 760),
        fill=COLORS["ink"],
        size=31,
        line_gap=10,
        serif=True,
    )
    stat_box(slide, (240, 790, 620, 920), f"{row2023['risk_without_dc']:.3f}", "2023 national rate excluding D.C.", COLORS["blue"], value_size=36)
    stat_box(slide, (770, 790, 1150, 920), f"{row2023['abs_diff']:.3f}", "D.C. lift to the 2023 national rate", COLORS["accent"], value_size=36)
    stat_box(slide, (1300, 790, 1680, 920), f"{row2023['pct_diff_vs_without']:.2f}%", "Percent effect of D.C. on the 2023 national rate", COLORS["gold"], value_size=34)
    return slide


def write_notes(compare: pd.DataFrame) -> None:
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    peak_with = compare.loc[compare["risk_with_dc"].idxmax()]
    peak_without = compare.loc[compare["risk_without_dc"].idxmax()]
    row2025 = compare.loc[compare["Year"] == 2025].iloc[0]
    post = compare.loc[compare["Year"] >= 2018]
    compare_recent = compare.loc[compare["Year"].between(2018, 2025), ["Year", "risk_with_dc", "risk_without_dc", "abs_diff", "pct_diff_vs_without"]]
    lines = [
        "# Corrected D.C. Sensitivity Deck",
        "",
        "## Fact Check",
        f"- 2023 national risk with D.C.: `{row2023['risk_with_dc']:.6f}`",
        f"- 2023 national risk without D.C.: `{row2023['risk_without_dc']:.6f}`",
        f"- 2023 absolute difference: `{row2023['abs_diff']:.6f}`",
        f"- 2023 percent difference: `{row2023['pct_diff_vs_without']:.6f}%`",
        f"- Peak year with D.C.: `{int(peak_with['Year'])}`",
        f"- Peak year without D.C.: `{int(peak_without['Year'])}`",
        f"- Post-2018 average with D.C.: `{post['risk_with_dc'].mean():.6f}`",
        f"- Post-2018 average without D.C.: `{post['risk_without_dc'].mean():.6f}`",
        "",
        "## 2018-2025 Comparison",
        "",
        "| Year | With D.C. | Without D.C. | Absolute Diff | Percent Diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in compare_recent.itertuples(index=False):
        lines.append(
            f"| {int(row.Year)} | {row.risk_with_dc:.6f} | {row.risk_without_dc:.6f} | {row.abs_diff:.6f} | {row.pct_diff_vs_without:.6f}% |"
        )
    lines.extend(
        [
            "",
            "## Slide Summary",
            "",
            "1. Title: corrected main finding is that the national rise survives excluding D.C.",
            "2. Corrected Core Insight: direct proof chart comparing the two national series.",
            f"3. With vs Without D.C.: peak with D.C. `{peak_with['risk_with_dc']:.3f}` in `{int(peak_with['Year'])}`; peak without D.C. `{peak_without['risk_without_dc']:.3f}` in `{int(peak_without['Year'])}`; 2025 values `{row2025['risk_with_dc']:.3f}` and `{row2025['risk_without_dc']:.3f}`.",
            "4. Why D.C. Looks Different: denominator and burstiness, using actual incidents and students.",
            "5. Model Results: Poisson count and Gamma severity outputs with D.C. and K-12 terms.",
            "6. Interpretation: what is supported versus not supported.",
            "7. Conclusion: D.C. amplifies but does not create the national trend.",
        ]
    )
    NOTES_PATH.write_text("\n".join(lines))


def write_fact_check(compare: pd.DataFrame) -> None:
    row2023 = compare.loc[compare["Year"] == 2023].iloc[0]
    peak_with = compare.loc[compare["risk_with_dc"].idxmax()]
    peak_without = compare.loc[compare["risk_without_dc"].idxmax()]
    post = compare.loc[compare["Year"] >= 2018]
    lines = [
        "FACT CHECK",
        f"2023 national risk with D.C.: {row2023['risk_with_dc']:.6f}",
        f"2023 national risk without D.C.: {row2023['risk_without_dc']:.6f}",
        f"absolute difference: {row2023['abs_diff']:.6f}",
        f"percent difference: {row2023['pct_diff_vs_without']:.6f}%",
        f"peak year with D.C.: {int(peak_with['Year'])}",
        f"peak year without D.C.: {int(peak_without['Year'])}",
        f"post-2018 average with D.C.: {post['risk_with_dc'].mean():.6f}",
        f"post-2018 average without D.C.: {post['risk_without_dc'].mean():.6f}",
    ]
    FACT_PATH.write_text("\n".join(lines))


def save_deck(slides: list[Image.Image]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_paths = []
    for idx, slide in enumerate(slides, start=1):
        path = OUT_DIR / f"slide_{idx:02d}.png"
        slide.convert("RGB").save(path, quality=95)
        slide_paths.append(path)
    rgb_slides = [Image.open(path).convert("RGB") for path in slide_paths]
    rgb_slides[0].save(PDF_PATH, save_all=True, append_images=rgb_slides[1:], resolution=150.0)


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    panel, compare = compute_series(panel)
    compare.to_csv(COMPARE_PATH, index=False)
    write_fact_check(compare)
    write_notes(compare)
    charts = build_charts(panel, compare)
    slides = [
        slide_1(compare, charts),
        slide_2(compare, charts),
        slide_3(compare, charts),
        slide_4(panel, compare, charts),
        slide_5(charts),
        slide_6(compare),
        slide_7(compare),
    ]
    save_deck(slides)


if __name__ == "__main__":
    main()
