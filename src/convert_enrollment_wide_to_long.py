#!/usr/bin/env python3
"""Convert wide enrollment CSVs to long format.

Expected output columns:
- school_id
- state
- year
- total_students
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

YEAR_TOKEN_RE = re.compile(r"(\d{4})\s*[-–]\s*(\d{2})$")


def parse_year_token(column_name: str) -> str | None:
    match = YEAR_TOKEN_RE.search(column_name)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def school_year_end_year(year_token: str) -> int:
    start_s, end_two_s = year_token.split("-")
    start = int(start_s)
    end_two = int(end_two_s)
    end_year = (start // 100) * 100 + end_two
    if end_year < start:
        end_year += 100
    return end_year


def find_state_column(columns: list[str]) -> str:
    for col in columns:
        if "State Name" in col and "Latest available year" in col:
            return col
    raise ValueError("Could not find state column.")


def build_year_column_map(columns: list[str], marker: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in columns:
        if marker not in col:
            continue
        token = parse_year_token(col)
        if token:
            mapping[token] = col
    return mapping


def convert_wide_to_long(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    columns = [str(c) for c in df.columns]

    state_col = find_state_column(columns)

    students_by_year = build_year_column_map(
        columns, "Total Students All Grades (Excludes AE)"
    )
    school_id_by_year = build_year_column_map(
        columns, "School ID (7-digit)"
    )

    # Ensure overlapping years only and remove duplicates
    ordered_year_tokens = sorted(
        set(students_by_year.keys()) & set(school_id_by_year.keys())
    )

    if not ordered_year_tokens:
        raise ValueError(
            "No overlapping school-year columns found for total students and school ID."
        )

    frames: list[pd.DataFrame] = []

    for year_token in ordered_year_tokens:
        part = df[
            [
                school_id_by_year[year_token],
                state_col,
                students_by_year[year_token],
            ]
        ].copy()

        part.columns = ["school_id", "state", "total_students"]
        part["year"] = school_year_end_year(year_token)

        frames.append(part)

    long_df = pd.concat(frames, ignore_index=True)[
        ["school_id", "state", "year", "total_students"]
    ]

    # Clean state
    long_df["state"] = long_df["state"].astype(str).str.strip()

    # Clean school_id (remove symbols like †)
    long_df["school_id"] = (
        long_df["school_id"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    long_df["school_id"] = pd.to_numeric(
        long_df["school_id"], errors="coerce"
    )

    # Clean total_students
    long_df["total_students"] = (
        long_df["total_students"]
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    long_df["total_students"] = pd.to_numeric(
        long_df["total_students"], errors="coerce"
    )

    # Drop invalid rows
    long_df = long_df.dropna(subset=["school_id", "total_students"])

    # Enforce types
    long_df["school_id"] = long_df["school_id"].astype("int64")
    long_df["year"] = long_df["year"].astype("int64")
    long_df["total_students"] = long_df["total_students"].astype("int64")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a wide enrollment CSV file to long format."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the input wide CSV.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to the output long CSV.",
    )
    args = parser.parse_args()

    convert_wide_to_long(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()