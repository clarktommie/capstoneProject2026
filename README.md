# School Shootings Data Project

This repository contains data work for analyzing K-12 school shooting trends and related policy context.

## Current Focus

- Prepared enrollment data for downstream analysis and database loading.
- Converted wide, year-based enrollment columns into a normalized long format.
- Documented indexing for faster year-based queries.

## Enrollment Conversion Script

Use the reusable script below to convert enrollment CSV files for any year range:

```bash
python3 src/convert_enrollment_wide_to_long.py <input_csv> <output_csv>
```

Example:

```bash
python3 src/convert_enrollment_wide_to_long.py data/1987_1995.csv data/1987_1995_long.csv
```

Output columns:

- `school_id`
- `state`
- `year`
- `total_students`

## Database Note

For large enrollment tables, create a year index:

```sql
CREATE INDEX idx_enrollment_year
ON enrollment_all_schools (year);
```

## Environment

- Python `>=3.12`
- Main dependencies are in `pyproject.toml` (pandas, plotly, matplotlib, supabase client, dotenv).
