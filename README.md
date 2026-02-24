# School Shootings Capstone Project

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

Template (multiline):

```bash
python src/convert_enrollment_wide_to_long.py \
    data/<file_name>.csv \
    data/<new_file_name>_long.csv
```

Concrete example:

```bash
python src/convert_enrollment_wide_to_long.py \
    data/2007_2016.csv \
    data/2007_2016_long.csvn>_long.csv
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

## Data Quality Notes

An enrollment anomaly review identified a sharp drop in 2019 that did not match prior trend behavior.

- **Observed issue:** abrupt structural break in national enrollment totals.
- **Likely cause:** inconsistent NCES reporting universe across source eras (template and inclusion-rule differences).
- **Corrective action:** rebuilt derived enrollment tables using a consistent download template and harmonized inclusion criteria.
- **Current status:** resulting longitudinal series is more comparable across years for incident-rate analysis.

For full diagnostics and methodology details, see `notebooks/about_the_data.ipynb`.

## Environment

- Python `>=3.12`
- Main dependencies are in `pyproject.toml` (pandas, plotly, matplotlib, supabase client, dotenv).

## About the Data Notebook Updates

The `notebooks/about_the_data.ipynb` notebook now includes presentation-ready diagnostics and architecture visuals to support the question:

**Are school shooting risks affected by policy/laws, and can we lower risk by changing laws?**

### Added visuals

- Entity Relationship Diagram (ERD) using Plotly
- Table row count comparison
- Column data type breakdown (categorical vs numeric vs datetime)
- Missing value percentage by column
- Missing value heatmap
- Year coverage timeline (data availability by year)
- Duplicate record check summary
- Enrollment dataset year-range alignment visualization
- Data ingestion pipeline flow diagram (ETL overview)
- Data source provenance chart

### Rendering note (Plotly in VS Code/Jupyter)

If Plotly does not render inline in your environment, use:

```python
import plotly.io as pio
pio.renderers.default = "notebook_connected"  # fallback: "browser"
```
And then render with:

```python
display(HTML(fig.to_html()))
```

