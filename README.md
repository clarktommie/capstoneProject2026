# School Shootings Capstone Project
### Authors: Tori-Ana, Tommie, Clara, T Foster, Alli, & Diya

This repository contains data engineering and exploratory modeling work for analyzing K-12 school shooting risk trends and their relationship to firearm policy context across U.S. states.

## Project Objective

Primary research question:

Can school shooting risk be explained in part by policy/law environment, and does policy variation align with lower risk over time?

## Current Analysis Scope

The current workflow (documented in `notebooks/about_the_data.ipynb`) includes:

- Data integration across incident, shooter, victim, weapon, enrollment, and law-policy tables.
- Data quality diagnostics (schema checks, missingness, duplicate checks, year coverage validation).
- Enrollment anomaly review and correction for longitudinal comparability.
- Construction of state-year analytic panels.
- Incident normalization as rate per 100,000 students.
- Early policy linkage analysis (e.g., background check indicator joins and correlations).
- Distribution diagnostics motivating count-model choices (zero-heavy, right-skewed, overdispersed outcomes).

## Data Pipeline Summary

1. Extract source tables from Supabase in paginated pulls.
2. Standardize keys and datatypes (`State`, `Year`) across sources.
3. Transform enrollment data from wide year columns to long format where needed.
4. Build derived tables (national enrollment trend, incident rate metrics, state-year merged views).
5. Run QA checks before downstream modeling.

## Key Data Quality Notes

- A structural break in enrollment was detected around 2019 during anomaly review.
- Likely driver: inconsistent NCES reporting universe/template differences across eras.
- Corrective action: harmonized enrollment derivations to improve comparability for longitudinal rate analysis.

## Potential Bias: Sample Restriction and Comparability

## Sample Restriction (1987–Present)

The panel is restricted to 1987 onward due to a structural break in the underlying data-generating process prior to 1987.

Pre-1987 observations are not comparable in measurement, reporting coverage, and policy codification. Including those years would violate panel homogeneity assumptions and introduce non-structural noise into fixed effects and dynamic specifications.

This restriction is not outcome-based filtering. It is a data consistency adjustment to ensure:

- Comparable incident classification
- Consistent enrollment coverage
- Reliable policy coding
- Stable variance structure

All models are therefore estimated on the 1987–present balanced policy-enrollment panel.

## Repository Structure

```text
.
├── docs/
├── images/
├── notebooks/
│   ├── about_the_data.ipynb
│   ├── data_conversion.ipynb
│   ├── shootings_exploration.ipynb
│   └── ...
├── outputs/
│   └── plotly_insights/
└── src/
    ├── convert_enrollment_wide_to_long.py
    ├── plotly_insights.py
    ├── simulate_law.py
    ├── streamlit_app.py
    └── ...
```

## Environment Setup

Requirements:

- Python `>=3.12`
- Dependencies listed in `pyproject.toml`

Using `uv`:

```bash
uv sync
uv run python -m ipykernel install --user --name schoolshootings
```

## Configuration

Create a `.env` file with your Supabase credentials (already expected by notebook/script imports):

```env
SUPABASE_URL=...
SUPABASE_KEY=...
```

## How To Run

Launch notebooks:

```bash
jupyter lab
```

Run enrollment conversion script:

```bash
python src/convert_enrollment_wide_to_long.py <input_csv> <output_csv>
```

Example:

```bash
python src/convert_enrollment_wide_to_long.py data/2007_2016.csv data/2007_2016_long.csv
```

Expected output columns:

- `school_id`
- `state`
- `year`
- `total_students`

## Notebook of Record: About the Data

`notebooks/about_the_data.ipynb` is the primary technical notebook for:

- Source table profiling and integration checks.
- Visual diagnostics (ERD, row counts, dtypes, missingness, timeline coverage, ETL diagram).
- Incident-rate construction and validation.
- Initial policy merge diagnostics and model-readiness checks.

If Plotly does not render inline in your environment:

```python
import plotly.io as pio
pio.renderers.default = "notebook_connected"  # fallback: "browser"
```

## Modeling Status

Work in progress:

- State-year count modeling (Negative Binomial candidates).
- Two-way fixed effects specifications (state + year).
- Severity-focused alternatives (victims per incident and aggregated fatalities).

These sections are exploratory and being refined into a reproducible modeling pipeline.

## Reproducibility Notes

- Prefer frozen outputs in `outputs/` for downstream modeling instead of repeated live pulls.
- Keep state/year key normalization consistent before merges.
- Apply duplicate and missingness checks after every major join.

## Disclaimer

This project analyzes sensitive violence-related data for research and policy insight purposes only. Interpret findings with caution; associations are not causal claims without stronger identification design.
