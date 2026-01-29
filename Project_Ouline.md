# School Shootings – Capstone (2026)

Kickoff notes for the team. This README is meant to get everyone set up quickly and aligned on what lives where.

## Goal
- Explore and analyze the K–12 School Shooting Database (v4.1, 08/28/2025) alongside firearm law datasets to surface patterns, trends, and policy context.
- Produce reproducible notebooks and simple Python utilities that the whole group can run locally.

## Repo Layout
- `data/` – CSVs delivered with the project (e.g., `Public v4.1 K-12 School Shooting Database (8 28 2025) (2)__*.csv`, firearm law *_content/time_series files). Do not commit new large data without discussing first.
- `notebooks/` – Exploratory work. `dataset_exploration.ipynb` reads the database CSVs from `../data/...`.
- `src/` – Python package code (currently minimal, ready to expand).
- `outputs/` – Place derived tables, charts, and figures here (git-ignore large binaries if needed).
- `docs/` – Project docs, references, and write-ups.
- `US_STATE_LAWS.tsv` – Reference table for firearm laws by state.

## Quick Start (VS Code)
1) Install Python 3.12+.
2) From the repo root:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
   - `pip install -e .`
3) Open the folder in VS Code and select the `.venv` interpreter when prompted.
4) Open `notebooks/dataset_exploration.ipynb` in VS Code’s Notebook view (no Jupyter server needed beyond the built-in support). It expects the CSVs to stay in `data/` with the full filenames already present in the repo.

## Working Agreements
- Keep notebooks tidy: run “Restart & Run All” before pushing; clear stray tmp cells.
- Prefer reusable functions in `src/` when you find yourself reusing notebook code.
- Document any new datasets in this README (file name, source, refresh cadence).

## Next Up (suggested)
- EDA pass on incident/victim/shooter/weapon tables; log top questions and data quality notes.
- Sketch minimal data dictionary in `docs/`.
- Stand up plotting helpers (matplotlib is already in deps).

## How to Contribute
- Branch per task; keep PRs small.
- Include a short note in PRs: purpose, main changes, validation you ran.

If anything feels unclear, jot it here so the next person lands faster.***
