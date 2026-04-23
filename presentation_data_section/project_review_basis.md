# Project Review Basis

This presentation package was rebuilt after reviewing the substantive analytic parts of the repo across docs, notebooks, scripts, and frozen outputs.

## Repo Surfaces Reviewed

### Docs and framing

- `README.md`
- `docs/Project_Ouline.md`
- `bias_analysis.txt`
- `pyproject.toml`

### Substantive notebooks

- `notebooks/01_data_access.ipynb`
- `notebooks/02_data_preparation.ipynb`
- `notebooks/03_exploratory_analysis.ipynb`
- `notebooks/04_modeling_or_analysis.ipynb`
- `notebooks/05_visualization_and_outputs.ipynb`
- `notebooks/06_smr_by_state_supabase.ipynb`
- `notebooks/about_the_data.ipynb`
- `notebooks/risk_per_100k_by_state.ipynb`
- `notebooks/risk_per_100k_by_state_executed.ipynb`
- `notebooks/shootings_by_city.ipynb`
- `notebooks/Research Visuals.ipynb`
- `Research Visuals.ipynb`

### Notebook stubs or low-content notebooks checked for scope

- `notebooks/data_visualizations_supabase.ipynb`
- `notebooks/risk_per_100k.ipynb`
- `notebooks/supabase_test.ipynb`
- `notebooks/visual_talking_points.ipynb`

### Source scripts and generated-story code reviewed

- `src/convert_enrollment_wide_to_long.py`
- `src/manim_gapminder.py`
- `src/manim_state_map.py`
- `src/manim_bias_story.py`
- `src/manim_data_story.py`
- `src/manim_model_summary.py`
- `src/manim_insights.py`
- `src/plotly_insights.py`
- `src/plotly_dc_cinematic_map.py`

### Frozen outputs reviewed

- `outputs/project_data_overview.html`
- `outputs/project_population_bias.html`
- `outputs/project_model_summary.html`
- `outputs/project_model_summary_docx/Project_Model_Summary.html`
- `outputs/plotly_insights/insights_summary.md`
- notebook outputs embedded in the executed `.ipynb` files above

### Non-analytic or empty files checked for relevance

- `src/streamlit_app.py` is empty
- `src/modal_app.py` is empty
- `src/simulate_law.py` is empty
- `src/test.py` is a minimal Manim smoke test

## Key Conclusions Pulled Into The Slides

- The project is structured around an `incident` anchor table, with shooter, victim, and weapon detail linked by `Incident_ID`.
- State-year modeling depends on enrollment and policy tables merged by `State + Year`.
- The repo explicitly treats the 2019 enrollment break as a comparability problem and rebuilds the series for cleaner longitudinal use.
- The main analytic window is `1987-2025`, with the restriction justified by comparability and panel-homogeneity concerns.
- The state-year panel preserves zero-incident years and uses enrollment to normalize risk.
- The modeling stack is not a single model:
  - Negative Binomial baseline for incident counts
  - Two-way fixed effects risk specifications with state effects, year effects, and log enrollment offset
  - Final preferred count specification in the README: Poisson with state-clustered standard errors
  - Negative Binomial for total victims
  - Gamma GLM with log link for victims per incident
- The clearest full-project finding is a late-year national rise in risk, with `0.71` incidents per `100,000` students in `2023` as the peak value cited in the repo summaries.
- Most single-policy effects in the count and total-victims models are weaker and less stable than the broader time pattern.
- The clearest policy-linked severity result is the positive `k12_settings_law` signal in the incident-level Gamma model.
- D.C. is a genuine rate-based outlier in the repo:
  - `16.36` incidents per `100,000` students in `2023`
  - strong positive D.C. state effect in the risk models, around `+2.86`
  - `SMR = 10.50` in the SMR notebook
- D.C. is not a significant severity outlier in the victims-per-incident Gamma model.
- Excluding D.C. does not overturn the broader policy story; the main difference shown in the robustness table is that `report_stolen_lost_law` crosses conventional significance when D.C. is removed.
- The repo also contains a second D.C. framing in `outputs/plotly_insights/insights_summary.md`, where D.C. appears in the dense-policy / lower-recent-burden comparison group. That is why the D.C. section in the presentation is framed as metric-sensitive rather than one-dimensional.
