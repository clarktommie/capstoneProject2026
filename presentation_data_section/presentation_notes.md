# Presentation Notes

## Section Summary

The project is built as a relational data system rather than a single flat file. Incidents anchor the event record, while shooter, victim, and weapon tables add detail through `Incident_ID`. Enrollment and firearm-policy tables attach context through `State + Year`, allowing the project to move from individual events to a state-year risk panel.

The analytical value of the repo comes from the cleaning and comparability work before modeling. The pipeline standardizes keys, preserves zero-incident years, and treats the enrollment break around 2019 as a structural problem that must be repaired before rates are interpreted. The main panel therefore emphasizes comparability over raw historical span and focuses on the `1987-2025` window.

## Modeling Summary

The repo uses several model types for different questions. Negative Binomial models are used as baseline count models for incident counts and total victims. The main state-year risk work adds state fixed effects, year fixed effects, and a log enrollment offset. The README then identifies Poisson with state-clustered standard errors as the preferred final count specification because estimated overdispersion is close to zero and the NB2 form is not as stable.

Severity is handled separately from risk. An incident-level Gamma GLM with log link models victims per incident, which means the project can distinguish between the frequency of incidents and the harm observed once an incident occurs.

## Findings Summary

The clearest full-project result is the late-year national rise in incident risk. Across the repo’s summaries and charts, the strongest pattern is the increase visible in the post-2018 period, with `0.71` incidents per `100,000` students in `2023` as the peak value cited in the reviewed outputs. The latest value reported in the Plotly insight summary is `0.30` in `2025`.

Most single-policy effects in the incident-count and total-victims models are smaller and less stable than the broader time pattern. The strongest policy-linked severity result appears in the victims-per-incident Gamma model, where the K-12 settings measure stands out more clearly than the rest of the policy terms.

## D.C. Findings

D.C. is one of the strongest rate-based outliers in the project. The notebooks show D.C. at `11.18` incidents per `100,000` students in `2022` and `16.36` in `2023`. The state effect for D.C. in the risk models is strongly positive, around `+2.86`, and the SMR notebook reports `SMR = 10.50`.

That D.C. result does not extend cleanly to every metric. In the incident-level Gamma severity model, D.C. is not a statistically significant severity outlier once an incident occurs. The no-D.C. robustness check also leaves the broader project story intact, indicating that D.C. is an important outlier but not the sole driver of the project’s main conclusions.
