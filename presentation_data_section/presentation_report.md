# Data And Findings Report

## Overview

This section of the project is strongest when read as a cleaned state-year risk analysis supported by event-level detail. The incident table is the anchor record, while shooter, victim, and weapon tables provide participant and event detail through `Incident_ID`. Enrollment and policy tables supply the state-year context needed for normalized comparisons and modeling.

## Data Structure And Preparation

The repo does not rely on raw incident counts alone. It standardizes keys across sources, preserves zero-incident years, and rebuilds the enrollment series after identifying a comparability break around 2019. That repair matters because the main rate used throughout the project is `incident_count / total_students x 100,000`, so any instability in the enrollment denominator would distort the risk analysis.

For that reason, the main analytic window is the cleaner `1987-2025` panel rather than the full historical record. The emphasis is on comparability and stable state-year inference, not on maximizing the raw span of years.

## Modeling Framework

The repo uses a small model stack rather than a single specification. Negative Binomial models appear as the baseline for incident counts and for total victims. The state-year risk models add state fixed effects, year fixed effects, and a log enrollment offset to separate risk from population size. The README then identifies Poisson with state-clustered standard errors as the preferred final count specification because estimated overdispersion is negligible and the NB2 form is less stable.

Severity is modeled separately at the incident level with a Gamma GLM for victims per incident. That separation is important because it distinguishes the probability of incidents from the harm observed once an incident occurs.

## Main Findings

The strongest full-project finding is the late-year increase in national incident risk. The reviewed outputs cite a peak rate of `0.71` incidents per `100,000` students in `2023`, and the Plotly insight summary reports `0.30` in `2025`. Across the incident-count and total-victims models, the broader time pattern is stronger and more stable than most individual policy coefficients.

The clearest policy-linked severity result appears in the victims-per-incident Gamma model, where the K-12 settings measure stands out more clearly than the other policy terms. That makes the project’s strongest substantive story a combination of rising recent risk and limited, uneven policy signals rather than a broad claim that many single laws show large independent effects.

## D.C. Findings

D.C. is a strong rate-based outlier in the project. The notebooks show `11.18` incidents per `100,000` students in `2022` and `16.36` in `2023`, and the state effect for D.C. in the risk models is strongly positive at roughly `+2.86`. The SMR notebook reinforces that result by reporting `SMR = 10.50`.

At the same time, D.C. is not a universal outlier across every model. In the incident-level Gamma severity model, D.C. is not statistically significant as a severity outlier once an incident occurs. The no-D.C. robustness check also leaves the broader project story largely intact, showing that D.C. is important in the data but not the sole explanation for the project’s main findings.
