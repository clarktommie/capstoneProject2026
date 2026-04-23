# Slide 1 — Title

## Rising National Risk Is The Main Finding In The School Shootings Capstone Project

- Project title: `School Shootings Capstone Project`
- The strongest supported result in the repo is a late-year rise in national incident risk.
- That time-trend signal is stronger and more stable than most single-policy coefficients.

**Visual**
Create a clean title slide with one centered line chart sparkline in the lower third:
- `1987-2025` national `risk_per_100k`
- annotate `2023 peak: 0.71`
- subtitle: `National incident risk rises sharply after 2018 and remains elevated relative to earlier years`


# Slide 2 — Core Insight

## National Incident Risk Moves Into A Higher Post-2018 Regime

- The Plotly insight summary reports a peak national rate of `0.71` incidents per `100,000` students in `2023`.
- The same summary reports `0.30` in `2025`, still above much of the earlier series.
- Across the repo, the post-2018 period is the clearest and most defensible headline pattern.

**Visual**
New presentation-quality line chart:
- x-axis: `Year (1987-2025)`
- y-axis: `National risk per 100k students`
- thin muted line for `1987-2017`, bold accent line for `2018-2025`
- vertical reference line at `2018`
- callouts at `2023 = 0.71` and `2025 = 0.30`
- title on chart: `National Risk Enters A Higher Regime After 2018`


# Slide 3 — National Pattern

## The National Increase Is A Sustained Pattern, Not A Single-Year Spike

- The project’s core panel is built on `1987-2025`, with enrollment-normalized risk rather than raw counts.
- The repo explicitly treats the enrollment break around `2019` as a comparability issue and rebuilds the denominator series.
- The Plotly insight summary reports `2025 enrollment YoY = -0.35%`, which helps separate population change from incident-risk change.

**Visual**
New two-layer national chart:
- main layer: annual national `risk_per_100k`
- overlay: `3-year rolling average`
- small footer annotation: `Rates are normalized by total students, not raw incident volume`
- no bars, no clutter, no secondary legend beyond `Annual risk` and `3-year average`


# Slide 4 — DC vs Other States

## D.C. Looks Different Because Small Enrollment Turns Short Bursts Into Extreme Rates

- In the notebooks, D.C. reaches `11.175213` in `2022` and `16.356796` in `2023`, then falls to `4.025333` in `2025`.
- The `2023` spike comes from `12 incidents` over `73,364 students`, which produces a very high per-100k rate.
- D.C. is therefore a rate outlier driven by denominator size and burstiness, not proof that the national pattern is only a D.C. artifact.

**Visual**
New two-panel visual designed to avoid distortion:
- left panel: D.C. annual risk line for selected years `2019, 2022, 2023, 2024, 2025`
- right panel: horizontal dot plot of all states’ `2023 risk_per_100k` with D.C. highlighted in accent color
- add a small text box under the right panel: `Why D.C. separates: high incident count relative to a much smaller student base`
- do not use a single all-state line chart with D.C. on the same scale


# Slide 5 — Model Results

## The Final Count Model Supports A Strong D.C. Risk Effect, While The Severity Model Does Not

- The README identifies the final count specification as `Poisson` with `state fixed effects`, `year fixed effects`, `log enrollment offset`, and `state-clustered standard errors`.
- The repo reports NB2 `alpha ≈ 0.03`, statistically indistinguishable from zero, with incomplete convergence, which is why Poisson becomes the preferred model.
- In the count model, `C(State)[T.DC] = 2.8558` with `95% CI [1.716, 3.996]`, while in the Gamma severity model `C(State)[T.DC] = -0.3264` with `95% CI [-1.338, 0.686]`.
- In the Gamma severity model, `k12_settings_law = 0.6781` with `95% CI [0.302, 1.054]`, making it the clearest positive severity-side signal in the project.

**Visual**
New coefficient plot with two grouped sections:
- section 1: `Count model` showing `D.C. state effect` as a large positive coefficient with confidence interval
- section 2: `Severity model` showing `D.C. state effect` crossing zero and `k12_settings_law` clearly positive
- vertical zero line centered in the plot
- title on chart: `Count And Severity Models Tell Different Stories About D.C.`


# Slide 6 — Interpretation

## The Repo Supports A Trend Story First, A DC Outlier Story Second, And A Limited Policy Story Third

- The data work strengthens the national trend finding because the panel is cleaned, enrollment-adjusted, and restricted for comparability.
- The models reinforce that the strongest recurring result is rising incident risk over time, not a large set of stable policy effects.
- D.C. matters because it is a real rate outlier, but the severity model and no-D.C. robustness framing show it is not the sole driver of the project’s conclusions.
- The repo’s own framing remains associational rather than causal.

**Visual**
New synthesis slide graphic:
- three connected blocks labeled `Data`, `Models`, and `D.C.`
- under `Data`: `1987-2025 cleaned panel`, `risk per 100k`
- under `Models`: `Poisson clustered count model`, `Gamma severity model`
- under `D.C.`: `strong rate outlier`, `not a severity outlier`


# Slide 7 — Conclusion

## The Cleanest Final Takeaway Is Rising National Risk, With D.C. As A Distinct But Not Decisive Outlier

- National school-shooting risk rises sharply in the post-2018 period and peaks at `0.71` in `2023`.
- D.C. is a genuine rate outlier because its smaller enrollment magnifies short bursts into extreme per-capita values.
- The final count model shows a strong D.C. risk effect, but the severity model does not, so D.C. should be interpreted as a special case rather than the whole story.
- The project’s strongest evidence supports monitoring the national trend more than claiming broad, uniform single-policy effects.

**Visual**
New closing slide visual:
- one large statement card with `0.71 in 2023`
- one secondary card with `D.C. is a rate outlier, not a severity outlier`
- one footer line: `Main conclusion: the time trend is the most stable finding in the project`
