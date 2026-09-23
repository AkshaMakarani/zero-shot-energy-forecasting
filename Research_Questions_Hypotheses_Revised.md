# Research Questions, Objectives and Hypotheses — revised against corrected evidence

**Thesis:** *Temporal Granularity and Forecast Configuration for Zero-Shot Energy Forecasting:
A Scenario-Oriented Evaluation of Time-Series Foundation Models*

**Revision date:** 12 September 2026.
**Supersedes:** `Research_Problem_Questions_Objectives_Hypotheses.md` (9 September 2026), which
is retained unchanged for comparison.

> ### Why this revision exists
>
> The 9 September document was written before two metric defects were found and fixed. Its
> hypotheses cite MASE figures produced by a denominator computed on the context window, and
> those figures are not valid. In one case — **H3, described in that document as "the
> mechanistic core of the thesis"** — the phenomenon the hypothesis was built to explain
> turned out to be an artefact of the defect, and the hypothesis is contradicted by the
> corrected data.
>
> Every hypothesis below now carries a verdict and the evidence behind it. **RQ5
> (domain adaptation / fine-tuning) is out of scope**, and objective O8 is withdrawn.

---

## 1. Research questions (current wording)

### Main research question

> Under which combinations of temporal granularity, historical context length and forecast
> horizon do pretrained time-series foundation models produce **accurate and practically
> suitable** zero-shot forecasts for electricity demand, solar and wind generation, and to what
> extent can differences in performance be explained by measurable characteristics of the
> series?

**RQ1 — Temporal granularity.** How does zero-shot forecasting accuracy vary across temporal
granularities for electricity demand, solar and wind when the underlying physical signals are
held constant through controlled resampling?

**RQ2 — Forecast configuration.** How do historical context length and forecast horizon,
expressed as both observation counts and real-world temporal duration, affect zero-shot
forecasting accuracy, and how do these effects interact with temporal granularity?

**RQ3 — Scenario-oriented model selection.** Which model–granularity–context–horizon
combinations provide the most favourable **accuracy–inference-cost trade-off** for
representative energy forecasting scenarios?

**RQ4 — Explaining performance and failure.** To what extent can differences in zero-shot
forecasting performance, including poor-performing configurations, be explained by measurable
characteristics of the series, such as zero-inflation, volatility and seasonal structure?

*Changes from 9 September: wording is materially unchanged for RQ1–RQ4. RQ5 removed.*

---

## 2. Objectives

| # | Objective | Serves | Status |
|---|---|---|---|
| O1 | Characterise the series along seasonal structure, zero-inflation, volatility, inter-site heterogeneity | RQ4 | Complete |
| O2 | Controlled zero-shot pipeline: identical windows, anchored origins, rolling-origin validation | all | Complete — 82,590 forecasts, 0 errors |
| O3 | Quantify accuracy against granularity, signal held constant by resampling | RQ1 | Complete |
| O4 | Quantify context and horizon effects **crossed with granularity** | RQ2 | **Outstanding — data exists, test not yet written up (3I)** |
| O5 | Operational selection matrix **with inference cost** | RQ3 | **Outstanding — cost term never entered selection (3J)** |
| O6 | Test whether series characteristics explain success and failure | RQ4 | Complete |
| O7 | Statistical significance of differences using paired tests on shared windows | RQ1, RQ2 | Complete for model and context; **RQ1 granularity uses a cluster bootstrap, not a paired test** |
| ~~O8~~ | ~~Evaluate targeted fine-tuning~~ | ~~RQ5~~ | **Withdrawn — out of scope** |

---

## 3. Hypotheses, with verdicts

### H1 — Granularity affects zero-shot accuracy — **SUPPORTED, with a refinement**

> Zero-shot accuracy differs systematically across temporal granularities for the same
> underlying physical signal.

**Corrected evidence.** From the de-confounded comparison (3D), where history and lead time are
both fixed in physical time, a cluster bootstrap over the five series gives **6 of 13
comparable rows significant** at 95%. The refinement is that the effect is *conditional on lead
time*:

| lead time | comparable rows | significant |
|---|---|---|
| 1 h | 3 | **0** — differences of 0.003–0.043 |
| 6 h | 3 | 2 |
| 24 h | 3 | **3** |
| 48 h | 4 | 2 |

**The claim to make is therefore narrower and more useful than the original H1:** granularity
is irrelevant at very short lead times and matters increasingly as lead time extends, with
coarser representations winning from roughly six hours ahead. Wind is the exception — neither
of its rows is significant, and its intervals are very wide.

*The 9 September evidence for H1 ("mean MASE 7.38–8.23 native against 0.83–1.06 daily") came
from the defective denominator and must not be quoted.*

---

### H2 — Configuration effects interact with granularity — **SUPPORTED for load and solar, null for wind**

> The effect of context length and of forecast horizon on accuracy depends on the temporal
> granularity at which the signal is represented.

**Evidence.** Fitting `log₁₀ relMAE ~ log₂(x) × granularity` per energy type:

| interaction | load | solar | wind |
|---|---|---|---|
| context × granularity | **p = 3×10⁻⁷** | **p = 0.002** | p = 0.23 |
| horizon × granularity | **p = 3×10⁻¹⁹** | **p = 4×10⁻²⁴** | p = 0.076 |

The horizon × granularity effect for load roughly doubles explained variance (R² 0.012 → 0.024),
so it is substantive rather than merely detectable at large *n*. This is the second half of RQ2
and it has a positive answer.

*Note: Stage 3H tests context × horizon at native resolution only. It does not test this
hypothesis, despite its header claiming to. 3I is required.*

---

### H3 — ~~Context must span the dominant seasonal cycle~~ — **REFUTED**

> **Original:** Accuracy degrades sharply when the context window is shorter than the dominant
> seasonal cycle; extending the context yields large gains in that regime and diminishing gains
> once the cycle is spanned.

**This hypothesis is contradicted in both directions.** The corrected context sweep:

| cell | shortest context spans | ≥ 1 day at 128 steps? | gain, smallest → largest context |
|---|---|---|---|
| wind, native | 2.1 h | **No** | **+0.1 %** |
| solar, native | 21.3 h | **No** | **+3.0 %** |
| load, native | 2.7 d | Yes | **+31.1 %** |
| load, daily | 128 d | Yes | **+28.9 %** |
| load, hourly | 5.3 d | Yes | +11.9 % |
| solar, hourly | 5.3 d | Yes | +3.2 % |
| wind, hourly | 5.3 d | Yes | **−17.7 %** |

H3 predicts large gains in the first two rows and diminishing gains in the rest. **The observed
pattern is the exact inverse.** The two cells whose context fails to span a day gain
essentially nothing — including wind native, whose largest context (2,048 steps = 34 hours)
comfortably clears the daily cycle. The largest gains occur where the smallest context already
spanned several days.

**Why the original hypothesis looked compelling.** It was built to explain a catastrophic wind
failure at native resolution — "best MASE 19.077" in the 9 September table. That number was
produced by the MASE denominator being computed on the context window, which collapsed the wind
native scale to 1.04 against 22.05 at daily for the same physical signal. After the fix, wind
native's median MASE is **0.309** and its relMAE is among the best in the study.
**The failure H3 existed to explain never existed.**

#### H3′ — replacement hypothesis — **SUPPORTED**

> The benefit of additional context is governed by how much *deterministic structure beyond the
> dominant cycle* a signal carries, not by whether the context window spans that cycle.

Electricity demand carries daily, weekly and seasonal structure simultaneously and keeps
improving out to 2,048 steps. Solar is dominated by a single daily cycle plus weather noise and
saturates almost immediately. Wind at fine resolution is close to a random walk: additional
history adds noise rather than signal, and at hourly resolution it actively harms accuracy
(−17.7 %, and the regression coefficient on context is **positive** for wind, +0.028, against
−0.052 for load).

**This is a stronger and more transferable claim than H3**, because it predicts the sign of the
context effect from a measurable property of the signal rather than from an arithmetic
coincidence between window length and cycle length.

---

### H4 — Zero-inflation degrades accuracy, and aggregation reverses it — **SUPPORTED for solar; does not generalise**

> Zero-inflation degrades zero-shot accuracy, and temporal aggregation that removes exact zeros
> restores it.

**Evidence — solar is a clean natural experiment:**

| solar granularity | % exact zeros | median relMAE |
|---|---|---|
| native (10 min) | 55.3 % | 1.099 |
| hourly | 51.9 % | 1.006 |
| **daily** | **0.0 %** | **0.771** |

Monotone in both columns, and at daily resolution the zeros vanish entirely and all three
models beat the benchmark.

**But the mechanism does not generalise.** Wind is also heavily zero-inflated (34.9 % native,
41.1 % hourly, 24.4 % daily) and shows no such relationship — wind native is *the most
zero-inflated sub-daily cell that nonetheless scores best*. H4 should therefore be reported as
an explanation of the solar result specifically, not as a general law. Some of wind's apparent
advantage may be an artefact of an inappropriate benchmark (see Open Issues).

---

### H5 — The accuracy-optimal configuration is not the cost-optimal one — **SUPPORTED** *(new, serves RQ3)*

> The model that maximises accuracy is not generally the model that offers the best
> accuracy-per-unit-inference-cost, so a selection matrix built on accuracy alone does not
> answer RQ3.

**Evidence.** Median inference cost per forecast: Chronos 0.017 s, Moirai 0.032 s,
**TimesFM 0.295 s — roughly 17× Chronos.** On a Pareto frontier of median relMAE against
inference time:

| energy type | Pareto-optimal configs | dominated | frontier composition |
|---|---|---|---|
| load | 7 of 132 | 125 | mixed Chronos / TimesFM |
| solar | 5 of 105 | 100 | **entirely Chronos** |
| wind | 8 of 96 | 88 | **6 of 8 Chronos** |

On accuracy alone TimesFM wins 7 of 9 cells and 7 of 10 Holm-surviving comparisons. Once cost
enters, **Chronos is the rational choice for solar and wind**, and TimesFM buys roughly 0.07
relMAE on load for 18× the runtime. The overwhelming majority of configurations are dominated —
both slower and less accurate than an available alternative.

---

### H6 — Series descriptors predict deployability on an unseen energy type — **PARTIALLY SUPPORTED** *(serves RQ4)*

> Measurable series characteristics generalise well enough to predict zero-shot performance on
> an energy type not seen during fitting.

| held out | model MAE | mean-baseline MAE | Spearman | AUC |
|---|---|---|---|---|
| load | **0.124** | 0.134 | +0.315 (p = 0.035) | 0.713 |
| solar | **0.125** | 0.154 | **+0.820** (p < 10⁻¹¹) | **0.881** |
| wind | 0.353 | **0.090** | −0.242 (n.s.) | 0.422 |

Two of three transfers beat the baseline. Wind fails because its own mean is already an
excellent predictor of its own performance — there is little variance left to explain, and a
rule fitted on load and solar extrapolates badly onto it. **Report the claim as bounded:**
descriptor-based transfer works between energy types with comparable behavioural variety and
fails onto a type whose behaviour is near-homogeneous.

---

## 4. Summary of verdicts

| hypothesis | serves | verdict |
|---|---|---|
| H1 — granularity affects accuracy | RQ1 | Supported; effect conditional on lead time |
| H2 — configuration × granularity interaction | RQ2 | Supported (load, solar); null (wind) |
| ~~H3 — context must span the dominant cycle~~ | RQ2 | **Refuted in both directions** |
| H3′ — context benefit tracks deterministic structure | RQ2 | Supported |
| H4 — zero-inflation, reversed by aggregation | RQ4 | Supported for solar; not general |
| H5 — accuracy-optimal ≠ cost-optimal | RQ3 | Supported |
| H6 — descriptors transfer across energy types | RQ4 | Partially (2 of 3) |

---

## 5. Contribution statement (revised)

> This dissertation establishes temporal granularity as an experimental variable in zero-shot
> energy forecasting. Using a controlled design in which one physical signal is resampled to
> multiple resolutions and evaluated on identical rolling-origin windows with anchored origins,
> it shows that the optimal granularity tracks forecast lead time, that the effect of context
> length on accuracy is governed by the depth of deterministic structure in the signal rather
> than by seasonal-cycle coverage, and that the accuracy-optimal model is not the
> cost-optimal one. It further contributes two methodological corrections of general relevance
> to the field: that a scale-free error metric computed on the context window silently
> confounds the treatment it is used to evaluate, and that fixing the number of evaluation
> windows rather than the evaluated duration systematically under-powers the highest-frequency
> cells — a defect sufficient, in this study, to reverse the sign of a reported effect in three
> of seven cells.

---

## 6. Open issues to resolve before writing the results chapter

1. **Benchmark appropriateness.** relMAE uses seasonal-naive with m = 48/24/7 across
   native/hourly/daily, so cross-granularity comparisons divide by *different* benchmarks, and
   for wind a daily-seasonal benchmark is close to meaningless. Report against random-walk
   naive alongside seasonal-naive. Affects H1 and H4's wind discussion.
2. **Pretraining contamination.** `australian_electricity_demand` and `wind_farms` are in
   Moirai's LOTSA corpus; `solar_10_minutes` (al-pv-2006) and `wind_farms` appear in Chronos's
   pretraining group. "Zero-shot" should become "no task-specific fine-tuning", with an audit
   table in the methodology.
3. **Moirai is under-configured.** `patch_size` is hardcoded to 32; Salesforce's own example
   uses `"auto"`. Moirai's deficit is 2.81× on load native but 1.05× on wind native — the
   signature of a configuration mismatch, not uniform weakness. The claim "Moirai never wins"
   is not safe until this is re-run.
4. **Five series per energy type.** Load is capped at 5 by the archive, but solar has 137 and
   wind 339 available. The cluster bootstrap behind H1 has only 5 clusters.
5. **O4 and O5 outstanding** — notebooks 3I and 3J, both computable from existing results with
   no additional inference.
