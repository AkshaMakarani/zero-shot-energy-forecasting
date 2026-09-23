# Stage 3 — Results and Methodological Findings

**Thesis:** *Temporal Granularity and Forecast Configuration for Zero-Shot Energy
Forecasting: A Scenario-Oriented Evaluation of Time-Series Foundation Models*

**Status as of 12 September 2026.** All notebooks have now been re-run with planned window
counts. Every number below is traceable to a CSV in `E:\Thesis\EnergyForcastModel`. Nothing
is provisional any more — where a conclusion moved between the 5-window and planned-window
runs, both figures are shown and the move is stated.

---

## 1. What is being tested

Three zero-shot foundation models — Chronos-Bolt-Small (~48 M), TimesFM-2.5-200M and
Moirai-1.1-R-Small (~14 M) — on three Monash energy datasets, each at three temporal
granularities:

| energy type | Monash config | native step | series used |
|---|---|---|---|
| load | `australian_electricity_demand` | 30 min | 5 |
| solar | `solar_10_minutes` | 10 min | 5 |
| wind | `wind_farms_minutely` | 1 min | 5 |

Granularities: native, hourly, daily. Model capacity is confounded with model identity
(48 M vs 200 M vs 14 M), which is a stated limitation, not a controlled factor.

### Research questions

- **RQ1** — Does temporal granularity change zero-shot accuracy once horizon is held
  constant in *physical time*?
- **RQ2** — How do context length and forecast horizon affect accuracy, and do those
  effects interact with granularity?
- **RQ3** — Which (model, granularity, context, horizon) configurations are deployable,
  and under what operational scenario?
- **RQ4** — Can series-level descriptors predict deployability on an unseen energy type?

---

## 2. The primary metric, and why it is not MASE

**relMAE** = forecast MAE ÷ MAE of a horizon-matched seasonal-naive benchmark computed at
the same origin. A value below 1.0 means the model beats the naive benchmark it would have
to replace in production.

Two defects had to be removed before any cross-granularity number meant anything.

### 2.1 The MASE denominator was being computed on the context window

The original code clamped the seasonal period with `m = min(m, ctx - 1)` and fell back to
`m = 1` when the context was short. The denominator therefore changed as the sweep changed
the context, so a change in "MASE" was partly a change in the normaliser.

The symptom was unmissable once looked for. Wind scales across the three granularities:

| | daily | hourly | native |
|---|---|---|---|
| before | 22.05 | 13.08 | **1.04** |
| after | 22.94 | 21.42 | **21.58** |

One physical signal cannot have a 20× spread in its own scale across representations. After
fixing the reference segment to a fixed reserved block, wind native mean MASE fell from
19.677 to a median of 0.309.

### 2.2 The corrected MASE still tracked the horizon-to-season ratio

Spearman(h/m, corrected MASE) = **+0.736, p = 0.024**. MASE was still partly measuring the
experimental design rather than forecast quality. Replacing it with relMAE removed this:
Spearman(h/m, relMAE) = **−0.239, p = 0.535**.

**relMAE is the headline metric. MASE is retained in every results file as a secondary
column so the correction is auditable.**

---

## 3. The evaluation-coverage defect — the most important methodological finding

### 3.1 What went wrong

Every Stage 3 notebook originally fixed `N_WINDOWS = 5`. A window spans
`horizon × step_minutes` of *real time*, so a fixed count means each cell is evaluated over
a wildly different stretch of reality — in the exact ratio of the metering rates, 30 : 10 : 1:

| cell | 5 windows × 24 steps covers |
|---|---|
| load, native (30 min) | 2.5 days |
| solar, native (10 min) | 20 hours |
| wind, native (1 min) | **2 hours** |
| any cell, daily | 120 days |

Those rows sat in the same table looking equally well supported.

### 3.2 It changed a conclusion

3B was re-run with planned window counts (30–90 windows per series instead of 5). Same
models, same metric, same horizon, native granularity, 128 → 2048 context:

| cell | 5 windows | planned windows | |
|---|---|---|---|
| load, native | 1.135 → 1.162 (**−2.4 %**) | 1.043 → 0.719 (**+31.1 %**) | **sign flip** |
| load, hourly | 1.057 → 0.827 (+21.7 %) | 0.979 → 0.862 (+11.9 %) | same direction |
| load, daily | 0.938 → 0.803 (+14.4 %) | 0.851 → 0.605 (+28.9 %) | same direction |
| solar, native | 1.057 → 1.098 (**−3.9 %**) | 1.193 → 1.158 (**+3.0 %**) | **sign flip** |
| solar, hourly | 0.983 → 1.056 (**−7.4 %**) | 1.146 → 1.110 (**+3.2 %**) | **sign flip** |
| wind, native | 0.725 → 0.660 (+9.0 %) | 0.594 → 0.593 (+0.1 %) | effect vanishes |
| wind, hourly | 1.039 → 1.139 (−9.6 %) | 1.040 → 1.224 (−17.7 %) | same direction |

Positive = longer context helps. **Three of seven cells reverse sign, and a fourth
(wind native) loses its apparent effect entirely.** The original conclusion — "more context
does not help these models" — was an artefact of under-powered evaluation.

The effect on statistical power is just as stark. Holm-corrected across the same 84 context
comparisons, **4 survived at 5 windows; 25 survive with planned windows.**

This matters beyond this thesis. Zero-shot evaluation on high-frequency data is cheap to
under-power precisely where the data is most plentiful, because a fixed window budget buys
the least calendar time exactly where the sampling rate is highest.

### 3.3 The fix

`C.windows_for()` takes the **larger** of a statistical floor (≥ 30 windows, so paired
Wilcoxon tests are viable) and a time floor (≥ 14 days of evaluated duration), capped for
affordability. `C.plan_windows()` prints, per cell, the window count, the evaluated
duration and a `meets_floors` flag **before anything runs**. Cells that cannot meet both
floors are reported as such rather than silently dropped.

3B, 3C and 3D were rebuilt on this basis and re-run. All three completed with **zero
runtime errors** (24,300 + 23,400 + 34,890 forecasts), anchored origins verified identical
across every context within a cell, and the MASE scale verified constant per series.

---

## 4. Three further corrections

**TimesFM's context floor.** TimesFM patches its context in blocks of 32, so any context
below 32 fails at runtime — 300 windows failed this way in the first 3D run. This is now
declared (`MODEL_MIN_CONTEXT`) and enforced at planning time. 3H ran 34,200 forecasts with
**zero runtime errors**.

**The 40-day history grid in 3D.** Enforcing the 32-step floor removed daily granularity
from the de-confounded comparison entirely, because 30 days of daily data is 30 steps.
Since the models accept 32–2,048 steps, one history *duration* only fits two granularities
at once if their step sizes differ by less than 64×:

| history | 30-min steps | hourly | daily | fits? |
|---|---|---|---|---|
| 7 d | 336 | 168 | 7 | daily below the floor |
| 30 d | 1,440 | 720 | 30 | daily below the floor by two steps |
| **40 d** | **1,920** | **960** | **40** | **all three fit** |
| 90 d | 4,320 | 2,160 | 90 | first two above the cap |

40 days is the smallest round figure giving `load` a genuine three-way comparison. Solar
and wind cannot get one at any history (10-min-to-daily is 144×, 1-min-to-daily is 1,440×).
**That is a deployment finding, not a gap: at fine metering there is no single history
window representable at both ends of the granularity range.**

**Model-loading cost.** `C.run()` was being called once per configuration, reloading all
three models each time — 144 load/free cycles in 3H to support 27 minutes of inference. The
rebuilt notebooks build every window first and score in a single pass: **three loads each.**

---

## 5. Findings by research question

### RQ1 — granularity at matched physical lead time

**The answer is that optimal granularity tracks lead time.** From 3D, at matched history and
matched lead, reading only rows where at least two granularities were actually in contention:

| energy type | history | 1 h lead | 6 h lead | 24 h lead | 48 h lead |
|---|---|---|---|---|---|
| load | 40 d | hourly **0.364** ≈ native 0.367 | hourly **0.564** < native 0.661 | daily **0.641** < native 0.788 < hourly 0.870 | hourly **0.797** < daily 0.846 |
| load | 7 d | native **0.381** < hourly 0.396 | hourly **0.655** < native 0.815 | hourly **0.936** < native 1.017 | — |
| solar | 40 d | — | — | daily **0.764** < hourly 1.091 | daily **0.778** < hourly 0.974 |
| solar | 7 d | native **0.794** < hourly 0.837 | hourly **0.904** < native 1.004 | — | — |
| wind | 40 d | — | — | daily **1.005** < hourly 1.238 | daily **0.980** < hourly 1.064 |

The pattern is consistent across all three energy types: **fine granularity wins at short
lead times, coarse granularity wins at long ones.** At a 1-hour lead the native or hourly
representation is best; by 24–48 hours ahead, daily aggregation wins on every type that can
be compared there. Across the 13 rows with a genuine comparison, hourly wins 6, daily 5,
native 2 — and which one wins is predicted by the lead time, not by the energy type.

This is the de-confounded answer Stage 2 could not give, and it is directly actionable:
**aggregate your data to roughly match your forecast lead time.** Forecasting a day ahead
from 1-minute meter readings is not using more information, it is asking a harder question.

Two caveats to report honestly:

- 13 of the 30 (type, history, lead) rows had only **one** granularity feasible, so they are
  not comparisons. The notebook now prints `new_n_gran` so these cannot be quoted by
  mistake.
- Of the 13 rows the 5-window and planned-window runs have in common, **5 pick a different
  granularity**. The planned-window figure is the one to quote.

The infeasibility staircase stands as a result in its own right: 64 of 108 combinations are
physically unreachable, each for a stated arithmetic reason.

### RQ2 — context, horizon, and their interaction

**Context (3B, planned windows, pooled over models):**

| cell | 128 | 256 | 512 | 1024 | 2048 | verdict |
|---|---|---|---|---|---|---|
| load daily | 0.851 | 0.838 | 0.720 | 0.642 | **0.605** | monotone, large |
| load hourly | 0.979 | 0.896 | 0.887 | 0.870 | **0.862** | monotone, modest |
| load native | 1.043 | 0.975 | 0.858 | 0.735 | **0.719** | monotone, large |
| solar hourly | 1.146 | 1.088 | **1.049** | 1.089 | 1.110 | shallow U, optimum at 512 |
| solar native | 1.193 | 1.188 | 1.188 | 1.163 | **1.158** | essentially flat |
| wind hourly | **1.040** | 1.030 | 1.119 | 1.234 | 1.224 | **more context hurts** |
| wind native | 0.594 | 0.609 | **0.588** | 0.588 | 0.593 | flat |

Three distinct regimes, and they are physically interpretable:

1. **Load rewards history without saturating** — up to 2,048 steps it is still improving,
   because demand carries weekly and seasonal structure a longer window can exploit.
2. **Solar has an optimum, not a slope.** Hourly solar is best at 512 steps (21 days) and
   gets worse either side. Extra history adds seasonal drift, not signal.
3. **Wind is penalised by history.** Hourly wind degrades 17.7 % from 128 to 2,048 steps.
   Wind is close to a random walk; a longer context dilutes the recent state that actually
   carries the forecast. This is a clean negative result and the sharpest differentiator
   between the three signals.

**Horizon (3C):** the dominant lever wherever the metering is fine.

| cell | h=6 | h=12 | h=24 | h=48 | degradation |
|---|---|---|---|---|---|
| load native | 0.534 | 0.716 | 0.858 | 0.969 | −81.6 % |
| solar native | 0.868 | 0.858 | 1.188 | 1.772 | −104.2 % |
| wind native | 0.570 | 0.613 | 0.588 | 0.663 | −16.2 % |
| load hourly | 0.562 | 0.654 | 0.887 | 0.818 | −45.4 % |
| load daily | 0.760 | 0.812 | 0.720 | 0.683 | +10.2 % |
| solar daily | 0.740 | 0.798 | 0.812 | 0.783 | −5.9 % |
| wind daily | 0.940 | 0.975 | 1.000 | 1.001 | −6.4 % |

**Horizon in steps hurts far more at fine granularity than at coarse.** Doubling the step
count costs a native-resolution forecast dearly and a daily forecast almost nothing — load
daily even improves. That is the same phenomenon RQ1 found from the other direction, and the
two results corroborate each other.

**Interaction (3H):** fitting `log₁₀ relMAE ~ log₂(context) + log₂(horizon)`:

| energy type | n | context coef | horizon coef | interaction | R² (no int.) | R² (with) | p |
|---|---|---|---|---|---|---|---|
| load | 7,200 | −0.0519 | +0.0678 | −0.0111 | 0.04170 | 0.04259 | **0.0097** |
| solar | 3,924 | −0.0327 | **+0.2028** | −0.0099 | 0.09048 | 0.09074 | 0.289 |
| wind | 11,796 | **+0.0282** | +0.0657 | −0.0082 | 0.01413 | 0.01434 | 0.106 |

Wind's positive context coefficient independently confirms the 3B finding. The interaction
is significant only for load and practically negligible even there — R² moves by 0.0009 on
n = 7,200. **The honest statement is that the optimal context does not meaningfully depend
on how far ahead you forecast**, which simplifies the deployment advice considerably.

**Significance (3E, Holm-corrected across 84 context comparisons): 41 significant at raw
α = 0.05, 25 surviving Holm** — up from 4 in the 5-window run. The largest surviving effects
are a monotone staircase on load with TimesFM: 128→256 +10.4 %, 256→512 +27.9 %,
512→1024 +28.5 %, 1024→2048 +17.5 % (daily); and 128→256 +19.5 %, 256→512 +16.2 %,
1024→2048 +11.9 % (native). Five surviving comparisons are *degradations*, four of them on
solar or wind — the strongest being wind hourly 512→1024 at **−29.4 %**.

**Model comparison (3E, Holm across 27 pairwise tests): 10 survive. TimesFM wins 7,
Chronos 3, Moirai 0.** Moirai is never best on any cell that survives correction. It is also
the smallest model at ~14 M parameters, so capacity and identity remain confounded — state
this as a limitation rather than a finding about architecture.

### RQ3 — deployable configurations

From 3F: **216 of 333 scored configurations beat the seasonal-naive benchmark.** The best
per energy type, quoting the worst-series figure rather than the median:

| energy type | configuration | median relMAE | worst series | p90 |
|---|---|---|---|---|
| load | native, 40 d history, 1 h lead, TimesFM | 0.227 | 0.438 | 1.27 |
| load | daily, 2048 d history, 24 d lead, TimesFM | 0.278 | **0.299** | 0.49 |
| solar | hourly, 7 d history, 1 h lead, Chronos | 0.644 | 0.895 | 5.78 |
| solar | daily, 125 d history, 48 d lead, TimesFM | 0.738 | **0.783** | 0.96 |
| wind | native, 8.5 h history, 6 min lead, TimesFM | 0.522 | 0.685 | 1.57 |
| wind | daily, 62 d history, 12 d lead, TimesFM | 0.941 | **0.991** | 1.30 |

**Read the worst-series and p90 columns together with the median.** Several configurations
with excellent medians have a p90 above 3 — solar hourly at a 1-hour lead has a median of
0.644 but a p90 of 5.78, meaning one forecast in ten is roughly six times worse than the
naive benchmark. The daily configurations are conspicuously the *stable* ones: their p90 sits
near their median, so they are the ones to recommend where a bad forecast is expensive.

Wind remains the hardest signal: it has deployable configurations, but 15 of the 111 job
slots are not deployable at any model, and almost all of those are wind or fine-grained
solar at multi-hour leads.

### RQ4 — predicting deployability on an unseen energy type

Unchanged by the re-run, as expected — 3G's core reads the descriptor features and the
Stage 2 results, not the sweep geometry.

| held out | model MAE | mean-baseline MAE | Spearman | p | AUC |
|---|---|---|---|---|---|
| load | **0.124** | 0.134 | +0.315 | 0.035 | 0.713 |
| solar | **0.125** | 0.154 | **+0.820** | 5.7×10⁻¹² | **0.881** |
| wind | 0.353 | **0.090** | −0.242 | 0.122 | 0.422 |

**Two of three succeed, and the failure is interpretable.** Wind's own mean is already an
excellent predictor of its own relMAE (baseline MAE 0.090, the lowest of the three), so
there is very little variance left to explain and a rule fitted on load + solar extrapolates
badly onto it. The conclusion to report is bounded, not universal: *descriptor-based transfer
works between energy types with comparable behavioural variety, and fails onto a type whose
behaviour is near-homogeneous.*

## 6. Notebook inventory

| notebook | answers | forecasts | status |
|---|---|---|---|
| `refined_stage2_baseline` | baseline at fixed context | 1,245 | run |
| `refined_stage3a_data_characteristics` | descriptors for RQ4 | — | run |
| `refined_stage3b_context_sweep` | RQ2, context | 24,300 | run, 0 errors |
| `refined_stage3c_horizon_sweep` | RQ2, horizon | 23,400 | run, 0 errors |
| `refined_stage3d_fixed_lead_time` | RQ1, de-confounded | 34,890 | run, 0 errors |
| `refined_stage3e_significance` | Wilcoxon + Holm | — | run |
| `refined_stage3f_selection_matrix` | RQ3 | — | **re-run after the patch below** |
| `refined_stage3g_rq4_validation` | RQ4, leave-one-type-out | — | run |
| `refined_stage3h_context_horizon_cross` | RQ2 interaction | 34,200 | run, 0 errors |

All share `refined_stage_common.py` (`__version__ = "2026.09.10-relmae"`). Every notebook
asserts that version at import after an `importlib.reload`, because a Jupyter kernel serving
a cached older copy surfaced as a missing-column error much later in the run.

### Verification performed on the re-run

- Zero runtime errors in all three sweeps; every model scored every window.
- Built geometry matches the printed plan exactly, per series and per cell.
- **Anchoring holds**: within every (series, granularity) cell, all five contexts forecast
  an identical set of origins. 0 violations.
- **MASE scale is constant** per (series, granularity) across all contexts. 0 violations.
- No cell was left with zero usable rows. Solar native loses 43–55 % of windows to the
  degenerate (constant-target) filter — night-time zeros, which is correct behaviour.

### Two reporting defects found during verification and fixed

**3D's sensitivity table counted grid changes as conclusion changes.** The history grid moved
from 30 d to 40 d and gained a 48 h lead between runs, so 28 of 41 rows compared something
against nothing. The printed "33 of 41 rows changed" was meaningless. Corrected: **5 of the
13 rows the two grids share pick a different granularity.** The cell is now self-contained
and re-runnable without repeating the 161-minute inference.

**3F's `worst_series_relMAE` was the maximum over windows, not the worst series.** relMAE is
unbounded when the seasonal-naive benchmark MAE approaches zero (solar at dawn, where
repeating yesterday is nearly exact), so the column reached 1.8 × 10⁷ and was uninformative
in 83 of 333 configurations. It is now the worst per-series median — what the name always
claimed — with a `p90_relMAE` tail-risk column alongside and `max_window_relMAE` retained as
a diagnostic. **Deployability verdicts use the median and are unaffected.**

The rebuilt notebooks archive the previous CSV as `*_5window.csv` and compare the two
gradients directly. Keep those archives — that comparison is the methodology-chapter
evidence.

## 7. Outstanding

1. **Re-run `refined_stage3f_selection_matrix`** to pick up the corrected worst-case columns
   (reads CSVs only, takes seconds).
2. **Re-run the last cell of `refined_stage3d_fixed_lead_time`** for the corrected
   sensitivity table (now self-contained — the header cell plus that one cell is enough).
3. Decide how to report wind at native resolution, which cannot reach the 14-day evaluation
   floor at any affordable window count (90 windows × 24 steps × 1 min = 1.5 days). Either
   state it as a limitation or give that one cell a dedicated long run.
4. Optional: 3H used a relaxed 1-day time floor and a 120 cap while 3B/3C/3D use 14 days and
   90. The difference is deliberate and documented in 3H, but be ready to justify it — 3H
   tests a contrast *between* cells, the others produce deployment recommendations.
5. Consider whether Moirai's consistent last place warrants a sentence on the
   capacity/identity confound in the limitations chapter.
