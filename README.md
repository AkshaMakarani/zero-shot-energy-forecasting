# Temporal Granularity and Forecast Configuration for Zero-Shot Energy Forecasting

Analysis code for the Master's thesis *"Temporal Granularity and Forecast Configuration
for Zero-Shot Energy Forecasting: A Scenario-Oriented Evaluation of Time-Series
Foundation Models."*

The study evaluates three zero-shot time-series foundation models — Chronos-Bolt-Small,
TimesFM-2.5-200M and Moirai-1.1-R-Small on electricity demand, solar and wind
generation across native, hourly and daily temporal representations, considering
accuracy (relMAE vs a seasonal-naive benchmark), context/horizon sensitivity,
inference cost, and series descriptors.

## Repository structure

- `Thesis_Complete_Analysis.ipynb` - consolidated analysis notebook.
- `refined_stage2_baseline.ipynb` - fixed-context baseline.
- `refined_stage3a_data_characteristics.ipynb` - series descriptors.
- `refined_stage3b_context_sweep.ipynb` - context-length sweep.
- `refined_stage3c_horizon_sweep.ipynb` - horizon sweep.
- `refined_stage3d_fixed_lead_time.ipynb` - RQ1 matched physical lead time.
- `refined_stage3e_significance.ipynb` - paired significance tests + Holm correction.
- `refined_stage3f_selection_matrix.ipynb` - configuration selection matrix.
- `refined_stage3g_rq4_validation.ipynb` - descriptor transfer validation.
- `refined_stage3h_context_horizon_cross.ipynb` - context × horizon interaction.
- `refined_stage3i_granularity_interaction.ipynb` - configuration × granularity interaction.
- `refined_stage3j_cost_tradeoff.ipynb` - accuracy vs inference-cost Pareto analysis.
- `refined_stage3k_benchmark_robustness.ipynb` - seasonal-naive vs random-walk robustness.
- `refined_stage3l_moirai_patch.ipynb` - Moirai model board patch.
- `refined_stage4_visualizations.ipynb` - figure generation.
- `refined_stage_common.py` — shared data configs, model wrappers, metrics, windowing.

Result CSVs, figures, trained-environment files and the thesis manuscript are not
tracked in this repository (see `.gitignore`).

## Data

Series are drawn from the Monash Time Series Forecasting Archive
(Godahewa et al., 2021).
