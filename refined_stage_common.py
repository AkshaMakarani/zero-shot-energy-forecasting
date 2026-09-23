"""
refined_stage_common.py — shared machinery for the refined thesis pipeline.

Every refined_stage*.ipynb imports from here, so the metric, the series selection and the
model wrappers are defined exactly once and cannot drift apart between notebooks.

WHAT IS DIFFERENT FROM THE ORIGINAL STAGE 2
-------------------------------------------
1. MASE scale is computed ONCE per (series, granularity) from a fixed reference segment —
   the part of the series that is never used as a forecast target — using the true seasonal
   period. It no longer depends on the context window, so MASE is comparable across every
   context length and horizon in the study.

     old:  m = min(m, ctx-1); if ctx.size - m < 16: m = 1
           -> wind native: m 1440 -> 511 -> 1, denominator collapsed to the mean 1-minute
              change, inflating MASE by roughly 12x. This is why wind native read 19.077.

2. Forecast origins are ANCHORED to the largest context in the experiment, so every context
   setting forecasts the identical target windows. Without this, a longer context silently
   shifts the test set later and the comparison is meaningless.

3. Degenerate windows (constant target — e.g. a wind farm offline, or solar at night) are
   detected, recorded and excluded from headline tables instead of being scored as perfect
   forecasts. In the original run 11 wind-native windows had MAE exactly 0 and 49% of
   solar-native windows scored MASE < 0.10.

4. Per-window diagnostics are saved (context length, seasonal m, scale, target statistics)
   so any future metric change is a CPU-only recomputation instead of another GPU run.

5. Results are summarised with median and IQR alongside the mean. On the original results
   the per-cell winner flips in 3 of 9 cells depending on which is used.

The models, the data and the seed are unchanged, so results remain comparable to Stage 2.
"""
from __future__ import annotations

import gc
import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- configuration

BASE = {
    "load":  {"config": "australian_electricity_demand", "native_freq": "30min",
              "native_minutes": 30,   "label": "AU Electricity Demand"},
    "solar": {"config": "solar_10_minutes",              "native_freq": "10min",
              "native_minutes": 10,   "label": "Solar Generation"},
    "wind":  {"config": "wind_farms_minutely",           "native_freq": "1min",
              "native_minutes": 1,    "label": "Wind Generation"},
}

GRANULARITIES = {
    "native": {"label": "minutes (native)", "resample": None,  "m": None},
    "1h":     {"label": "hourly",           "resample": "1h",  "m": 24},
    "1D":     {"label": "daily",            "resample": "1D",  "m": 7},
}

# steps per day at native metering — the true seasonal period, never clamped
NATIVE_M = {"load": 48, "solar": 144, "wind": 1440}

SERIES_PER_TYPE = 5
RANDOM_SEED     = 7
RESAMPLE_AGG    = "mean"

# selection eligibility, kept identical to Stage 2 so the same series are drawn
CONTEXT_LENGTH_BASELINE = 512
MIN_NATIVE = CONTEXT_LENGTH_BASELINE + 48 * 10

CHRONOS_MODEL       = "amazon/chronos-bolt-small"
TIMESFM_REPO        = "google/timesfm-2.5-200m-pytorch"
TIMESFM_REPO_LEGACY = "google/timesfm-2.0-500m-pytorch"
MOIRAI_REPO         = "Salesforce/moirai-1.1-R-small"

__version__ = "2026.09.10-relmae"   # the METRIC contract; unchanged by the additions below
# Capabilities added after the metric was frozen. New notebooks assert on these instead of
# bumping __version__, so notebooks already run against the metric above stay valid.
__features__ = {"plan_windows", "model_min_context", "extended_descriptors"}

# TimesFM reshapes context into 32-step patches, so a context below 32 raises
# "shape '[1, -1, 32]' is invalid". Declared here so a sweep can exclude those cells by
# design instead of collecting runtime errors (300 of them in the first 3D run).
MODEL_MIN_CONTEXT = {"chronos": 1, "timesfm": 32, "moirai": 1}

PANEL_CACHE = "refined_panel_cache.pkl"

# minimum reference segment beyond one seasonal period, for a stable MASE denominator
MIN_REF_BEYOND_M = 50


def step_minutes(etype: str, gran: str) -> int:
    """Wall-clock minutes represented by one step at this granularity."""
    if gran == "native":
        return BASE[etype]["native_minutes"]
    return {"1h": 60, "1D": 1440}[gran]


def seasonal_m(etype: str, gran: str) -> int:
    """True dominant seasonal period in steps. Never clamped to the context length."""
    g = GRANULARITIES[gran]
    return int(NATIVE_M[etype] if g["m"] is None else g["m"])


def human_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    if minutes < 1440:
        return f"{minutes/60:.1f} h"
    return f"{minutes/1440:.1f} d"


# ------------------------------------------------------------------------------ data layer

def load_monash(config: str):
    """Materialise every series. Kept for reference — prefer load_monash_selected().

    For wind_farms_minutely this holds 339 series of up to 527,040 points as float64,
    about 1.4 GB at once. On a machine that starts swapping, that is the difference
    between a slow load and an apparently hung one.
    """
    from datasets import load_dataset
    ds = load_dataset("Monash-University/monash_tsf", config, trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    out = []
    for row in ds[split]:
        arr = np.asarray(row["target"], float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out.append({"target": arr, "start": row.get("start", None)})
    return out


def load_monash_selected(config: str, n=SERIES_PER_TYPE, seed=RANDOM_SEED,
                         min_len=MIN_NATIVE, verbose=True):
    """Same seed-7 draw as load_monash + select_series, without holding the whole archive.

    Pass 1 measures each series and immediately discards it, keeping only (index, length).
    Pass 2 materialises just the five that were drawn. Peak memory drops from roughly
    1.4 GB to a few MB for wind, and the selection is provably identical because the
    eligibility test and the permutation see exactly the same lengths in the same order.
    """
    from datasets import load_dataset
    ds = load_dataset("Monash-University/monash_tsf", config, trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = ds[split]
    total = len(rows)

    sizes = []                                   # (row_index, finite_length)
    for i in range(total):
        arr = np.asarray(rows[i]["target"], float)
        k = int(np.isfinite(arr).sum())
        del arr
        if k:
            sizes.append((i, k))
        if verbose and total > 50 and (i + 1) % max(1, total // 10) == 0:
            print(f"      scanned {i+1}/{total} series", flush=True)

    elig = [t for t in sizes if t[1] >= min_len] or sorted(sizes, key=lambda t: -t[1])
    rng = np.random.default_rng(seed)
    pick = [elig[i] for i in rng.permutation(len(elig))[:n]]

    out = []
    for idx, _ in pick:
        row = rows[idx]
        arr = np.asarray(row["target"], float)
        arr = arr[np.isfinite(arr)]
        out.append({"target": arr, "start": row.get("start", None)})
    if verbose:
        print(f"      {total} series available -> selected rows "
              f"{[i for i, _ in pick]} (seed {seed})")
    return out


def to_series(entry, freq) -> pd.Series:
    try:
        idx = pd.date_range(pd.Timestamp(entry["start"]), periods=len(entry["target"]), freq=freq)
    except Exception:
        idx = pd.date_range("2015-01-01", periods=len(entry["target"]), freq=freq)
    return pd.Series(entry["target"], index=idx)


def select_series(raw, n=SERIES_PER_TYPE, seed=RANDOM_SEED, min_len=MIN_NATIVE):
    """Seed-7 draw, identical to Stage 2 so the refined run uses the same five series."""
    elig = [e for e in raw if e["target"].size >= min_len] \
           or sorted(raw, key=lambda e: -e["target"].size)
    rng = np.random.default_rng(seed)
    return [elig[i] for i in rng.permutation(len(elig))[:n]]


def get_data(cache_path: str = PANEL_CACHE, verbose: bool = True):
    """Load (or reuse) the five series per energy type at native resolution.

    Cached to disk so the seven refined notebooks do not each re-download Monash.
    Delete the cache file to force a refresh.
    """
    if os.path.exists(cache_path):
        data = pd.read_pickle(cache_path)
        if verbose:
            print(f"loaded cached panel <- {cache_path}")
            for et, lst in data.items():
                print(f"  {et:6s} {len(lst)} series, {min(len(s) for s in lst):,}"
                      f"–{max(len(s) for s in lst):,} points")
        return data

    print("No cache yet — downloading and scanning the Monash archive. This is the slow")
    print("step (wind is 339 series of up to 527,040 points) and happens ONCE; every")
    print("later notebook reads the cache in seconds.\n")
    data = {}
    for et, info in BASE.items():
        t0 = time.perf_counter()
        if verbose:
            print(f"  {et} — {info['config']}", flush=True)
        pick = load_monash_selected(info["config"], verbose=verbose)
        data[et] = [to_series(e, info["native_freq"]) for e in pick]
        if verbose:
            print(f"      {len(data[et])} series kept in {time.perf_counter()-t0:.0f}s\n",
                  flush=True)
    pd.to_pickle(data, cache_path)
    if verbose:
        print(f"cached -> {cache_path} ({os.path.getsize(cache_path)/1e6:.0f} MB)")
    return data


def resample_series(s: pd.Series, rule, agg: str = RESAMPLE_AGG) -> pd.Series:
    if rule is None:
        return s
    r = s.resample(rule)
    return (r.mean() if agg == "mean" else r.sum()).dropna()


def panel(data, etype: str, gran: str):
    """List of 1-D float arrays for one (energy type, granularity) cell."""
    rule = GRANULARITIES[gran]["resample"]
    return [resample_series(s, rule).values.astype(float) for s in data[etype]]


# ---------------------------------------------------------------------------- the metric

def reference_scale(values: np.ndarray, m: int, n_reserved: int):
    """Mean absolute seasonal difference over the segment never used as a forecast target.

    `n_reserved` is how many points at the end of the series are reserved as targets.
    Everything before that is fair game as a reference: it is model input, never truth.

    Returns (scale, reason). The scale is NaN — and the caller must treat the cell as
    infeasible — when the reference segment cannot support the seasonal period, or when
    the series is flat. Neither case silently falls back to m=1, which is the bug this
    replaces: that fallback is what turned wind native's denominator into the mean
    1-minute change and inflated its MASE roughly 12-fold.
    """
    m = int(max(1, m))
    ref = np.asarray(values[:len(values) - n_reserved], dtype=float)
    if ref.size < m + MIN_REF_BEYOND_M:
        return float("nan"), (f"reference segment {ref.size:,} points, needs "
                              f"{m + MIN_REF_BEYOND_M:,} for m={m}")
    d = np.abs(ref[m:] - ref[:-m])
    if d.size == 0:
        return float("nan"), f"no seasonal differences available at m={m}"
    s = float(d.mean())
    if s <= 1e-8:
        return float("nan"), ("series is flat over the reference segment — MASE is "
                              "undefined, so it is excluded rather than scored as 0")
    return s, ""


def seasonal_naive(values: np.ndarray, origin: int, horizon: int, m: int):
    """The seasonal-naive forecast for the same window the model is asked to predict.

    Repeats the last complete season, so for i in 0..h-1 the prediction is
    values[origin + i - m*(floor(i/m) + 1)]. Every index is strictly before the origin,
    which matters when the horizon exceeds the seasonal period (daily: h=14, m=7) —
    a naive `values[origin-m : origin-m+h]` would reach past the origin and leak.

    Returns None when the series does not extend far enough back.
    """
    m = int(max(1, m))
    h = int(horizon)
    i = np.arange(h)
    idx = origin + i - m * (i // m + 1)
    if idx.min() < 0:
        return None
    return np.asarray(values[idx], dtype=float)


def metrics(truth, pred, scale: float, snaive_mae: float | None = None) -> dict:
    """MAE, RMSE, sMAPE, MASE — plus relMAE when a seasonal-naive benchmark is supplied.

    MASE divides by the mean m-step seasonal difference, which is the textbook definition
    but is NOT matched to the forecast horizon. When the horizon is far shorter than the
    seasonal period the score is flattered: at 1-minute wind a 48-step forecast error is
    divided by day-over-day variation (m=1440), and the corrected Stage 2 run shows
    Spearman(h/m, median MASE) = +0.74, p = 0.02 across the nine cells.

    relMAE divides by the seasonal-naive forecast's error over the SAME horizon, so
    numerator and denominator span the same forecast distance and the value is comparable
    across granularities. Below 1.0 means "better than repeating the last season".
    """
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    e = pred - truth
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    den = np.abs(truth) + np.abs(pred)
    smape = float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(e) / np.where(den == 0, 1.0, den))) * 100)
    mase = mae / scale if (scale and np.isfinite(scale)) else float("nan")
    rel = (mae / snaive_mae
           if (snaive_mae is not None and np.isfinite(snaive_mae) and snaive_mae > 1e-12)
           else float("nan"))
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "MASE": mase, "relMAE": rel}


# ----------------------------------------------------------------------------- the windows

@dataclass
class Window:
    etype: str
    series_id: int
    gran: str
    context_steps: int
    horizon_steps: int
    m: int
    scale: float
    context: np.ndarray = field(repr=False)
    truth: np.ndarray = field(repr=False)
    origin_index: int = 0
    degenerate: bool = False
    truth_std: float = 0.0
    truth_pct_zero: float = 0.0
    truth_n_unique: int = 0
    snaive_mae: float = float("nan")   # horizon-matched benchmark, see metrics()

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("context")
        d.pop("truth")
        d["history_minutes"] = self.context_steps * step_minutes(self.etype, self.gran)
        d["lead_minutes"] = self.horizon_steps * step_minutes(self.etype, self.gran)
        return d


def build_windows(data, context_steps: int, horizon_steps, n_windows: int = 10,
                  granularities=None, etypes=None, anchor_context=None, verbose=True):
    """Rolling-origin windows with anchored origins and a fixed per-cell MASE scale.

    context_steps  : history fed to the model, in steps
    horizon_steps  : int, or dict {gran: int} when the horizon differs per granularity
    anchor_context : origins are placed so this many steps of history are always available.
                     Pass the LARGEST context in the sweep so every context setting
                     forecasts identical targets. Defaults to context_steps.

    Cells that cannot support the request are skipped and reported, never silently shrunk.
    """
    grans = granularities or list(GRANULARITIES)
    ets = etypes or list(BASE)
    W, skipped = [], []

    def _per_gran(x, gk, et):
        """Accept a scalar, a {gran: v} dict, or a {(etype, gran): v} dict.

        Returns None when this cell has no value — a sweep legitimately excludes cells
        that cannot hold the requested context, and those are skipped and reported
        rather than crashing the run.
        """
        if isinstance(x, dict):
            v = x.get((et, gk), x.get(gk))
        else:
            v = x
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if not np.isfinite(v) else int(v)

    for et in ets:
        for gk in grans:
            h = _per_gran(horizon_steps, gk, et)
            c = _per_gran(context_steps, gk, et)
            anchor = _per_gran(anchor_context, gk, et) if anchor_context is not None else c
            if h is None or c is None:
                skipped.append((et, gk, "-", "no context/horizon defined for this cell"))
                continue
            if anchor is None:
                anchor = c
            m = seasonal_m(et, gk)
            for sid, vals in enumerate(panel(data, et, gk)):
                n = len(vals)
                reserved = n_windows * h
                need = anchor + reserved
                if n < need:
                    skipped.append((et, gk, sid, f"need {need:,} points, have {n:,}"))
                    continue
                sc, why = reference_scale(vals, m, reserved)
                if not np.isfinite(sc):
                    skipped.append((et, gk, sid, why))
                    continue
                for w in range(n_windows):
                    end = n - w * h
                    st = end - h
                    ctx = vals[st - c:st]
                    tru = vals[st:end]
                    uniq = int(len(np.unique(tru)))
                    sn = seasonal_naive(vals, st, h, m)
                    sn_mae = float(np.mean(np.abs(tru - sn))) if sn is not None else float("nan")
                    W.append(Window(
                        etype=et, series_id=sid, gran=gk,
                        context_steps=int(c), horizon_steps=h,
                        m=m, scale=sc, context=ctx.copy(), truth=tru.copy(),
                        origin_index=int(st),
                        # a 1-step target is trivially "constant"; only flag genuinely
                        # flat multi-step targets (an offline farm, solar at night)
                        degenerate=bool(uniq <= 1 and len(tru) > 1),
                        truth_std=float(np.std(tru)),
                        truth_pct_zero=float(np.mean(tru == 0) * 100),
                        truth_n_unique=uniq,
                        snaive_mae=sn_mae,
                    ))
    if verbose:
        ctxs = sorted({w.context_steps for w in W})
        print(f"{len(W)} windows built (context={ctxs}, n_windows={n_windows})")
        n_sn = sum(1 for w in W if np.isfinite(w.snaive_mae))
        print(f"  seasonal-naive benchmark available for {n_sn}/{len(W)} windows "
              f"(relMAE); MASE is available for all")
        deg = sum(w.degenerate for w in W)
        if deg:
            print(f"  {deg} degenerate (constant target) — kept in the file, excluded "
                  f"from headline tables")
        for s in skipped:
            print(f"  SKIPPED {s[0]}/{s[1]}/series{s[2]}: {s[3]}")
    return W


def max_feasible_context(data, etype: str, gran: str, horizon: int, n_windows: int,
                         cap: int = 2048, floor: int = 16):
    """Largest context every series of this cell can supply with origins anchored.

    A daily-resampled solar series is 365 points long, so it cannot hold a 512-step
    context AND reserve 10 x 14 target steps. The original Stage 2 silently shrank the
    context here (`ctx = min(CONTEXT_LENGTH, st)`), which is why the daily cells ran at
    225-351 rather than 512. Making the limit explicit keeps the comparison auditable.
    """
    lens = [len(v) for v in panel(data, etype, gran)]
    if not lens:
        return None
    avail = min(lens) - n_windows * int(horizon)
    c = min(int(cap), int(avail))
    return c if c >= floor else None


def plan_contexts(data, horizons, n_windows: int, cap: int = 2048, verbose: bool = True):
    """Per-cell context plan, reported as a table before anything runs."""
    rows = []
    for et in BASE:
        for gk in GRANULARITIES:
            h = int(horizons[gk] if isinstance(horizons, dict) else horizons)
            c = max_feasible_context(data, et, gk, h, n_windows, cap)
            d = step_minutes(et, gk)
            rows.append({"etype": et, "gran": gk, "horizon_steps": h,
                         "max_context": c,
                         "history": human_duration(c * d) if c else "-",
                         "lead": human_duration(h * d)})
    # object dtype keeps an infeasible cell as None rather than turning the column to NaN
    tab = pd.DataFrame(rows).astype({"max_context": object})
    if verbose:
        print(tab.to_string(index=False))
    return tab


def min_context_for(models=("chronos", "timesfm", "moirai")) -> int:
    """Smallest context every named model can accept."""
    return max(MODEL_MIN_CONTEXT.get(m, 1) for m in models)


def windows_for(etype: str, gran: str, horizon: int, min_windows: int = 20,
                min_days: float = 14.0, max_windows: int | None = None) -> int:
    """Window count that satisfies BOTH a statistical floor and an evaluation-time floor.

    Fixing the window COUNT rather than the evaluated DURATION makes cells wildly unequal,
    because one window spans horizon x step_minutes of real time. With 10 windows of 48
    steps the original design evaluated 140 days of daily data but only 8 HOURS of
    1-minute wind — and presented both in the same table.

        load  native: 10 x 48 x 30 min = 10.00 d
        solar native: 10 x 48 x 10 min =  3.33 d
        wind  native: 10 x 48 x  1 min =  0.33 d      <- 30:10:1, the metering ratio

    min_windows keeps paired tests viable; min_days keeps a claim from resting on one
    afternoon. Taking the max of the two satisfies both.
    """
    d = step_minutes(etype, gran)
    need_time = int(np.ceil(min_days * 1440 / (int(horizon) * d)))
    n = max(int(min_windows), need_time)
    if max_windows is not None:
        # A short horizon at fine metering can demand tens of thousands of windows to
        # cover min_days (1-minute wind, 6-step horizon, 14 days -> 3,360 per series).
        # A cap keeps a sweep affordable; plan_windows() then reports the cell as not
        # meeting the time floor rather than pretending it does.
        n = min(n, int(max_windows))
    return int(n)


def plan_windows(data, horizons, contexts=None, min_windows: int = 20,
                 min_days: float = 14.0, max_windows: int | None = None,
                 verbose: bool = True):
    """Per-cell window plan, printed before a run the way plan_contexts() is.

    Caps the request at what each cell's shortest series can actually supply once the
    context is reserved, and reports the evaluated duration that results.
    """
    rows = []
    for et in BASE:
        for gk in GRANULARITIES:
            h = int(horizons[gk] if isinstance(horizons, dict) else horizons)
            d = step_minutes(et, gk)
            want = windows_for(et, gk, h, min_windows, min_days, max_windows)
            c = 0
            if contexts is not None:
                c = contexts.get((et, gk), contexts.get(gk)) if isinstance(contexts, dict) else contexts
                c = 0 if c is None or (isinstance(c, float) and not np.isfinite(c)) else int(c)
            shortest = min(len(v) for v in panel(data, et, gk))
            supply = max(0, (shortest - c) // h)
            n = int(min(want, supply))
            rows.append({"etype": et, "gran": gk, "horizon_steps": h,
                         "wanted": want, "series_supports": supply, "n_windows": n,
                         "evaluated": human_duration(n * h * d),
                         "meets_floors": bool(n >= min_windows
                                              and n * h * d >= min_days * 1440)})
    tab = pd.DataFrame(rows)
    tab["n_forecasts"] = tab.n_windows * 5 * 3      # 5 series x 3 models
    if verbose:
        print(tab.to_string(index=False))
        print(f"\ntotal forecasts for this plan: {int(tab.n_forecasts.sum()):,}"
              f"  (~{tab.n_forecasts.sum()*0.05/60:.0f} min of inference)")
        bad = tab[~tab.meets_floors]
        if len(bad):
            print(f"\n{len(bad)} cell(s) cannot meet both floors "
                  f"(>= {min_windows} windows AND >= {min_days} days) — report them as such:")
            print(bad[["etype", "gran", "n_windows", "evaluated"]].to_string(index=False))
    return tab


def feasible(etype: str, gran: str, history_minutes: float, lead_minutes: float,
             max_context: int = 2048, max_horizon: int = 64, min_context: int = 16):
    """Can this granularity represent this history and this lead time within model limits?

    Returns (ok, context_steps, horizon_steps, reason). Used by the fixed-lead-time
    comparison, where fine granularities simply cannot reach long lead times.
    """
    d = step_minutes(etype, gran)
    c = history_minutes / d
    h = lead_minutes / d
    if h < 1:
        return False, None, None, "lead time shorter than one step"
    if h != int(h):
        return False, None, None, "lead time not a whole number of steps"
    if h > max_horizon:
        return False, None, None, f"{int(h)} steps > horizon cap {max_horizon}"
    if c < min_context:
        return False, None, None, (f"{c:.0f} steps < minimum context {min_context} "
                                   f"(TimesFM needs >= 32: it patches context in 32s)")
    if c > max_context:
        return False, None, None, f"{c:,.0f} steps > context cap {max_context}"
    return True, int(round(c)), int(h), ""


# ------------------------------------------------------------------------------- the models

def _torch():
    import torch
    return torch


class FM(ABC):
    name = "fm"
    max_context = 2048
    max_horizon = 64

    def configure(self, max_context: int, max_horizon: int):
        """Declare the largest context/horizon this run will ask for, before warmup."""
        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        return self

    @abstractmethod
    def predict(self, context, horizon): ...

    def warmup(self):
        return self

    def free(self):
        for a in ("pipe", "model", "module"):
            if getattr(self, a, None) is not None:
                setattr(self, a, None)
        if getattr(self, "_cache", None):
            self._cache.clear()
        gc.collect()
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass          # CPU-only or torch absent: nothing to release


class ChronosFM(FM):
    """chronos-forecasting >= 2.0 renamed predict_quantiles(context=...) -> (inputs=...).
    The series is passed POSITIONALLY, which is valid on both 1.x and 2.x."""
    name = "chronos"

    def __init__(self):
        self.pipe = None

    def warmup(self):
        torch = _torch()
        from chronos import BaseChronosPipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        try:
            self.pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=dev,
                                                            torch_dtype=dt)
        except TypeError:      # transformers >= 5 renamed torch_dtype -> dtype
            self.pipe = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=dev,
                                                            dtype=dt)
        return self

    def predict(self, context, horizon):
        torch = _torch()
        if self.pipe is None:
            self.warmup()
        ctx = torch.tensor(np.asarray(context, dtype=float), dtype=torch.float32)
        q, _ = self.pipe.predict_quantiles(ctx,                    # positional (the fix)
                                           prediction_length=horizon,
                                           quantile_levels=[0.1, 0.5, 0.9])
        return q[0, :, 1].float().cpu().numpy().astype(float)


class TimesFMFM(FM):
    """Handles BOTH pip APIs:
       timesfm >= 2.x : TimesFM_2p5_200M_torch + ForecastConfig  (2.5 checkpoints)
       timesfm 1.x    : TimesFm + TimesFmHparams/TimesFmCheckpoint (2.0 checkpoint)
    """
    name = "timesfm"

    def __init__(self):
        self.model = None
        self._api = None

    def warmup(self):
        torch = _torch()
        import timesfm
        self._h = int(self.max_horizon)
        self._ctx = (int(self.max_context) // 32) * 32
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):                 # modern API
            self._api = "2p5"
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
            try:
                self.model.compile(timesfm.ForecastConfig(
                    max_context=self._ctx, max_horizon=self._h,
                    normalize_inputs=True, use_continuous_quantile_head=True,
                    force_flip_invariance=True, infer_is_positive=True,
                    fix_quantile_crossing=True))
            except Exception:
                self.model.compile(timesfm.ForecastConfig(
                    max_context=self._ctx, max_horizon=self._h, normalize_inputs=True))
        elif hasattr(timesfm, "TimesFm"):                              # legacy API
            self._api = "1x"
            backend = "gpu" if torch.cuda.is_available() else "cpu"
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend, per_core_batch_size=16,
                    horizon_len=self._h, context_len=self._ctx,
                    input_patch_len=32, output_patch_len=128,
                    num_layers=50, model_dims=1280, use_positional_embedding=False),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=TIMESFM_REPO_LEGACY))
        else:
            raise ImportError(
                "Unrecognised timesfm API: neither TimesFM_2p5_200M_torch nor TimesFm "
                "found. Install:  pip install 'timesfm[torch]'  (>=2.x)  or "
                "pip install 'timesfm[torch]==1.3.0'  (legacy).")
        print(f"    timesfm API = {self._api}  (max_context={self._ctx}, max_horizon={self._h})")
        return self

    def predict(self, context, horizon):
        if self.model is None:
            self.warmup()
        x = np.asarray(context[-self._ctx:], dtype=float)
        if self._api == "2p5":
            pt, _ = self.model.forecast(horizon=max(int(horizon), 1), inputs=[x])
        else:
            pt, _ = self.model.forecast([x], freq=[0])
        return np.asarray(pt[0][:horizon], dtype=float)


class MoiraiFM(FM):
    """Samples num_samples paths, so it is seeded before every forward pass — without
    that, Moirai's metrics change between identical runs while the other two are stable."""
    name = "moirai"

    def __init__(self, num_samples=100, patch_size=32, max_cached=6):
        self.ns = num_samples
        self.ps = patch_size
        self.module = None
        self._cache = {}
        self.max_cached = max_cached

    def warmup(self):
        from uni2ts.model.moirai import MoiraiModule
        self.module = MoiraiModule.from_pretrained(MOIRAI_REPO)
        return self

    def _forecaster(self, horizon, ctx_len):
        torch = _torch()
        from uni2ts.model.moirai import MoiraiForecast
        key = (int(horizon), int(ctx_len))
        if key not in self._cache:
            # a sweep visits many (horizon, context) pairs; keep VRAM bounded on 6 GB cards
            if len(self._cache) >= self.max_cached:
                self._cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            m = MoiraiForecast(module=self.module, prediction_length=int(horizon),
                               context_length=int(ctx_len), patch_size=self.ps,
                               num_samples=self.ns, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._cache[key] = m.to(dev).eval()
        return self._cache[key]

    def predict(self, context, horizon):
        torch = _torch()
        if self.module is None:
            self.warmup()
        ctx = np.asarray(context, dtype=float)
        model = self._forecaster(horizon, len(ctx))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pt = torch.tensor(ctx, dtype=torch.float32, device=dev).reshape(1, -1, 1)
        obs = torch.ones_like(pt, dtype=torch.bool)
        pad = torch.zeros(1, pt.shape[1], dtype=torch.bool, device=dev)
        torch.manual_seed(RANDOM_SEED)              # determinism across runs
        with torch.no_grad():
            fc = model(past_target=pt, past_observed_target=obs, past_is_pad=pad)
        return np.median(fc[0].cpu().numpy(), axis=0)[:horizon].astype(float)


FM_CLASSES = {"chronos": ChronosFM, "timesfm": TimesFMFM, "moirai": MoiraiFM}


def make_models(which=("chronos", "timesfm", "moirai")):
    """Instantiate without loading weights. Weights load one model at a time in run()."""
    return {k: FM_CLASSES[k]() for k in which}


# --------------------------------------------------------------------------- the run loop

def gpu_mem() -> str:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    except Exception:
        pass
    return "CPU"


def run(models: dict, windows, extra: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score every window with every model, one model resident in VRAM at a time.

    Every per-window diagnostic is written to the output frame, so a later change of
    metric is a pandas operation rather than another GPU run.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k):
            return x

    max_ctx = max(w.context_steps for w in windows)
    max_hor = max(w.horizon_steps for w in windows)
    rows, first_tb = [], {}

    for name, model in models.items():
        try:
            model.configure(max_ctx, max_hor)
            if verbose:
                print(f"loading {name} (max_context={max_ctx}, max_horizon={max_hor}) ...",
                      flush=True)
            model.warmup()
        except Exception:
            print(f"  SKIPPED {name} — could not load:")
            traceback.print_exc()
            continue

        n_ok = 0
        for w in tqdm(windows, desc=name):
            t0 = time.perf_counter()
            err = ""
            try:
                p = np.asarray(model.predict(w.context, w.horizon_steps), dtype=float)
                if p.shape[0] < w.horizon_steps:
                    raise ValueError(f"forecast too short: {p.shape[0]} < {w.horizon_steps}")
                p = p[:w.horizon_steps]
                if not np.all(np.isfinite(p)):
                    raise ValueError("forecast contains NaN/inf")
                met = metrics(w.truth, p, w.scale, w.snaive_mae)
                n_ok += 1
            except Exception as e:
                met = {"MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan,
                       "MASE": np.nan, "relMAE": np.nan}
                err = f"{type(e).__name__}: {e}"
                first_tb.setdefault(name, traceback.format_exc())
            rows.append({**w.meta(), "gran_label": GRANULARITIES[w.gran]["label"],
                         "model": name, "infer_sec": time.perf_counter() - t0,
                         "error": err, **met, **(extra or {})})
        if verbose:
            print(f"  {name}: {n_ok}/{len(windows)} scored | {gpu_mem()} -> freeing")
        model.free()

    for name, tb in first_tb.items():
        print("=" * 78)
        print(f"FIRST ERROR for {name}:")
        print(tb)

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No results — every model failed to load. See tracebacks above.")
    nan_rate = res.groupby("model")["MASE"].apply(lambda s: float(s.isna().mean()))
    if verbose:
        print("\nNaN rate per model:")
        for k, v in nan_rate.items():
            print(f"  {k:8s} {v:6.1%}")
        print("\nAll windows scored." if (nan_rate == 0).all()
              else "\n!! Some windows failed — read the tracebacks before trusting tables.")
    return res


# ------------------------------------------------------------------------------ reporting

def clean(res: pd.DataFrame, drop_degenerate: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Drop unscored and (by default) degenerate windows, reporting what went."""
    out = res.dropna(subset=["MASE"])
    n_fail = len(res) - len(out)
    n_deg = 0
    if drop_degenerate and "degenerate" in out.columns:
        n_deg = int(out["degenerate"].sum())
        out = out[~out["degenerate"]]
    if verbose:
        print(f"{len(out):,}/{len(res):,} windows used"
              + (f" | {n_fail} failed" if n_fail else "")
              + (f" | {n_deg} degenerate excluded" if n_deg else ""))
    if out.empty:
        raise RuntimeError("Nothing left after cleaning — inspect the raw results.")
    return out


def summary_table(res: pd.DataFrame, value: str = "MASE",
                  index=("etype", "gran_label"), columns="model") -> pd.DataFrame:
    """Median, IQR and mean per cell — because the mean alone flips three of nine winners."""
    g = res.groupby(list(index) + [columns])[value]
    tab = pd.DataFrame({
        "median": g.median(),
        "iqr": g.quantile(0.75) - g.quantile(0.25),
        "mean": g.mean(),
        "n": g.size(),
    }).reset_index()
    return tab


def mean_rank(res: pd.DataFrame, value: str = "MASE",
              cell=("etype", "gran_label"), model_col: str = "model") -> pd.DataFrame:
    """Rank models within each cell on the median, then average the ranks.

    Rank aggregation is robust to the scale differences between cells, which a mean over
    raw MASE is not.
    """
    med = res.groupby(list(cell) + [model_col])[value].median().reset_index()
    med["rank"] = med.groupby(list(cell))[value].rank(method="average")
    out = (med.groupby(model_col)["rank"].agg(["mean", "size"])
              .rename(columns={"mean": "mean_rank", "size": "n_cells"})
              .sort_values("mean_rank"))
    return out


def winner_table(res: pd.DataFrame, value: str = "MASE",
                 cell=("etype", "gran_label"), model_col: str = "model") -> pd.DataFrame:
    """Best model per cell on median and on mean, flagging where the two disagree."""
    med = res.pivot_table(index=list(cell), columns=model_col, values=value, aggfunc="median")
    avg = res.pivot_table(index=list(cell), columns=model_col, values=value, aggfunc="mean")
    out = pd.DataFrame({"by_median": med.idxmin(axis=1), "by_mean": avg.idxmin(axis=1)})
    out["best_median"] = med.min(axis=1).round(3)
    out["disagrees"] = np.where(out.by_median != out.by_mean, "yes", "")
    return out


def save(res: pd.DataFrame, path: str, verbose: bool = True) -> str:
    if path.endswith(".parquet"):
        res.to_parquet(path, index=False)
    else:
        res.to_csv(path, index=False)
    if verbose:
        print(f"{len(res):,} rows -> {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    return path


__all__ = [n for n in dir() if not n.startswith("_")]


# ------------------------------------------------------------------------- self-check
# This file is a LIBRARY: the notebooks do `import refined_stage_common as C`.
# Running it directly does no forecasting — it runs an environment pre-flight instead,
# so that `python refined_stage_common.py` tells you whether the machine is ready.

def selfcheck() -> int:
    import importlib
    print("=" * 74)
    print("  refined pipeline — environment pre-flight")
    print("=" * 74)

    print("\n[1] packages")
    need = {"numpy": "core", "pandas": "core", "scipy": "stats", "matplotlib": "plots",
            "statsmodels": "regression in 3a / 3e / 3h",
            "sklearn": "leave-one-type-out validation in 3g",
            "datasets": "Monash download",
            "torch": "all three models", "chronos": "Chronos-Bolt",
            "timesfm": "TimesFM", "uni2ts": "Moirai", "tqdm": "progress bars"}
    missing = []
    for mod, why in need.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            print(f"    ok       {mod:14s} {str(v):12s} ({why})")
        except Exception as e:
            missing.append(mod)
            print(f"    MISSING  {mod:14s} {'':12s} ({why})  -> {type(e).__name__}")

    print("\n[2] gpu")
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"    CUDA ok: {p.name}, {p.total_memory/1e9:.1f} GB")
            print("    models are loaded one at a time and freed, so 6 GB is enough")
        else:
            print("    no CUDA visible — notebooks still run, just far slower")
    except Exception as e:
        print(f"    torch unavailable ({type(e).__name__}) — GPU notebooks cannot run")

    print("\n[3] metric sanity")
    t = np.array([1.0, 2.0, 3.0])
    assert metrics(t, t, 2.0)["MAE"] == 0.0
    assert metrics(t, t + 1, 2.0)["MASE"] == 0.5
    assert np.isnan(metrics(t, t + 1, float("nan"))["MASE"])
    assert metrics(t, t + 1, 2.0, 4.0)["relMAE"] == 0.25
    sq = np.tile([1.0, 5.0, 9.0, 5.0], 30)          # period 4, exactly repeating
    assert np.allclose(seasonal_naive(sq, 60, 6, 4), sq[60:66])
    assert seasonal_naive(sq, 3, 6, 4) is None      # not enough history -> refuses
    # a smooth signal with a long cycle and a slow drift — the shape of 1-minute wind,
    # where the one-step change is tiny and the one-cycle change is not
    rng = np.random.default_rng(0)
    n = 6000
    tt = np.arange(n)
    x = 10 * np.sin(2 * np.pi * tt / 1440) + np.cumsum(rng.normal(0, 0.05, n))
    s_cycle, _ = reference_scale(x, 1440, 480)
    s_one, _ = reference_scale(x, 1, 480)
    flat, why = reference_scale(np.zeros(1000), 24, 48)
    assert np.isfinite(s_cycle) and np.isfinite(s_one)
    assert np.isnan(flat)
    print("    MASE + relMAE arithmetic ok; seasonal-naive indexing ok")
    print(f"    seasonal scale at m=1440 -> {s_cycle:.4f}, at m=1 -> {s_one:.4f} "
          f"({s_cycle/s_one:.0f}x apart)")
    print("      the old code collapsed m to 1 for wind native, which is why its MASE "
          "read 19.077")
    print(f"    flat series correctly refused: {why[:58]}...")

    print("\n[4] configuration")
    rows = [{"etype": e, "gran": g, "step_minutes": step_minutes(e, g),
             "seasonal_m": seasonal_m(e, g),
             "one_cycle": human_duration(seasonal_m(e, g) * step_minutes(e, g))}
            for e in BASE for g in GRANULARITIES]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n    checkpoints:")
    for k, v in [("chronos", CHRONOS_MODEL), ("timesfm", TIMESFM_REPO),
                 ("moirai", MOIRAI_REPO)]:
        print(f"      {k:8s} {v}")

    print("\n[5] cached data")
    print(f"    {PANEL_CACHE}: "
          + (f"present ({os.path.getsize(PANEL_CACHE)/1e6:.0f} MB)"
             if os.path.exists(PANEL_CACHE)
             else "not built yet — refined_stage2_baseline creates it on first run"))

    print("\n" + "=" * 74)
    if missing:
        PIP = {"sklearn": "scikit-learn", "chronos": "chronos-forecasting"}
        print(f"  NOT READY — install: pip install "
              f"{' '.join(PIP.get(m, m) for m in missing)}")
        print("  (chronos is 'chronos-forecasting'; sklearn is 'scikit-learn')")
    else:
        print("  READY — open refined_stage2_baseline.ipynb and run it top to bottom.")
        print("  This file is imported by the notebooks; there is nothing to run here.")
    print("=" * 74)
    return 1 if missing else 0


if __name__ == "__main__":
    # No sys.exit by default: under the VS Code debugger a non-zero exit is reported as
    # "Exception has occurred: SystemExit", which looks like a crash when it is just the
    # NOT READY status. Pass --strict for a real exit code (CI, shell scripting).
    import sys
    _code = selfcheck()
    if "--strict" in sys.argv:
        sys.exit(_code)
