# The decomposition step (MSTL)

This guide explains how to detrend and deseasonalise a series inside the framework: what the
step does, how each setting changes the result, how to search over settings, and what to watch
out for. The code is in [`forecasting_time_series/decomposition.py`](../forecasting_time_series/decomposition.py).

---

## 1. What the step does

The decomposition splits the series into parts that add up exactly to it:

```
series = trend + seasonal(daily) + seasonal(weekly) + … + remainder
```

The **forecasting method only sees the remainder**: the part not explained by the trend and the
seasonal patterns. After the method has forecast the remainder, the step rebuilds a forecast of
the series by **extending the trend and the seasonal components over the forecast period** and
adding them back.

This lets a simple method concentrate on what is left, and it gives you explicit control over
how the trend is continued (held flat, continued, damped …).

## 2. Where it sits in the pipeline

Processing always runs in this order, and is undone in reverse on the forecasts:

```
transform  →  decomposition  →  differencing  →  normalisation  →  method
   ↑               ↑                 ↑               ↑                │
undo transform ← add back trend ← undo          ← de-normalise  ←  forecast
                 and seasonality   differencing
```

- **Transform first.** A `log`, `sqrt` or `asinh` transform before the decomposition turns
  patterns whose size grows with the level (multiplicative) into patterns of constant size
  (additive), which is what MSTL assumes.
- **Differencing after.** Applied to the remainder. Usually not needed once the trend has been
  removed, but it is still available in the grid.
- **Every step is fitted on the training data of each validation split only**, so nothing from
  the test window leaks into the decomposition.

The setting is `decomposition_range` in `ForecastExperiment`; its default is `(None,)`
(no decomposition).

## 3. How MSTL works

MSTL (Multiple Seasonal-Trend decomposition using Loess) extends STL, which handles one seasonal
period, to several periods by **taking turns** between them:

1. The periods are sorted, shortest first: e.g. 96 (daily), then 672 (weekly).
2. STL with period 96 is run; its seasonal part becomes the **daily** component.
3. STL with period 672 is run on the series **minus the daily component**; its seasonal part
   becomes the **weekly** component.
4. This is repeated (2 rounds), each component being re-estimated with the others removed.
5. The **trend** comes from the last STL pass (the one with the longest period); the remainder is
   what is left.

Because the daily shape is removed before the weekly pass, the components do not overlap:

- **daily** = the typical shape of a day, common to all days (night level, morning and evening
  peaks, midday solar dip);
- **weekly** = only what differs between days of the week on top of that (e.g. lower weekend
  prices).

Each STL pass uses **loess**: at each point, a small weighted regression over nearby values. The
lengths of these smoothers decide how quickly each component may change (section 6).

## 4. How forecasts are rebuilt

For a forecast `h` steps after the last training observation `n`:

**Seasonal components** – each repeats its own last full cycle:

```
S_p(n + h) = S_p(n − p + ((h − 1) mod p))
```

The daily component repeats the last day, the weekly component the last week; they are added up.

**Trend** – extended according to the `trend` setting, using the last `trend_window` (= k) values
of the fitted trend T:

| `trend`  | Rule | Behaviour |
|----------|------|-----------|
| `flat`   | T(n + h) = T(n) | holds the last level; ignores `trend_window` |
| `drift`  | T(n + h) = T(n) + b·h, with b = (T(n) − T(n − k)) / k | continues the average recent slope |
| `linear` | straight line fitted to the last k + 1 trend values, continued | like drift, but less sensitive to the two end points; starts from the fitted line, so it can jump slightly from T(n) |
| `damped` | T(n + h) = T(n) + b·(φ + φ² + … + φʰ), φ = `damping` | slope fades out; the total added never exceeds b·φ / (1 − φ) |

With 15-minute data the slope b is per quarter-hour, so φ applies per quarter-hour too:
φ = 0.99 halves the slope's effect after about 69 steps (≈ 17 hours); φ = 0.999 after about
693 steps (≈ 7 days).

## 5. Settings reference

A decomposition option is a dict. Only `periods` is required; everything else has a default.

| Setting | Default | Allowed values | What it changes |
|---------|---------|----------------|-----------------|
| `periods` | – (required) | whole numbers > 1, e.g. `(96, 672)` | Seasonal cycles to remove, in observations. One component per period. |
| `window` | `None` (all data) | `None`, or more than 2 × the longest period | **Which data** MSTL is fitted on: the last `window` observations. Earlier rows are dropped, also for the method. |
| `seasonal_windows` | `None` → `(11, 15)` for two periods (statsmodels: 7 + 4·i) | one odd number ≥ 3 per period, in the order of the sorted periods | **Memory of each seasonal pattern, in cycles**: how many days (daily) or weeks (weekly) each pattern is smoothed over. Small = adapts quickly but noisy. |
| `trend_smoother` | `None` → statsmodels default (≈ 1121 for a weekly period with seasonal window 15, i.e. ≈ 11.7 days of 15-minute data) | odd number > the longest period (≥ 673 for 672) | **Memory of the trend, in observations**: how many observations the trend is smoothed over. Smaller = follows changes faster but noisier. |
| `trend` | `'drift'` | `'flat'`, `'drift'`, `'linear'`, `'damped'` | **How the trend is extended** over the forecast period (section 4). |
| `trend_window` | longest period | whole number ≥ 2 | For `drift` / `linear` / `damped`: how many of the last trend values set the slope. Ignored for `flat`. |
| `damping` | `0.99` | 0 < φ ≤ 1 | For `damped`: how fast the slope fades, per step. Ignored otherwise. |
| `seasonal_jump` | `1` | whole number ≥ 1 | Speed/accuracy of the seasonal smoother (see 5.1). Keep at 1. |
| `trend_jump` | `20` | whole number ≥ 1 | Speed/accuracy of the trend smoother (see 5.1). |
| `low_pass_jump` | `20` | whole number ≥ 1 | Speed/accuracy of STL's internal low-pass filter (see 5.1). |
| `robust` | `False` | `True` / `False` | Down-weights outliers such as price spikes. Several times slower. |

Unknown or invalid settings are rejected when the experiment is created, with a message saying
what is allowed.

### 5.1 The "jumps"

A jump of n means a smoother is computed exactly at every n-th point only, with straight lines
in between. 1 = exact. Measured on 8 weeks of RO 15-minute prices:

| seasonal / trend / low-pass jump | Time per fit | Difference from exact (EUR/MWh) |
|---|---|---|
| 1 / 1 / 1 (exact) | 15.8 s | – |
| 5 / 5 / 5 | 3.0 s | seasonal 2.97 |
| **1 / 20 / 20 (default)** | **0.9 s** | seasonal 0.04, trend 0.01 |
| 1 / 50 / 50 | 0.4 s | seasonal 0.11 |

The seasonal smoother is cheap but works on short sub-series, so skipping points there loses
real information: keep `seasonal_jump=1`. The trend and low-pass smoothers are expensive but
very smooth, so large jumps there are almost free.

## 6. Four different "lengths" – which is which

They are easy to confuse. In order of how they act:

```
◄──────────────────────────── training data of one split ────────────────────────────►
                     ◄──────────────────── window ─────────────────────►
                     (MSTL fitted here; earlier rows are dropped)
                                                ◄─ trend_smoother ─►   weighted memory of the trend
                                     ◄──── seasonal_windows × period ──►  weighted memory of each pattern
                                                         ◄ trend_window ►  slope used to extend the trend
                                                                         │ forecast →
```

| Length | Unit | Decides | Affects the fit? |
|--------|------|---------|------------------|
| `window` | observations | which data exists for MSTL and the method | yes – all components |
| `seasonal_windows` | cycles of each period | how quickly each seasonal pattern may change | yes – seasonal components |
| `trend_smoother` | observations | how quickly the trend may change | yes – trend (and so the remainder) |
| `trend_window` | observations | which recent trend values set the extension slope | **no** – only the extension |

So the trend and the seasonal patterns **can** have different memories, but through the
smoothers, not through `window`. `window` is a hard cut-off shared by everything; the smoothers
are soft, weighted memories inside it.

Two consequences worth remembering:

- **The weekly pattern needs a long window to adapt.** With an 8-week window there are only 8
  values per weekday and quarter-hour; the default weekly smoother (15 weeks) is longer than
  that, so the weekly pattern is effectively **fixed** over the window. For a weekly pattern that
  adapts, use a smaller weekly smoother (e.g. `seasonal_windows=(11, 5)`) or a longer window.
- **The trend cannot be faster than about one cycle of the longest period.** `trend_smoother`
  is used by every STL pass and must exceed the longest period (> 672 = one week). For a trend
  extension that reacts faster, shorten `trend_window` instead.

## 7. Using it in an experiment

### One option, written directly

```python
ForecastExperiment(train, 'price', ...,
                   decomposition_range=[None, {'periods': (96, 672), 'window': 8 * 672, 'trend': 'flat'}])
```

Always consider keeping `None` among the options, so the grid can tell whether the decomposition
helps at all.

### Many options with `mstl_options`

`forecasting_time_series.decomposition.mstl_options` builds every combination of the values you give (one value
or a list of values per setting; `periods` is fixed) and removes duplicates, e.g. it does not
create several `flat` options that differ only by `trend_window`.

```python
import forecasting_time_series.decomposition as mstl

decomposition_range = [None] + mstl.mstl_options(
    periods=(96, 672),
    window=[4 * 672, 8 * 672],              # 4 or 8 weeks
    seasonal_windows=[(11, 15), (7, 5)],    # one tuple per option: (daily, weekly)
    trend_smoother=[None, 673],             # default (~12 days) or ~1 week
    trend=['flat', 'drift', 'damped'],
    trend_window=[96, 672],                 # slope over the last day or week
    damping=0.99,
)
```

Note that `seasonal_windows` options are **tuples** (one value per period), so several options
are a list of tuples. A single tuple is treated as one option.

The number of options multiplies quickly (the example gives 2 × 2 × 2 × (1 + 2 + 2) = 40), and
each one multiplies the size of the whole grid. Start small.

### Reading the results

The results table has a `decomposition` column with a label such as

```
mstl 96/672, window 5376, drift 672, seasonal smoothers 7/5, trend smoother 673
```

Defaults are left out of the label (e.g. the jumps appear only when changed). The settings
themselves are in `experiment.results[i].processing_parameters['decomposition']`.

## 8. History planning and failures

The grid search plans one shared set of validation splits for all combinations (see the
experiment's report). The decomposition adds two requirements per combination:

- **MSTL needs more than two cycles of the longest period** (≥ 1345 observations for 672) in the
  training data of every split. statsmodels would otherwise silently drop that period; the
  framework refuses instead.
- **Everything the method needs must fit inside the window.** With a window, only `window` rows
  remain after the decomposition, minus any rows removed by differencing. A combination whose
  method needs more (e.g. Naive length 4 × offset 672 = 2688 rows with a 4-week window of 2688
  rows plus differencing at lag 1) is marked as failed up front:

  ```
  needs more history than the decomposition window keeps (2689 rows needed, window 2688)
  ```

Failed combinations are listed by `experiment.failed_table()`.

## 9. Speed and caching

A fitted decomposition depends only on the data and on the **fit settings** (`periods`,
`seasonal_windows`, `trend_smoother`, the jumps, `robust`; with `window` reflected in the data).
It is therefore **cached** and reused:

- by every method combination evaluated on the same validation split;
- by options that differ only in how the trend is **extended** (`trend`, `trend_window`,
  `damping`) – they share one fit.

So the cost is roughly *number of splits × number of distinct fit settings × number of
transforms × time per fit*. With 28 splits, one fit setting, two transforms and the default
jumps on an 8-week window, that is about 28 × 2 × 0.9 s ≈ 1 minute. The cache keeps the last 256
fits per process; with `n_jobs > 1` each worker process has its own cache.

## 10. Practical guidance (15-minute electricity prices)

A sensible way to explore:

1. Start with `periods=(96, 672)`, `window=8 * 672`, default smoothers, and compare
   `trend=['flat', 'drift']` against `None`.
2. If the weekly pattern seems stale, try `seasonal_windows=[(11, 15), (11, 5)]`.
3. If the trend lags behind level changes, try `trend_smoother=673` and/or a shorter
   `trend_window` (e.g. 96 = one day) with `drift` or `damped`.
4. Try `robust=True` only once the rest is settled – it is slower, but helps when spikes distort
   the components.
5. Combine with `transform_range=(None, 'asinh')`: price spikes and negative prices make
   `asinh` a natural partner (`log` fails on negative prices).

Things to keep in mind:

- **Yearly seasonality cannot be estimated** from about one year of data; only daily and weekly
  patterns are realistic.
- **The end of the trend is the least reliable part** (the smoother only sees one side there),
  and it is exactly what the extension starts from. Prefer slopes measured over several steps
  (`trend_window`) to very short ones.
- **Overlap with seasonal methods.** Naive with offset 672 already models the weekly pattern;
  after deseasonalising, it mostly forecasts leftovers. On the RO data, MSTL with Naive scored
  about 52 RMSE against 45 for plain Naive with offset 672 – the decomposition is expected to pay
  off more with methods that model the remainder (e.g. ARIMA, RF).
- **No missing values** are allowed inside the window.

## 11. Code map

| Where | What |
|-------|------|
| `forecasting_time_series/decomposition.py` | settings and validation (`normalise_spec`), labels, `mstl_options`, the `MSTLDecomposition` step, trend and seasonal extension, caching |
| `forecasting_time_series/processing.py` | pipeline order (`PROCESSING_KEYS`), `build_decomposition`, `decomposition_limits` |
| `forecasting_time_series/grid_search.py` | history requirements per combination and the "window too short" check |
| `forecasting_time_series/experiment.py` | the `decomposition_range` setting and its validation |
