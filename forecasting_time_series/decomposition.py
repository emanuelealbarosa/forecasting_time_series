"""Detrending and deseasonalising with MSTL (STL with several seasonal periods).

The series is split into trend + one seasonal component per period + remainder; the
forecasting method then works on the remainder only. To turn a forecast of the remainder
back into a forecast of the series, the trend and the seasonal components are extended
over the forecast period and added back:
    seasonal : the last full cycle of each seasonal component is repeated;
    trend    : extended according to `trend`:
        'flat'   - the last trend value is held;
        'drift'  - the average slope over the last `trend_window` observations continues;
        'linear' - a straight line fitted to the last `trend_window` trend values continues;
        'damped' - like 'drift', but the slope shrinks by the factor `damping` at every step.

Settings (a dict; only 'periods' is required). See docs/decomposition.md for a full guide.
    periods        : seasonal periods in observations, e.g. (96, 672) for daily and weekly with 15-minute data
    window         : fit on the last `window` observations only (None → all); earlier rows are dropped.
                     Must be more than two cycles of the longest period.
    seasonal_windows : length of the seasonal smoother for each period, in cycles of that period
                     (odd, >= 3; None → statsmodels default 7 + 4·i, i.e. (11, 15) for two periods)
    trend_smoother : length of the trend smoother, in observations (odd, > the longest period;
                     None → statsmodels default, about 1.5·period / (1 - 1.5 / seasonal window))
    seasonal_jump, trend_jump, low_pass_jump : STL "jumps" - compute each loess smoother only at every
                     n-th point and interpolate in between (1 = exact). The trend and low-pass smoothers are
                     slow and very smooth, so large jumps there cost almost nothing in accuracy.
    robust         : down-weight outliers (slower)
    trend, trend_window, damping : see above (trend_window defaults to the longest period)

A fitted decomposition depends only on the data it is fitted on and on these settings, so
it is cached: the many method combinations evaluated on the same validation split reuse it.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import OrderedDict

import numpy as np

DEFAULTS = {'method': 'mstl', 'window': None, 'seasonal_jump': 1, 'trend_jump': 20, 'low_pass_jump': 20,
            'seasonal_windows': None, 'trend_smoother': None, 'robust': False, 'trend': 'drift',
            'trend_window': None, 'damping': 0.99}
TREND_TYPES = ('flat', 'drift', 'linear', 'damped')
CACHE_SIZE = 256

_cache: OrderedDict = OrderedDict()


def normalise_spec(spec: dict) -> dict:
    """Validate a settings dict, fill in defaults and drop settings that do not apply."""
    if not isinstance(spec, dict):
        raise ValueError(f'a decomposition must be None or a dict of MSTL settings; got {spec!r}')
    unknown = set(spec) - set(DEFAULTS) - {'periods'}
    if unknown:
        raise ValueError(f'unknown MSTL settings: {sorted(unknown)}; allowed: {sorted(set(DEFAULTS) | {"periods"})}')
    s = {**DEFAULTS, **spec}
    if s['method'] != 'mstl':
        raise ValueError(f"only method 'mstl' is available; got {s['method']!r}")
    periods = s.get('periods')
    if isinstance(periods, int):
        periods = (periods,)
    if not periods or not all(_positive_int(p) and p > 1 for p in periods):
        raise ValueError(f'periods must be whole numbers above 1, e.g. (96, 672); got {periods!r}')
    s['periods'] = tuple(sorted(periods))
    longest = max(s['periods'])
    # statsmodels silently drops any period that is not shorter than half the series
    if s['window'] is not None and not (_positive_int(s['window']) and s['window'] > 2 * longest):
        raise ValueError(f'window must be None or more than two cycles of the longest period '
                         f'(at least {2 * longest + 1} observations); got {s["window"]!r}')
    if s['seasonal_windows'] is not None:
        windows = s['seasonal_windows']
        windows = (windows,) if isinstance(windows, int) else tuple(windows)
        if len(windows) != len(s['periods']) or not all(_odd_at_least(w, 3) for w in windows):
            raise ValueError(f'seasonal_windows must give one odd number >= 3 per period '
                             f'({len(s["periods"])} values, in the order of the sorted periods); got {windows!r}')
        s['seasonal_windows'] = windows
    if s['trend_smoother'] is not None and not (_odd_at_least(s['trend_smoother'], longest + 1)):
        raise ValueError(f'trend_smoother must be None or an odd number above the longest period '
                         f'(at least {longest + 1 if longest % 2 == 0 else longest + 2}); got {s["trend_smoother"]!r}')
    for jump in ('seasonal_jump', 'trend_jump', 'low_pass_jump'):
        if not _positive_int(s[jump]):
            raise ValueError(f'{jump} must be a whole number of at least 1; got {s[jump]!r}')
    if s['trend'] not in TREND_TYPES:
        raise ValueError(f'trend must be one of {TREND_TYPES}; got {s["trend"]!r}')
    if s['trend'] in ('drift', 'linear', 'damped'):
        s['trend_window'] = s['trend_window'] or max(s['periods'])
        if not (_positive_int(s['trend_window']) and s['trend_window'] >= 2):
            raise ValueError(f'trend_window must be a whole number of at least 2; got {s["trend_window"]!r}')
    else:
        s['trend_window'] = None
    if s['trend'] == 'damped':
        if not (isinstance(s['damping'], (int, float)) and 0 < s['damping'] <= 1):
            raise ValueError(f'damping must be above 0 and at most 1; got {s["damping"]!r}')
    else:
        s['damping'] = None
    return s


def label(spec: dict) -> str:
    s = normalise_spec(spec)
    parts = [f'mstl {"/".join(str(p) for p in s["periods"])}', f'window {s["window"] or "all"}']
    trend = s['trend']
    if s['trend_window']:
        trend += f' {s["trend_window"]}'
    if s['damping'] is not None:
        trend += f' x{s["damping"]:g}'
    parts.append(trend)
    if s['seasonal_windows'] is not None:
        parts.append('seasonal smoothers ' + '/'.join(str(w) for w in s['seasonal_windows']))
    if s['trend_smoother'] is not None:
        parts.append(f'trend smoother {s["trend_smoother"]}')
    jumps = (s['seasonal_jump'], s['trend_jump'], s['low_pass_jump'])
    if jumps != (DEFAULTS['seasonal_jump'], DEFAULTS['trend_jump'], DEFAULTS['low_pass_jump']):
        parts.append('jumps ' + '/'.join(str(j) for j in jumps))
    if s['robust']:
        parts.append('robust')
    return ', '.join(parts)


def minimum_rows(spec: dict) -> int:
    """Training observations needed: more than two full cycles of the longest period."""
    return 2 * max(normalise_spec(spec)['periods']) + 1


def mstl_options(periods, window=(None,), trend=('drift',), trend_window=(None,), damping=(0.99,),
                 seasonal_windows=(None,), trend_smoother=(None,), seasonal_jump=(1,), trend_jump=(20,),
                 low_pass_jump=(20,), robust=(False,)) -> list[dict]:
    """Every combination of the given MSTL settings, as a list ready for `decomposition_range`.
    Each argument is one value or a list of values to try; `periods` is fixed. For seasonal_windows,
    one option is a tuple with one value per period, e.g. (11, 15), so several options are a list of
    tuples, e.g. [(11, 15), (7, 5)]. Settings that do not apply (e.g. trend_window for a flat trend)
    are ignored, so no duplicates are produced."""
    def as_tuple(value):
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)

    if seasonal_windows is None or (isinstance(seasonal_windows, tuple) and
                                    all(isinstance(w, int) for w in seasonal_windows)):
        seasonal_windows = [seasonal_windows]  # a single option given directly
    names = ('window', 'trend', 'trend_window', 'damping', 'seasonal_windows', 'trend_smoother',
             'seasonal_jump', 'trend_jump', 'low_pass_jump', 'robust')
    values = (window, trend, trend_window, damping, seasonal_windows, trend_smoother,
              seasonal_jump, trend_jump, low_pass_jump, robust)
    options, seen = [], set()
    for combination in itertools.product(*(as_tuple(v) for v in values)):
        spec = normalise_spec({'periods': tuple(periods), **dict(zip(names, combination))})
        key = repr(sorted(spec.items()))
        if key not in seen:
            seen.add(key)
            options.append(spec)
    return options


class MSTLDecomposition:
    """Processing step: replaces the series by its MSTL remainder; adds the extended
    trend and seasonal components back onto forecasts."""

    def __init__(self, spec: dict):
        self.spec = normalise_spec(spec)
        self.label = label(self.spec)
        self.trend = None
        self.seasonal = None

    def fit_transform(self, y):
        y = np.asarray(y, dtype=float)
        if self.spec['window'] is not None:
            y = y[-self.spec['window']:]
        if len(y) < minimum_rows(self.spec):
            raise ValueError(f'too few observations for MSTL: {len(y)}, need at least '
                             f'{minimum_rows(self.spec)} (more than two cycles of the longest period)')
        if np.isnan(y).any():
            raise ValueError('MSTL cannot handle missing values')
        self.trend, self.seasonal, remainder = _fit_cached(y, self.spec)
        return remainder.copy()

    def inverse(self, remainder):
        """`remainder` continues the series right after the training data."""
        remainder = np.asarray(remainder, dtype=float)
        steps = np.arange(1, len(remainder) + 1)
        return remainder + self._trend_ahead(steps) + self._seasonal_ahead(steps)

    def _seasonal_ahead(self, steps):
        total = np.zeros(len(steps))
        n = len(self.trend)
        for column, period in enumerate(self.spec['periods']):
            # value one full cycle earlier: position n - period + ((h - 1) mod period)
            total += self.seasonal[n - period + (steps - 1) % period, column]
        return total

    def _trend_ahead(self, steps):
        last = self.trend[-1]
        kind = self.spec['trend']
        if kind == 'flat':
            return np.full(len(steps), last)
        k = min(self.spec['trend_window'], len(self.trend) - 1)
        recent = self.trend[-(k + 1):]
        if kind == 'linear':
            slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)
            return intercept + slope * (len(recent) - 1 + steps)
        slope = (recent[-1] - recent[0]) / k
        if kind == 'drift':
            return last + slope * steps
        phi = self.spec['damping']  # damped: slope * (phi + phi^2 + ... + phi^h)
        return last + slope * np.cumsum(phi ** steps)


FIT_SETTINGS = ('periods', 'seasonal_jump', 'trend_jump', 'low_pass_jump', 'robust', 'seasonal_windows',
                'trend_smoother')


def _fit_cached(y, spec):
    # the trend extension does not affect the fit, so e.g. 'flat' and 'drift' share one fit
    key = (hashlib.sha1(y.tobytes()).hexdigest(), tuple(spec[name] for name in FIT_SETTINGS))
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    fitted = _fit(y, spec)
    _cache[key] = fitted
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return fitted


def _fit(y, spec):
    from statsmodels.tsa.seasonal import MSTL  # imported here: only needed when MSTL is used

    stl_kwargs = {'seasonal_jump': spec['seasonal_jump'], 'trend_jump': spec['trend_jump'],
                  'low_pass_jump': spec['low_pass_jump'], 'robust': spec['robust']}
    if spec['trend_smoother'] is not None:
        stl_kwargs['trend'] = spec['trend_smoother']  # used by every STL pass, hence > the longest period
    result = MSTL(y, periods=spec['periods'], windows=spec['seasonal_windows'], stl_kwargs=stl_kwargs).fit()
    trend = np.asarray(result.trend, dtype=float)
    seasonal = np.asarray(result.seasonal, dtype=float).reshape(len(y), -1)
    remainder = np.asarray(result.resid, dtype=float)
    return trend, seasonal, remainder


def _positive_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 1


def _odd_at_least(value, minimum) -> bool:
    return _positive_int(value) and value % 2 == 1 and value >= minimum
