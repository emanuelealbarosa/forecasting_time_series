"""Inputs shared by the methods that forecast a whole day at once from the days before it (LEAR, MLP).

Day d (the day forecast) is described by:
  - the full profile of the outcome on each lag day (e.g. d-1, d-2, d-3 and d-7),
  - the full profiles of lagged exogenous variables on their lag days (exog_lag_days) - variables
    known only up to the previous day, such as another zone's prices or the actual load,
  - the exogenous variables known in advance (not in exog_lag_days, or listed with lag 0) over day d,
    such as day-of-week dummies and load forecasts.
build() turns training data (whole, consecutive days) into these inputs for every calibration day
with all lags available, and for day d; each method then arranges them as it needs: LEAR takes the
known-in-advance inputs of one step per model, the MLP the whole day's profile in one network.

Parameters shared by those methods:
  steps_per_day            - observations per day (96 for 15-minute data)
  lag_days                 - the past days whose outcome profiles are inputs, e.g. (1, 2, 3, 7)
  lag_resolution           - 'full': every step of each lag day; 'hourly': its 24 hourly averages
  minimum_calibration_days - fewest days with all lags available that a fit may use
  exog_lag_days            - {column: lag days} for exogenous variables used through their past,
                             e.g. {'HU': (1, 2, 3, 7)}. Lag 0 also uses the value over day d, so the
                             column must then be known in advance (e.g. a load forecast).
  exog_lag_transform       - 'asinh' (median/MAD scaling, fitted on the training data) or None:
                             applied to the lagged exogenous variables
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

LAG_RESOLUTIONS = ('full', 'hourly')
EXOG_LAG_TRANSFORMS = ('asinh', None)
MAD_TO_SD = 1.4826


def largest_lag(method_parameters):
    """The furthest day back any input needs (outcome or exogenous)."""
    exog_lags = [lag for lags in (method_parameters.get('exog_lag_days') or {}).values() for lag in lags]
    return max(list(method_parameters['lag_days']) + exog_lags)


def required_history(method_parameters):
    """Training observations needed: the largest lag plus the minimum calibration days, in steps."""
    days = largest_lag(method_parameters) + method_parameters['minimum_calibration_days']
    return days * method_parameters['steps_per_day']


def known_in_advance(exog_variables, exog_lag_days):
    """Exogenous variables whose values over the forecast day are needed: those not listed in
    exog_lag_days, and those listed with lag 0."""
    exog_lag_days = exog_lag_days or {}
    return [c for c in exog_variables if c not in exog_lag_days or 0 in exog_lag_days[c]]


def check_parameters(p):
    """Raise ValueError for malformed shared parameters."""
    steps = p.get('steps_per_day')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f'steps_per_day must be a whole number of at least 1, got {steps!r}')
    lags = p.get('lag_days')
    if isinstance(lags, int) or not lags or \
            not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in lags):
        raise ValueError(f'lag_days must be a tuple of whole numbers of days of at least 1, e.g. (1, 2, 3, 7); '
                         f'got {lags!r}')
    if p.get('lag_resolution') not in LAG_RESOLUTIONS:
        raise ValueError(f'lag_resolution must be one of {LAG_RESOLUTIONS}, got {p.get("lag_resolution")!r}')
    if p['lag_resolution'] == 'hourly' and steps % 24:
        raise ValueError(f"lag_resolution 'hourly' needs steps_per_day divisible by 24, got {steps}")
    minimum = p.get('minimum_calibration_days')
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError(f'minimum_calibration_days must be a whole number of at least 2, got {minimum!r}')
    exog_lags = p.get('exog_lag_days') or {}
    if not isinstance(exog_lags, dict):
        raise ValueError(f"exog_lag_days must be a dict such as {{'HU': (1, 2, 3, 7)}}, got {exog_lags!r}")
    for column, lags in exog_lags.items():
        if isinstance(lags, int) or not lags or \
                not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in lags):
            raise ValueError(f'exog_lag_days[{column!r}] must be a tuple of whole numbers of days of at least 0, '
                             f'e.g. (1, 7); got {lags!r}')
    if p.get('exog_lag_transform', 'asinh') not in EXOG_LAG_TRANSFORMS:
        raise ValueError(f'exog_lag_transform must be one of {EXOG_LAG_TRANSFORMS}, '
                         f'got {p.get("exog_lag_transform")!r}')


@dataclass
class DailyInputs:
    """The inputs of every calibration day (rows, oldest first) and of the forecast day d.
      targets         - positions, among the whole training days, of the calibration days
      y               - (calibration days × steps) the outcome on those days
      lags            - (calibration days × lag inputs) the lag profiles of the outcome first (the first
                        outcome_inputs columns), then those of the lagged exogenous variables
      lags_next       - (lag inputs,) the same for day d
      same_day        - (calibration days × steps × known columns) the known-in-advance variables
      same_day_next   - (horizon × known columns) the same over day d
      same_day_columns - names of the known-in-advance columns
      small           - (calibration days × steps × lag profiles) each lag profile at the same step
                        (for a small model of the noise variance)"""
    targets: np.ndarray
    y: np.ndarray
    lags: np.ndarray
    lags_next: np.ndarray
    outcome_inputs: int
    same_day: np.ndarray
    same_day_next: np.ndarray
    same_day_columns: list
    small: np.ndarray


class DailyInputBuilder:

    def __init__(self, method_parameters, name='this method'):
        check_parameters(method_parameters)
        self.name = name  # for error messages, e.g. 'LEAR'
        self.steps_per_day = method_parameters['steps_per_day']
        self.lag_days = tuple(sorted(set(method_parameters['lag_days'])))
        self.lag_resolution = method_parameters['lag_resolution']
        self.minimum_calibration_days = method_parameters['minimum_calibration_days']
        self.exog_lag_days = {column: tuple(sorted(set(lags)))
                              for column, lags in (method_parameters.get('exog_lag_days') or {}).items()}
        self.exog_lag_transform = method_parameters.get('exog_lag_transform', 'asinh')
        self.largest_lag = largest_lag(method_parameters)

    # main
    def build(self, data, outcome_variable, time_variable, horizon, exog_variables=(), future_exog=None):
        """The inputs of `data` (whole days) and of the next day's first `horizon` steps. `future_exog`
        holds the exogenous variables known in advance over those steps; lagged-only variables
        (exog_lag_days without lag 0) are not used from it."""
        if horizon > self.steps_per_day:
            raise ValueError(f'{self.name} forecasts one day at a time: horizon {horizon} is longer than '
                             f'steps_per_day={self.steps_per_day}')
        exog_variables = list(exog_variables or [])
        unknown = [c for c in self.exog_lag_days if c not in exog_variables]
        if unknown:
            raise ValueError(f'exog_lag_days lists {unknown}, which are not among the exogenous variables')
        same_day = known_in_advance(exog_variables, self.exog_lag_days)   # used over day d
        lagged = list(self.exog_lag_days)                                 # used through their lag days
        columns = [outcome_variable] + same_day + [c for c in lagged if c not in same_day]
        values = self.whole_days(data, columns, time_variable)            # (days, steps, columns)
        position = {column: i for i, column in enumerate(columns)}
        days = values[:, :, 0]
        same_day_past = values[:, :, [position[c] for c in same_day]]
        same_day_next = self.exog_of_forecast_day(future_exog, same_day, horizon)  # (horizon, columns)
        # lagged exogenous variables on the outcome's scale: asinh with median/MAD of the training data
        scaled = {c: self.scale_exog(values[:, :, position[c]]) for c in lagged}
        for i, c in enumerate(same_day):
            if c in scaled:  # a lag-0 column: its day-d values on the same scale as its lags
                same_day_past[:, :, i] = scaled[c][0]
                same_day_next[:, i] = scaled[c][1](same_day_next[:, i])
        lagged_days = {c: scaled[c][0] for c in lagged}

        targets = np.arange(self.largest_lag, len(days))  # calibration days with every lag available
        if len(targets) < self.minimum_calibration_days:
            raise ValueError(f'{self.name} has {len(targets)} calibration days with all lags available; '
                             f'minimum_calibration_days={self.minimum_calibration_days}')
        profiles = [(days, lag) for lag in self.lag_days] + \
                   [(lagged_days[c], lag) for c in lagged for lag in self.exog_lag_days[c] if lag >= 1]
        lags = np.hstack([self.lag_profile(series[targets - lag]) for series, lag in profiles])
        lags_next = np.hstack([self.lag_profile(series[len(days) - lag]) for series, lag in profiles])
        small = np.stack([series[targets - lag] for series, lag in profiles], axis=-1)
        return DailyInputs(targets=targets, y=days[targets], lags=lags, lags_next=lags_next,
                           outcome_inputs=len(self.lag_days) * self.lag_profile(days[:1]).shape[1],
                           same_day=same_day_past[targets], same_day_next=same_day_next,
                           same_day_columns=same_day, small=small)

    def scale_exog(self, day_values):
        """(scaled days × steps, function scaling new values the same way): asinh with the median and
        MAD of the training values, or unchanged."""
        if self.exog_lag_transform is None:
            return day_values, lambda x: np.asarray(x, dtype=float)
        location = np.median(day_values)
        scale = np.median(np.abs(day_values - location)) * MAD_TO_SD
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        function = lambda x: np.arcsinh((np.asarray(x, dtype=float) - location) / scale)
        return function(day_values), function

    # subsidiary
    def whole_days(self, data, columns, time_variable):
        """`columns` of `data` as a (days × steps_per_day × columns) array, oldest day first.
        A leading partial day is dropped; the days must be complete and consecutive."""
        values = data[columns].to_numpy(dtype=float)
        n = self.steps_per_day
        if time_variable is not None and time_variable in data.columns:
            day = pd.to_datetime(data[time_variable]).dt.normalize()
            counts = day.map(day.value_counts())
            if counts.iloc[-1] != n or not (day.iloc[-n:] == day.iloc[-1]).all():
                raise ValueError(f'{self.name} needs the training data to end with a complete day of {n} steps')
            start = int(np.argmax((counts == n).to_numpy()))  # drop a leading partial day
            if (counts.iloc[start:] != n).any():
                raise ValueError(f'{self.name} needs whole days: some days have fewer or more than {n} steps')
            values, day = values[start:], day.iloc[start:]
            day_dates = day.iloc[::n].reset_index(drop=True)
            gaps = day_dates.diff().dt.days.iloc[1:]
            if (gaps != 1).any():
                missing_after = day_dates.iloc[int(np.argmax((gaps != 1).to_numpy()))]
                raise ValueError(f'{self.name} needs consecutive days; a day is missing after '
                                 f'{missing_after:%Y-%m-%d}')
        else:  # no dates: the data are taken to be whole days ending at the end of a day
            values = values[len(values) % n:]
        if np.isnan(values).any():
            raise ValueError(f'{self.name} cannot use missing values in the training data')
        return values.reshape(-1, n, len(columns))

    def exog_of_forecast_day(self, future_exog, exog_variables, horizon):
        if not exog_variables:
            return np.zeros((horizon, 0))
        if future_exog is None or len(future_exog) < horizon:
            raise ValueError(f'{self.name} needs the exogenous variables {exog_variables} over the {horizon} '
                             f'steps forecast (future_exog)')
        values = future_exog[exog_variables].to_numpy(dtype=float)[:horizon]
        if np.isnan(values).any():
            raise ValueError('future_exog contains missing values')
        return values

    def whole_day_design(self, d: DailyInputs):
        """(calibration days × inputs, day d's inputs) for a model of the whole day at once (MLP, random
        forest): the lag profiles, then each known-in-advance variable's day-d profile (hourly averages
        with lag_resolution 'hourly')."""
        same_day = self.lag_profile(np.transpose(d.same_day, (0, 2, 1)))     # (days, columns, steps)
        same_day_next = self.lag_profile(d.same_day_next.T)                  # (columns, steps)
        X = np.hstack([d.lags, same_day.reshape(len(d.targets), -1)])
        x_next = np.concatenate([d.lags_next, same_day_next.ravel()])
        return X, x_next

    def lag_profile(self, day_values):
        """The inputs one day contributes: every step, or hourly averages."""
        if self.lag_resolution == 'hourly':
            return day_values.reshape(*day_values.shape[:-1], 24, -1).mean(axis=-1)
        return day_values
