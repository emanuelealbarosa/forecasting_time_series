"""Building experiments for the daily methods (LEAR, MLP, LSTM, RF) from a description of their inputs.

Every daily method reads the same kinds of inputs (forecasting_time_series/methods/daily_inputs.py):
  - the outcome's own profile on the lag days (lag_days, e.g. (1, 2, 3, 7): days d-1, d-2, d-3, d-7);
  - known_inputs: variables known in advance (calendar dummies, day-ahead forecasts), over day d;
  - lagged_inputs: variables known only after the fact (actual load, another market's price), through
    their profiles on the days lagged_input_days back (never their day-d values).
daily_experiment turns that description into a ForecastExperiment (exog_variables, exog_lag_days,
steps_per_day, training windows in weeks), so a method's own grid is all that is left to give:

    experiment = daily_experiment(train, 'RF', steps_per_day=96,
                                  known_inputs=calendar + forecasts, lagged_inputs=actuals,
                                  window_weeks=(24,),
                                  method_ranges={'n_estimators_range': [100], 'min_samples_leaf_range': [5]},
                                  outcome_variable='price', time_variable='date')

The other settings of ForecastExperiment (metric, validation splits, ensembles ...) are passed through.
"""
from __future__ import annotations

from .experiment import ForecastExperiment

DAILY_METHODS = ('LEAR', 'MLP', 'LSTM', 'RF')


def _tuple(values):
    return tuple(values) if isinstance(values, (list, tuple, range)) else (values,)


def daily_experiment(train, method: str, steps_per_day: int, *,
                     known_inputs=(), lagged_inputs=(), lagged_input_days=(2, 7), exog_lag_days=None,
                     lag_days=(1, 2, 3, 7), lag_resolution: str = 'hourly', minimum_calibration_days: int = 28,
                     window_weeks=(None,),
                     transform=(None,), decomposition=(None,), differencing=(0,), normalisation=(None,),
                     method_ranges: dict | None = None, **experiment_settings) -> ForecastExperiment:
    """A ForecastExperiment of a daily method on `train`.

    known_inputs: columns known in advance, used over day d (e.g. calendar dummies, day-ahead forecasts).
    lagged_inputs: columns known only after the fact, used through their profiles on lagged_input_days.
    exog_lag_days: further {column: lag days} (e.g. another zone's price on d-1: {'HU': (1,)}); these
        columns come before the known inputs among the exogenous variables.
    lag_days, lag_resolution ('hourly': 24 averages per day, 'full': every step), minimum_calibration_days:
        the method's input settings (see methods/daily_inputs.py).
    window_weeks: the training windows searched, in weeks (None = expanding window).
    transform, decomposition, differencing, normalisation: the processing settings searched, a list or
        tuple of options each, or a single option (see _options and processing.py).
    method_ranges: the method's own settings and grids, e.g. {'penalty_range': ['aic']} for LEAR or
        {'hidden_layers_range': [(64, 32)], 'max_epochs': 500, ...} for the MLP (see experiment.METHODS).
    experiment_settings: passed to ForecastExperiment (outcome_variable, time_variable, metric,
        validation_maximum_number_of_splits, ensembles, ensemble_top_n, ...). forecasting_period defaults
        to steps_per_day (one whole day)."""
    if method not in DAILY_METHODS:
        raise ValueError(f'daily_experiment builds {DAILY_METHODS}; got {method!r}')
    extra_lags = {column: _tuple(lags) for column, lags in (exog_lag_days or {}).items()}
    lags = {**extra_lags, **{column: _tuple(lagged_input_days) for column in lagged_inputs}}
    exog_variables = list(dict.fromkeys([*extra_lags, *known_inputs, *lagged_inputs]))
    if 'differencing_range' in experiment_settings or 'transform_range' in experiment_settings:
        raise ValueError('give the processing options as transform, decomposition, differencing, normalisation')
    experiment_settings.setdefault('forecasting_period', steps_per_day)
    return ForecastExperiment(train, method=method,
                              exog_variables=exog_variables,
                              method_parameter_ranges={'steps_per_day': steps_per_day,
                                                       'minimum_calibration_days': minimum_calibration_days,
                                                       'lag_days': _tuple(lag_days),
                                                       'lag_resolution': lag_resolution,
                                                       'exog_lag_days': lags,
                                                       **(method_ranges or {})},
                              transform_range=_options(transform),
                              decomposition_range=list(_options(decomposition)),
                              differencing_range=_options(differencing),
                              normalisation_range=_options(normalisation),
                              training_window_range=tuple(None if weeks is None else weeks * 7 * steps_per_day
                                                          for weeks in _tuple(window_weeks)),
                              **experiment_settings)


def _options(values) -> tuple:
    """The options searched: a list or tuple is several options, anything else (None, a string, a number,
    a settings dict) is one. An option that is itself a tuple - the transform ('power', 0.5), sequential
    differencing (1, 96) - is given inside a list or tuple: transform=[('power', 0.5)], differencing=[(1, 96)]."""
    return tuple(values) if isinstance(values, (list, tuple)) else (values,)
