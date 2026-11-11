"""Day-ahead time series forecasting: a grid search over methods (Naive, LEAR, MLP, LSTM, random forest) and
processing settings, ensembles, block-wise selection and a results log.

    import forecasting_time_series as fts
    experiment = fts.ForecastExperiment(data, method='LEAR', ...).run()

    experiment = fts.daily_experiment(train, 'RF', steps_per_day=96, known_inputs=..., ...)
    result = fts.evaluate({'RF': lambda train: fts.daily_experiment(train, 'RF', ...)}, data,
                          test_days=28, steps_per_day=96, save_to='results')
    combined = fts.combine_runs('results', [run_id_1, run_id_2], timezone=...)  # runs saved separately

The names below are the public interface; the submodules stay importable for everything else.
Importing the package does not import torch, dash or plotly: the MLP and LSTM load torch when they are
used, and the dashboard is its own module (from forecasting_time_series import dashboard)."""
from importlib.metadata import PackageNotFoundError, version as _version

from . import metrics
from .block_selection import BlockSelection
from .builders import daily_experiment
from .calendar_features import calendar_features, future_dates
from .decomposition import mstl_options
from .ensemble import CombinedEnsembles, combine
from .evaluation import (SIX_HOUR_BLOCKS, EvaluationResult, evaluate, split_train_test, forecast_blockwise,
                         forecast_candidates, forecast_combined)
from .experiment import METHODS, ForecastExperiment
from .results_log import load_forecasts, load_runs, load_validation_forecasts, save_run
from .saved_runs import combine_runs

try:
    __version__ = _version('forecasting-time-series')
except PackageNotFoundError:  # used from a source folder without being installed
    __version__ = '0+unknown'

__all__ = ['BlockSelection', 'CombinedEnsembles', 'EvaluationResult', 'ForecastExperiment', 'METHODS',
           'SIX_HOUR_BLOCKS', 'calendar_features', 'combine', 'combine_runs', 'daily_experiment', 'evaluate', 'future_dates',
           'load_forecasts', 'load_runs', 'load_validation_forecasts', 'metrics', 'mstl_options', 'save_run',
           'split_train_test', 'forecast_blockwise', 'forecast_candidates', 'forecast_combined', '__version__']
