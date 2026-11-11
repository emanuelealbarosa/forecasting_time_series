from __future__ import annotations

import copy
import os
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from . import decomposition as mstl
from . import ensemble as ensemble_module
from . import forecast
from . import grid_builder
from . import grid_search
from . import method as method_registry
from . import metrics
from . import processing
from .ensemble import EnsembleResult
from .grid_search import SearchResult, SplitPlan
from .methods import daily_inputs

# Preprocessing options tried when neither the user nor the method sets them.
PROCESSING_DEFAULTS = {
    'transform_range': (None, 'sqrt'),
    'differencing_range': (0, 1),
    'normalisation_range': (None, 'minmax'),
    'decomposition_range': (None,),
}

# One entry per method:
#   supports_exog             - whether the method can use exogenous regressors
#   default_parameter_ranges  - grid ranges used unless overridden (partial overrides are merged)
#   default_processing_ranges - (optional) replaces entries of PROCESSING_DEFAULTS for this method
METHODS = {
    'Naive': {'supports_exog': False,
              'default_parameter_ranges': {
                  'predictors_maximum_length_range': [10],
                  'predictors_offset_range': [1],
                  'average_type_range': ['mean', 'median']},
              # averaging is linear, so normalising never changes Naive's forecasts
              'default_processing_ranges': {'normalisation_range': (None,)}},
    # LEAR (forecasting_time_series/methods/lear.py): one Lasso model per step of the day, forecasting a whole day at once
    # The input set is fixed by design (the Lasso selects among the inputs), so only the penalty is a
    # range; known-in-advance covariates such as day-of-week dummies come in through exog_variables.
    'LEAR': {'supports_exog': True,
             'default_parameter_ranges': {
                 'steps_per_day': 24,                  # 24 hourly, 96 for 15-minute data
                 'minimum_calibration_days': 14,       # fewest days a fit may use
                 'lag_days': (1, 2, 3, 7),             # full outcome profiles of days d-1, d-2, d-3, d-7
                 'lag_resolution': 'full',             # 'full' (every step) or 'hourly' (24 averages)
                 'exog_lag_days': {},                  # exogenous variables used through their past, e.g.
                                                       # {'HU': (1, 2, 3, 7)}; the others: at day d only
                 'exog_lag_transform': 'asinh',        # scaling of those lagged variables, or None
                 'lag_interactions': (),               # known-in-advance columns (e.g. weekday dummies)
                                                       # multiplying each outcome lag input
                 'penalty_range': ['aic']},            # searched: 'aic', 'bic', 'cv' or a fixed alpha
             # the original benchmark: asinh with median/MAD scaling and nothing else; the daily and
             # weekly patterns are handled by the per-step models, the lag days and the dummies
             'default_processing_ranges': {'transform_range': ('asinh',),
                                           'differencing_range': (0,),
                                           'normalisation_range': (None,),
                                           'decomposition_range': (None,)}},
    # MLP (forecasting_time_series/methods/mlp.py): one PyTorch network with an output per step of the day, on LEAR's
    # inputs (plus each known-in-advance variable's whole day-d profile); the network settings are ranges
    'MLP': {'supports_exog': True,
            'default_parameter_ranges': {
                'steps_per_day': 24,
                'minimum_calibration_days': 28,
                'lag_days': (1, 2, 3, 7),
                'lag_resolution': 'full',
                'exog_lag_days': {},
                'exog_lag_transform': 'asinh',
                'hidden_layers_range': [(256, 128)],  # neurons per hidden layer
                'dropout_range': [0.1],
                'weight_decay_range': [1e-3],         # AdamW's L2-like penalty
                'learning_rate_range': [1e-3],
                'activation': 'relu',
                'batch_size': 16,                     # calibration days per mini-batch
                'max_epochs': 500,
                'patience': 30,                       # early stopping: epochs without improvement
                'validation_days': 14,                # last calibration days held out for early stopping
                'refit': True,                        # then retrain on every day for the best epoch count
                'n_seeds': 3,                         # networks averaged
                'seed': 0},
            'default_processing_ranges': {'transform_range': ('asinh',),
                                          'differencing_range': (0,),
                                          'normalisation_range': (None,),
                                          'decomposition_range': (None,)}},
    # LSTM (forecasting_time_series/methods/lstm.py): a recurrent network reading the lag days in order (one step per day),
    # its last state joined with day d's known-in-advance inputs, then dense layers with an output per step
    'LSTM': {'supports_exog': True,
             'default_parameter_ranges': {
                 'steps_per_day': 24,
                 'minimum_calibration_days': 28,
                 'lag_days': (1, 2, 3, 4, 5, 6, 7),   # the days of the sequence
                 'lag_resolution': 'hourly',
                 'exog_lag_days': {},                 # lagged-only variables: only their smallest lag counts
                 'exog_lag_transform': 'asinh',
                 'sequence_known_inputs': True,       # forecasts and dummies of each lag day in its step
                 'sequence_steps': 'day',             # 'day': a step per lag day; 'intraday': per hour
                 'head_inputs': 'profile',            # day d's known inputs: 'profile' or 'daily_mean'
                 'lstm_units_range': [64],            # size of the LSTM's state
                 'lstm_layers': 1,
                 'hidden_layers_range': [()],         # dense layers after the LSTM, () for none
                 'dropout_range': [0.1],
                 'weight_decay_range': [1e-3],
                 'learning_rate_range': [1e-3],
                 'activation': 'relu',
                 'batch_size': 16,
                 'max_epochs': 500,
                 'patience': 30,
                 'validation_days': 14,
                 'refit': True,
                 'n_seeds': 3,
                 'seed': 0},
             'default_processing_ranges': {'transform_range': (None,),
                                           'differencing_range': (0,),
                                           'normalisation_range': (None,),
                                           'decomposition_range': (None,)}},
    # RF (forecasting_time_series/methods/random_forest.py): one multi-output random forest with an output per step of the
    # day, on the MLP's inputs; the forest's settings are ranges
    'RF': {'supports_exog': True,
           'default_parameter_ranges': {
               'steps_per_day': 24,
               'minimum_calibration_days': 28,
               'lag_days': (1, 2, 3, 7),
               'lag_resolution': 'hourly',
               'exog_lag_days': {},
               'exog_lag_transform': 'asinh',
               'n_estimators_range': [300],
               'max_depth_range': [None],         # None: grow until the leaves are small
               'min_samples_leaf_range': [1],     # fewest calibration days per leaf
               'max_features_range': [0.33],      # share of the inputs tried at each split, or 'sqrt'
               'max_samples': None,               # share of the days drawn per tree, None = all
               'target_anchor_range': [None],     # None: forecast the price; else the price minus a past anchor
               'seed': 0,
               'n_jobs': -1},                     # every core for the trees (the search itself runs in one)
           'default_processing_ranges': {'transform_range': (None,),
                                         'differencing_range': (0,),
                                         'normalisation_range': (None,),
                                         'decomposition_range': (None,)}},
}

# methods forecasting one whole day at a time from the days before it, on the inputs of
# forecasting_time_series/methods/daily_inputs.py (exog_lag_days, steps_per_day)
DAILY_METHODS = {'LEAR', 'MLP', 'LSTM', 'RF'}  # their modules: method.method_module (imported on first use)


@dataclass
class ForecastExperiment:
    """Grid-search one forecasting method on one series and rank the results."""

    # required
    data: pd.DataFrame
    outcome_variable: str

    # what to run
    method: str = 'Naive'
    method_parameter_ranges: dict | None = None   # None → defaults for `method`

    # data layout
    time_variable: str | None = None
    time_sequence_variable: str | None = None     # None → a constant column is added

    # exogenous regressors
    exog_variables: tuple[str, ...] | list[str] | str = ()   # columns of `data`
    future_exog: pd.DataFrame | None = None                  # their values after the last observation

    # forecasting and validation
    forecasting_period: int = 7
    metric: str = 'rmse'                           # 'rmse', 'mae' or 'mape' (in percent)
    validation_minimum_training_split: float = 0.7
    validation_maximum_number_of_splits: int = 7
    score_aggregation: str = 'pooled'              # 'pooled': metric over all errors of all splits;
                                                   # 'mean' / 'median': metric per split, then averaged
    # training data of each validation split: None = expanding window (everything before the split's
    # test window); a number = fixed window (only that many most recent rows). Several values are
    # tested as a grid dimension, e.g. (None, 4 * 672, 8 * 672). The final forecast uses the same rule.
    training_window_range: tuple = (None,)

    # running the search
    n_jobs: int = 1                                # worker processes: 1 = none, -1 = every CPU core
    verbose: bool = True                           # print progress while searching

    # preprocessing, always applied in this order: transform → decomposition → differencing → normalisation
    # None → the method's default, else PROCESSING_DEFAULTS (see forecasting_time_series/processing.py for every option)
    transform_range: tuple | None = None       # None, 'sqrt', 'log', 'asinh', ('power', p), ('asinh', centre)
    differencing_range: tuple | None = None    # 0, a lag such as 1 or 12, or several lags such as (1, 12)
    normalisation_range: tuple | None = None   # None, 'minmax', 'mean', 'median'
    decomposition_range: tuple | None = None   # None, or MSTL settings dicts: see forecasting_time_series/decomposition.py
                                               # (decomposition.mstl_options(...) builds a list of them)

    # ensembles: every subset of 2+ of the top n single combinations, combined step by step
    # and scored on the same validation splits (e.g. n = 3 → #1+#2, #1+#3, #2+#3, #1+#2+#3)
    ensembles: bool = False
    ensemble_top_n: int = 3
    ensemble_aggregations: tuple = ('mean', 'median')

    # filled in by run()
    results: list[SearchResult] = field(default_factory=list, init=False, repr=False)  # best first
    failed: list[SearchResult] = field(default_factory=list, init=False, repr=False)
    split_plan: SplitPlan | None = field(default=None, init=False, repr=False)  # how the splits were chosen
    ensemble_results: list[EnsembleResult] = field(default_factory=list, init=False, repr=False)  # best first
    ensemble_member_ranks: list[int] = field(default_factory=list, init=False, repr=False)  # distinct members
    # rolling forecasts already made, per (rank, data, start, horizon): reused by ensembles
    _rolling_cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._check_method()
        self._check_columns()
        self._check_validation_settings()
        self._check_processing()
        self._check_ensembles()
        self._check_training_windows()
        self._check_exog()
        self._check_future_exog()

        defaults = copy.deepcopy(METHODS[self.method]['default_parameter_ranges'])
        self.method_parameter_ranges = {**defaults, **(self.method_parameter_ranges or {})}
        self._check_method_parameters()

        if self.time_sequence_variable is None:
            self.data = self.data.assign(_series=1)
            self.time_sequence_variable = '_series'

    # ------------------------------------------------------------------ main
    def run(self) -> ForecastExperiment:
        search = grid_search.GridSearch(self.data, self.method,
                                        self._general_parameters(),
                                        self._parameter_ranges(),
                                        score_aggregation=self.score_aggregation,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose).run()
        self.results = search.results
        self.failed = search.failed
        self.split_plan = search.plan
        self._rolling_cache = {}  # the ranking may have changed
        self._warn_if_short_series()
        self.ensemble_results, self.ensemble_member_ranks = [], []
        if self.ensembles:
            members = ensemble_module.distinct_members(self.results, self.ensemble_top_n)
            self.ensemble_member_ranks = [rank for rank, _ in members]
            if len(members) >= 2:
                self.ensemble_results = ensemble_module.evaluate_ensembles(members, self.ensemble_aggregations,
                                                                           self.metric, self.score_aggregation)
        return self  # allows experiment.run().report()

    @property
    def number_of_splits(self) -> int | None:
        """Validation splits every combination was scored on (None before run())."""
        return self.split_plan.number_of_splits if self.split_plan else None

    def best(self, n: int = 5) -> list[SearchResult]:
        self._check_has_run()
        if not self.results:
            raise RuntimeError('no combination succeeded on every validation split: see failed_table()')
        return self.results[:n]

    def results_table(self, n: int | None = None) -> pd.DataFrame:
        """Ranked results: one row per parameter combination, one column per parameter,
        then the score. `n` limits the table to the best n rows (None → all)."""
        rows = self.best(n if n is not None else len(self.results))
        table = pd.DataFrame([{**self._settings_row(result), self.metric: result.score} for result in rows])
        table.index = pd.RangeIndex(1, len(table) + 1, name='rank')
        return table

    def failed_table(self) -> pd.DataFrame:
        """Every combination that was not ranked, in grid order, with how many validation
        splits it succeeded on and why it failed."""
        self._check_has_run()
        n = self.number_of_splits
        table = pd.DataFrame([{**self._settings_row(result),
                               'history_needed': result.required_history,
                               'splits_scored': f'{len(result.split_scores)}/{n}',
                               'reason': '; '.join(dict.fromkeys(result.errors)) or 'unknown'}
                              for result in self.failed],
                             index=pd.Index([result.index for result in self.failed], name='grid_position'))
        return table

    def failure_summary(self) -> Counter:
        """How many combinations failed, by first failure reason (details in brackets are
        dropped so that, e.g., different row counts are grouped together)."""
        return Counter(result.errors[0].split(' (')[0] if result.errors else 'unknown'
                       for result in self.failed)

    def failed_split_share(self) -> float:
        """Share of all validation splits tried (every combination × every split) that failed."""
        self._check_has_run()
        attempted = self.results + self.failed
        n = self.number_of_splits
        failed = sum(n - len(result.split_scores) for result in attempted)
        return failed / (n * len(attempted)) if attempted else 0.0

    def report(self, n: int = 5) -> None:
        self._check_has_run()
        plan = self.split_plan
        total = len(self.results) + len(self.failed)
        print(f'{self.method}: {total} combinations, {len(self.results)} ranked, {len(self.failed)} failed '
              f'({self.failed_split_share():.0%} of validation splits failed)')
        splits = 'split' if plan.number_of_splits == 1 else 'splits'
        observations = 'observation' if self.forecasting_period == 1 else 'observations'
        windows = f'{plan.number_of_splits} validation {splits} of {self.forecasting_period} {observations}'
        if self.score_aggregation == 'pooled':
            print(f'Ranked by {self.metric} over all errors of {windows}, pooled together')
        else:
            print(f'Ranked by {self.score_aggregation} {self.metric} across {windows}')
        if self.training_window_range != (None,):
            windows_tested = ', '.join(grid_search.training_window_label(w) for w in self.training_window_range)
            print(f'Training windows tested: {windows_tested} (fixed lengths in observations)')
        if plan.constraining:
            example = ', '.join(f'{k}={v}' for k, v in plan.constraining[0].items())
            print(f'  (the minimum training share allows {plan.allowed_by_training_share} splits; '
                  f'reduced to {plan.number_of_splits} so that {len(plan.constraining)} combination(s) '
                  f'needing {plan.required_history} rows of history can run, e.g. {example})')
        print()
        for reason, count in self.failure_summary().most_common():
            print(f'  {count} failed: {reason}')
        if self.failed:
            print('  (full list: failed_table())\n')
        if not self.results:
            print('No combination succeeded on every validation split.')
            return
        table = self.results_table(n).reset_index()  # rank as an ordinary column
        print(table.to_string(index=False, formatters={self.metric: lambda x: f'{x:,.2f}'}))
        if self.ensembles:
            self._report_ensembles(n)

    def ensemble_table(self, n: int | None = None) -> pd.DataFrame:
        """Ranked ensembles: which single combinations (by rank) they combine, how, and the score."""
        self._check_has_run()
        rows = self.ensemble_results[:n] if n is not None else self.ensemble_results
        table = pd.DataFrame([{'members': result.label, 'size': result.size,
                               'aggregation': result.aggregation, self.metric: result.score}
                              for result in rows])
        table.index = pd.RangeIndex(1, len(table) + 1, name='ensemble_rank')
        return table

    def validation_score(self, rank: int = 1, ensemble: bool = False) -> float:
        """Walk-forward validation score of the combination (or ensemble) at `rank`."""
        return (self._ensemble_at(rank) if ensemble else self._result_at(rank)).score

    def best_parameters(self, rank: int = 1) -> dict:
        """Settings of the combination at `rank`: method parameters, processing parameters and
        training window (None = expanding)."""
        result = self._result_at(rank)
        return {'method_parameters': dict(result.method_parameters),
                'processing_parameters': dict(result.processing_parameters),
                'training_window': result.training_window}

    def forecast(self, horizon: int | None = None, rank: int = 1, ensemble: bool = False,
                 data: pd.DataFrame | None = None, future_exog: pd.DataFrame | None = None) -> list[float]:
        """Fit the combination at `rank` on all of this experiment's data and forecast
        `horizon` steps beyond its end (default: forecasting_period).
        With ensemble=True, `rank` refers to the ensemble ranking: each member is fitted
        and forecast, and the forecasts are combined with the ensemble's aggregation.
        `data` replaces the experiment's data as the history to fit on (same columns), e.g.
        to forecast each day of a test period from everything observed before it.
        A combination validated with a fixed training window is fitted on the last
        `training_window` rows of the history only, as in validation.
        `future_exog`: the exogenous variables over the `horizon` steps (known in advance, e.g. calendar
        dummies); needed when the experiment uses exogenous variables. Without it, the experiment's own
        future_exog is used - only when forecasting from the experiment's data (`data` not given)."""
        if ensemble:
            chosen = self._ensemble_at(rank)
            member_forecasts = [self.forecast(horizon, rank=member, data=data, future_exog=future_exog)
                                for member in chosen.members]
            return [float(value) for value in ensemble_module.combine(member_forecasts, chosen.aggregation)]
        horizon = horizon or self.forecasting_period
        history = self.data if data is None else data
        if self.known_future_exog:
            if future_exog is None and data is None:
                future_exog = self.future_exog
            future_exog = self._checked_future_exog(future_exog, horizon, history)
        general_parameters = {**self._general_parameters(), 'forecasting_period': horizon}
        parameters = {'method': self.method, 'general_parameters': general_parameters,
                      **self.best_parameters(rank)}
        if parameters['training_window'] is not None:
            history = history.iloc[-parameters['training_window']:]
        if self.time_sequence_variable not in history.columns:  # the constant column added in __post_init__
            history = history.assign(**{self.time_sequence_variable: 1})
        predictions = forecast.forecast_agent(history, parameters).forecast(future_exog)
        if len(predictions) < horizon:
            raise RuntimeError(f'the model produced {len(predictions)} of {horizon} forecasts')
        return predictions

    def rolling_forecasts(self, data: pd.DataFrame, start: int, horizon: int | None = None, rank: int = 1,
                          ensemble: bool = False, n_jobs: int | None = None,
                          verbose: bool | None = None) -> list[float]:
        """Forecast data[start:] in blocks of `horizon` steps (default: forecasting_period), each block
        refitted on all of `data` before it - e.g. a test period, one day at a time.

        Exogenous variables are taken from `data` for each block (they must be known in advance).
        With ensemble=True, each member's rolling forecasts are combined step by step; members already
        forecast (e.g. the best single combination) are reused, not refitted.
        n_jobs: worker processes for the blocks (None → the experiment's n_jobs, -1 → every core).
        verbose: print progress (None → the experiment's verbose)."""
        horizon = horizon or self.forecasting_period
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        verbose = self.verbose if verbose is None else verbose
        if ensemble:
            members = self._ensemble_at(rank).members
        else:
            self._result_at(rank)  # raises for an invalid rank
            members = (rank,)
        fingerprint = int(pd.util.hash_pandas_object(data, index=False).sum())
        key = lambda member: (member, fingerprint, start, horizon)
        missing = [member for member in members if key(member) not in self._rolling_cache]
        if missing:
            blocks = list(range(start, len(data), horizon))
            tasks = [(member, block) for member in missing for block in blocks]
            outcomes = _run_rolling_tasks(self, data, horizon, tasks, n_jobs, verbose)
            for member in missing:
                values = [value for (m, _), forecast_values in zip(tasks, outcomes) if m == member
                          for value in forecast_values]
                self._rolling_cache[key(member)] = values[:len(data) - start]
        member_forecasts = [self._rolling_cache[key(member)] for member in members]
        if not ensemble:
            return list(member_forecasts[0])
        aggregation = self._ensemble_at(rank).aggregation
        return [float(value) for value in ensemble_module.combine(member_forecasts, aggregation)]

    def score(self, predictions, actual) -> float:
        """Score predictions against actual values with this experiment's metric."""
        return metrics.evaluator({'type': self.metric}).score(list(predictions), list(actual))

    def _settings_row(self, result: SearchResult) -> dict:
        return {**result.method_parameters, **processing.describe(**result.processing_parameters),
                'training_window': grid_search.training_window_label(result.training_window)}

    def _check_has_run(self) -> None:
        if self.split_plan is None:
            raise RuntimeError('no results: call run() first')

    def _warn_if_short_series(self) -> None:
        """Warn once if even a single validation split is below the minimum training share."""
        n = len(self.data)
        if not self.split_plan.meets_training_share:
            earliest_training = max(n - self.split_plan.number_of_splits * self.forecasting_period, 0)
            warnings.warn(
                f'Short series: with {n} observations and forecasting_period={self.forecasting_period}, '
                f'even a single validation split leaves only {earliest_training} training observations '
                f'({earliest_training / n:.0%}), below validation_minimum_training_split='
                f'{self.validation_minimum_training_split:.0%}. Scores rest on one window and little '
                f'history, so treat the ranking with caution.',
                stacklevel=3)

    def _result_at(self, rank: int) -> SearchResult:
        self.best(1)  # raises if run() was not called or nothing was ranked
        if not 1 <= rank <= len(self.results):
            raise ValueError(f'rank must be between 1 and {len(self.results)}')
        return self.results[rank - 1]

    def _ensemble_at(self, rank: int) -> EnsembleResult:
        self._check_has_run()
        if not self.ensemble_results:
            raise RuntimeError('no ensembles: set ensembles=True (and rank at least 2 combinations)')
        if not 1 <= rank <= len(self.ensemble_results):
            raise ValueError(f'ensemble rank must be between 1 and {len(self.ensemble_results)}')
        return self.ensemble_results[rank - 1]

    def _report_ensembles(self, n: int) -> None:
        ranks = self.ensemble_member_ranks
        skipped = (ranks[-1] - len(ranks)) if ranks else 0  # duplicates passed over before the last member
        note = ('; pairs use the mean only, since the median of two values is their mean'
                if {'mean', 'median'} <= set(self.ensemble_aggregations) else '')
        print(f'\nEnsembles of the top {len(ranks)} distinct combinations '
              f'(ranks {", ".join(f"#{r}" for r in ranks)}'
              f'{f"; {skipped} skipped as duplicates of a better-ranked forecast" if skipped else ""}): '
              f'{len(self.ensemble_results)} tested '
              f'({" and ".join(self.ensemble_aggregations)} of each subset of 2 or more{note})')
        if not self.ensemble_results:
            print('  none: at least 2 ranked combinations with distinct forecasts are needed')
            return
        table = self.ensemble_table(n).reset_index()
        print(table.to_string(index=False, formatters={self.metric: lambda x: f'{x:,.2f}'}))
        best_ensemble, best_single = self.ensemble_results[0], self.results[0]
        better = best_ensemble.score < best_single.score
        print(f'\nBest ensemble ({best_ensemble.label}, {best_ensemble.aggregation}) '
              f'{self.metric} {best_ensemble.score:,.2f} vs best single {best_single.score:,.2f}: '
              f'{"the ensemble is better" if better else "the single combination is at least as good"}')

    # ------------------------------------------------------------------ checks
    def _check_method(self):
        if self.method not in METHODS:
            raise ValueError(f'unknown method {self.method!r}; choose from {list(METHODS)}')

    def _check_method_parameters(self):
        ranges = self.method_parameter_ranges
        known = METHODS[self.method]['default_parameter_ranges']
        unknown = [name for name in ranges if name not in known]
        if unknown:
            raise ValueError(f'unknown {self.method} parameters {unknown}; choose from {list(known)}')
        if self.method not in DAILY_METHODS:
            return
        module = method_registry.method_module(self.method)
        for combination in grid_builder.builder(self.method, {'method_parameter_ranges': ranges}).get_method_grid():
            module.check_parameters(combination)  # raises ValueError for a malformed option
        not_exog = [c for c in (ranges.get('exog_lag_days') or {}) if c not in self.exog_variables]
        if not_exog:
            raise ValueError(f'exog_lag_days lists {not_exog}: add them to exog_variables too')
        known = daily_inputs.known_in_advance(self.exog_variables, ranges.get('exog_lag_days'))
        not_known = [c for c in (ranges.get('lag_interactions') or ()) if c not in known]
        if not_known:
            raise ValueError(f'lag_interactions lists {not_known}: they must be exogenous variables known '
                             f'in advance (in exog_variables, not lagged-only in exog_lag_days)')
        steps = ranges['steps_per_day']
        # every forecast must cover exactly one day, starting at its first step
        if self.forecasting_period != steps:
            raise ValueError(f'{self.method} forecasts one whole day at a time: set forecasting_period={steps} '
                             f'(steps_per_day), got {self.forecasting_period}')
        not_whole_days = [w for w in self.training_window_range if w is not None and w % steps]
        if not_whole_days:
            raise ValueError(f'{self.method} training windows must be whole days (multiples of steps_per_day={steps}); '
                             f'got {not_whole_days}')

    def _check_columns(self):
        if self.outcome_variable not in self.data.columns:
            raise ValueError(f'column {self.outcome_variable!r} not in data')
        if self.time_variable is not None and self.time_variable not in self.data.columns:
            raise ValueError(f'column {self.time_variable!r} not in data')
        if self.time_sequence_variable is not None and self.time_sequence_variable not in self.data.columns:
            raise ValueError(f'column {self.time_sequence_variable!r} not in data')

    def _check_validation_settings(self):
        if not 0 < self.validation_minimum_training_split < 1:
            raise ValueError('validation_minimum_training_split must be between 0 and 1')
        if self.forecasting_period < 1:
            raise ValueError('forecasting_period must be at least 1')
        if self.validation_maximum_number_of_splits < 1:
            raise ValueError('validation_maximum_number_of_splits must be at least 1')
        if self.metric not in metrics.METRICS:
            raise ValueError(f'unknown metric {self.metric!r}; choose from {list(metrics.METRICS)}')
        if self.score_aggregation not in grid_search.SCORE_AGGREGATIONS:
            raise ValueError(f'score_aggregation must be one of {list(grid_search.SCORE_AGGREGATIONS)}')
        if self.n_jobs != -1 and self.n_jobs < 1:
            raise ValueError('n_jobs must be -1 (every core) or at least 1')

    def _check_processing(self):
        builders = {'transform_range': processing.build_transform,
                    'differencing_range': processing.build_differencing,
                    'normalisation_range': processing.build_normaliser,
                    'decomposition_range': processing.build_decomposition}
        method_defaults = METHODS[self.method].get('default_processing_ranges', {})
        for name, build in builders.items():
            options = getattr(self, name)
            if options is None:
                options = method_defaults.get(name, PROCESSING_DEFAULTS[name])
            if isinstance(options, dict):  # a single MSTL settings dict
                options = (options,)
            options = tuple(options)
            if name == 'decomposition_range':  # fill in MSTL defaults, so every option is complete
                options = tuple(None if o is None else mstl.normalise_spec(o) for o in options)
            if not options:
                raise ValueError(f'{name} needs at least one option')
            for option in options:
                build(option)  # raises ValueError for an unknown or malformed option
            setattr(self, name, options)

    def _check_training_windows(self):
        options = self.training_window_range
        options = (options,) if options is None or isinstance(options, int) else tuple(options)
        if not options:
            raise ValueError('training_window_range needs at least one option (None = expanding window)')
        for option in options:
            if option is not None and (isinstance(option, bool) or not isinstance(option, int) or option < 1):
                raise ValueError(f'training windows must be None (expanding) or a whole number of observations; '
                                 f'got {option!r}')
        self.training_window_range = tuple(dict.fromkeys(options))  # drop duplicates, keep order

    def _check_ensembles(self):
        if isinstance(self.ensemble_top_n, bool) or not isinstance(self.ensemble_top_n, int) \
                or self.ensemble_top_n < 2:
            raise ValueError('ensemble_top_n must be a whole number of at least 2')
        if isinstance(self.ensemble_aggregations, str):
            self.ensemble_aggregations = (self.ensemble_aggregations,)
        self.ensemble_aggregations = tuple(self.ensemble_aggregations)
        unknown = [a for a in self.ensemble_aggregations if a not in ensemble_module.ENSEMBLE_AGGREGATIONS]
        if not self.ensemble_aggregations or unknown:
            raise ValueError(f'ensemble_aggregations must be chosen from '
                             f'{list(ensemble_module.ENSEMBLE_AGGREGATIONS)}; got {self.ensemble_aggregations}')

    def _check_exog(self):
        if isinstance(self.exog_variables, str):
            self.exog_variables = (self.exog_variables,)
        self.exog_variables = tuple(self.exog_variables)
        if not self.exog_variables:
            return
        if not METHODS[self.method]['supports_exog']:
            raise ValueError(f'{self.method} cannot use exogenous regressors')
        if self.outcome_variable in self.exog_variables:
            raise ValueError('the outcome cannot also be a regressor')
        missing = [c for c in self.exog_variables if c not in self.data.columns]
        if missing:
            raise ValueError(f'regressor columns not in data: {missing}')
        not_numeric = [c for c in self.exog_variables if not pd.api.types.is_numeric_dtype(self.data[c])]
        if not_numeric:
            raise ValueError(f'regressors must be numeric (e.g. 0/1 dummies): {not_numeric}')
        if self.data[list(self.exog_variables)].isna().any().any():
            raise ValueError('regressors contain missing values')

    def _check_future_exog(self):
        if self.future_exog is None:
            return
        if not self.exog_variables:
            raise ValueError('future_exog given, but exog_variables is empty')
        self.future_exog = self._checked_future_exog(self.future_exog, self.forecasting_period, self.data)

    @property
    def known_future_exog(self) -> list[str]:
        """The exogenous variables whose values over the forecast horizon are needed (known in advance).
        For LEAR and MLP, variables listed in exog_lag_days without lag 0 are used only through their past."""
        if self.method in DAILY_METHODS:
            lags = (self.method_parameter_ranges or {}).get('exog_lag_days') or {}
            return daily_inputs.known_in_advance(self.exog_variables, lags)
        return list(self.exog_variables)

    def _checked_future_exog(self, future_exog, horizon, history) -> pd.DataFrame | None:
        """The exogenous variables known in advance over the next `horizon` steps after `history`,
        checked (None if there are none). If future_exog has the time column, its first date must
        come after history's last."""
        known = self.known_future_exog
        if not known:
            return None
        if future_exog is None:
            raise ValueError(f'this experiment uses exogenous variables known in advance {known}: '
                             f'pass their values over the {horizon} steps forecast as future_exog')
        missing = [c for c in known if c not in future_exog.columns]
        if missing:
            raise ValueError(f'future_exog is missing columns: {missing}')
        if len(future_exog) < horizon:
            raise ValueError(f'future_exog needs at least {horizon} rows, got {len(future_exog)}')
        future_exog = future_exog.iloc[:horizon]
        if future_exog[known].isna().any().any():
            raise ValueError('future_exog contains missing values')
        time = self.time_variable
        if time is not None and time in future_exog.columns and time in history.columns:
            first, last = pd.to_datetime(future_exog[time].iloc[0]), pd.to_datetime(history[time].iloc[-1])
            if first <= last:
                raise ValueError(f'future_exog starts at {future_exog[time].iloc[0]}, not after the end of '
                                 f'the data ({history[time].iloc[-1]})')
        columns = known + ([time] if time in future_exog.columns else [])
        return future_exog[columns].reset_index(drop=True)

    # ------------------------------------------- translation to the existing dicts
    def _general_parameters(self) -> dict:
        return {'outcome_variable': self.outcome_variable,
                'time_variable': self.time_variable,
                'time_sequence_variable': self.time_sequence_variable,
                'exog_variables': list(self.exog_variables),
                'known_future_exog': self.known_future_exog,
                'forecasting_period': self.forecasting_period,
                'metrics': {'type': self.metric},
                'validation_minimum_training_split': self.validation_minimum_training_split,
                'validation_maximum_number_of_splits': self.validation_maximum_number_of_splits}

    def _parameter_ranges(self) -> dict:
        return {'method_parameter_ranges': self.method_parameter_ranges,
                'processing_parameter_ranges': {
                    'transform_range': list(self.transform_range),
                    'differencing_range': list(self.differencing_range),
                    'normalisation_range': list(self.normalisation_range),
                    'decomposition_range': list(self.decomposition_range)},
                'training_window_range': list(self.training_window_range)}


# ---------------------------------------------------------------------- rolling forecasts, optionally in parallel
# Module-level functions, so worker processes can run them.

def _forecast_block(experiment, data, horizon, member, block_start) -> list[float]:
    # only the variables known in advance are taken from the block itself (see known_future_exog)
    future_exog = data.iloc[block_start:block_start + horizon] if experiment.known_future_exog else None
    return experiment.forecast(horizon=horizon, rank=member, data=data.iloc[:block_start], future_exog=future_exog)


def _run_rolling_tasks(experiment, data, horizon, tasks, n_jobs, verbose) -> list[list[float]]:
    """Forecast every (member rank, block start) task; results in task order."""
    workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    workers = max(1, min(workers, len(tasks)))
    total, report_every = len(tasks), max(1, len(tasks) // 20)
    in_terminal = sys.stdout.isatty()

    def progress(done):
        if verbose and (done % report_every == 0 or done == total):
            message = f'Rolling forecasts: {done}/{total} blocks ({workers} process{"es" if workers > 1 else ""})'
            print(f'\r{message}' if in_terminal else message, end='' if in_terminal else '\n', flush=True)

    outcomes = []
    if workers == 1:
        for done, (member, block) in enumerate(tasks, start=1):
            outcomes.append(_forecast_block(experiment, data, horizon, member, block))
            progress(done)
    else:
        # each worker receives the experiment and the data once (initializer), then only small tasks
        light = copy.copy(experiment)
        light._rolling_cache = {}
        with ProcessPoolExecutor(workers, initializer=_init_rolling_worker, initargs=(light, data, horizon)) as pool:
            for done, outcome in enumerate(pool.map(_rolling_task_in_worker, tasks), start=1):
                outcomes.append(outcome)
                progress(done)
    if verbose and in_terminal:
        print()
    return outcomes


_ROLLING_CONTEXT = None


def _init_rolling_worker(experiment, data, horizon) -> None:
    global _ROLLING_CONTEXT
    _ROLLING_CONTEXT = (experiment, data, horizon)
    # one maths-library thread per process: the processes already use every core, and
    # several threads per process would only compete with each other
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _rolling_task_in_worker(task) -> list[float]:
    experiment, data, horizon = _ROLLING_CONTEXT
    return _forecast_block(experiment, data, horizon, *task)
