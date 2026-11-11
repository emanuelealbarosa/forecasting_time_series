"""Evaluating experiments on a test period that their search never saw, and saving the run.

    result = evaluate({'LEAR': lambda train: daily_experiment(train, 'LEAR', ...),
                       'MLP': lambda train: daily_experiment(train, 'MLP', ...)},
                      data, test_days=28, steps_per_day=96, save_to='results')
    result.table     # each model's validation and test scores
    result.folder    # the saved run (see results_log.py)

The steps, each available on its own:
  split_train_test  the last test_days whole days are the test period; everything before is training data
  (the builders)    each experiment is built on the training data only and searched (experiment.run)
  forecast_candidates   its candidates (every ranked single and ensemble, or the best of each) forecast the test
                    days one day at a time, refitted on everything before each day
  forecast_combined     with 2+ experiments: the best combination of their top members (CombinedEnsembles)
  forecast_blockwise    each part of the day takes the candidate that did best on it on the validation days
                    (BlockSelection)
  save              forecasts, settings and one row per model in <save_to>/runs.csv (results_log.save_run)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics, results_log
from .block_selection import BlockSelection
from .ensemble import CombinedEnsembles
from .experiment import ForecastExperiment

# four parts of the day of six (local) hours each, for forecast_blockwise
SIX_HOUR_BLOCKS = {'00-05': range(0, 6), '06-11': range(6, 12), '12-17': range(12, 18), '18-23': range(18, 24)}


@dataclass
class EvaluationResult:
    test: pd.DataFrame                              # the test period's rows of the data
    experiments: dict[str, ForecastExperiment]      # each experiment, searched
    forecasts: dict[str, list[float]] = field(default_factory=dict)   # model label → test forecasts
    records: list[dict] = field(default_factory=list)                 # one per model, as save_run takes them
    folder: Path | None = None                      # the saved run, if saved

    @property
    def table(self) -> pd.DataFrame:
        """One row per model: its kind, validation score and test RMSE and MAE."""
        if not self.records:
            return pd.DataFrame(columns=['model', 'kind', 'validation', 'test_rmse', 'test_mae'])
        experiment = next(iter(self.experiments.values()))
        actual = self.test[experiment.outcome_variable].to_numpy(dtype=float)
        return pd.DataFrame([{'model': r['model'], 'kind': r['kind'],
                              f'validation_{r["validation_metric"]}': r['validation_score'],
                              'test_rmse': metrics.rmse(np.asarray(r['forecast'], dtype=float), actual),
                              'test_mae': metrics.mae(np.asarray(r['forecast'], dtype=float), actual)}
                             for r in self.records])


def split_train_test(data: pd.DataFrame, test_days: int, steps_per_day: int, shift_days: int = 0):
    """(data, train, test): the test period is the last test_days whole days, moved shift_days earlier
    (the data after it is dropped, never seen at all); the training data is everything before it."""
    if shift_days:
        data = data.iloc[:-shift_days * steps_per_day]
    test_size = test_days * steps_per_day
    if test_size >= len(data):
        raise ValueError(f'{test_days} test days leave no training data ({len(data)} rows)')
    return data, data.iloc[:-test_size], data.iloc[-test_size:]


def _record(label, experiment_name, kind, forecast, metric, validation_score, validation_splits, settings,
            validation_forecast) -> dict:
    return {'model': label, 'experiment': experiment_name, 'kind': kind, 'forecast': forecast,
            'validation_metric': metric, 'validation_score': validation_score,
            'validation_days': validation_splits, 'settings': settings, 'validation_forecast': validation_forecast}


def forecast_candidates(name: str, experiment: ForecastExperiment, data: pd.DataFrame, start: int, horizon: int,
                    candidates: str = 'all', n_jobs: int = 1, verbose: bool = True) -> list[dict]:
    """The test forecasts of a searched experiment's candidates, from row `start` of `data`, one record each.
    candidates: 'all' (every ranked single combination and every ensemble; the ensembles reuse their
    members' forecasts, so only the extra singles cost time) or 'best' (the best single and best ensemble)."""
    if candidates not in ('all', 'best'):
        raise ValueError(f"candidates must be 'all' or 'best', got {candidates!r}")
    choices = {'best single': dict(rank=1, ensemble=False)}
    if experiment.ensemble_results:
        choices['best ensemble'] = dict(rank=1, ensemble=True)
    if candidates == 'all':
        windows = experiment.results_table()['training_window']
        choices = {f'single #{rank} (window {windows[rank]})': dict(rank=rank, ensemble=False)
                   for rank in range(1, len(experiment.results) + 1)}
        choices.update({f'ensemble {e.label}, {e.aggregation}': dict(rank=rank, ensemble=True)
                        for rank, e in enumerate(experiment.ensemble_results or [], start=1)})
    records = []
    for choice_name, choice in choices.items():
        if verbose:
            print(f'\nTest forecasts, {choice_name}:')
        forecast = experiment.rolling_forecasts(data, start, horizon, n_jobs=n_jobs, verbose=verbose, **choice)
        records.append(_record(f'{name}: {choice_name}', name, 'ensemble' if choice['ensemble'] else 'single',
                               forecast, experiment.metric, experiment.validation_score(**choice),
                               experiment.number_of_splits, results_log.candidate_settings(experiment, **choice),
                               results_log.validation_forecast(experiment, **choice)))
    return records


def _methods(experiments: dict) -> str:
    return ' + '.join(dict.fromkeys(e.method for e in experiments.values()))


def forecast_combined(experiments: dict[str, ForecastExperiment], data: pd.DataFrame, start: int, horizon: int,
                  n_jobs: int = 1, verbose: bool = True, pool: CombinedEnsembles | None = None) -> dict:
    """The test forecasts of the best combination of the experiments' top members (each experiment's
    ensemble_top_n), alone or combined, within or across experiments, ranked on the validation days they
    share; one record. pool: that CombinedEnsembles of the experiments, if already made (run or not)."""
    combined = pool if pool is not None else CombinedEnsembles(experiments)
    if not combined.results:
        combined.run()
    if verbose:
        combined.report()
    best = combined.results[0]
    label = f'All methods: best ({combined.label(best)}{"" if best.size == 1 else ", " + best.aggregation})'
    if verbose:
        print('\nTest forecasts, best combination of all methods:')
    forecast = combined.rolling_forecasts(data, start, horizon, n_jobs=n_jobs, verbose=verbose)
    members = [{'experiment': combined.members[p].method, 'rank': combined.members[p].rank,
                **results_log.single_settings(combined.members[p].experiment, combined.members[p].rank)}
               for p in best.members]
    settings = {'method': _methods(experiments), 'aggregation': best.aggregation, 'members': members,
                'fit_seconds': results_log.total_fit_seconds(members)}
    metric = next(iter(experiments.values())).metric
    return _record(label, ' + '.join(experiments), 'all methods', forecast, metric, combined.validation_score(),
                   combined.number_of_splits, settings, results_log.pool_validation_forecast(combined, best))


def forecast_blockwise(experiments: dict[str, ForecastExperiment], data: pd.DataFrame, start: int, horizon: int,
                   blocks: dict = SIX_HOUR_BLOCKS, timezone: str | None = None, n_jobs: int = 1,
                   verbose: bool = True, selection: BlockSelection | None = None) -> dict:
    """The test forecasts of the block-wise selection: each part of the day (blocks: {name: local hours})
    takes the candidate (single or ensemble, from any experiment) that scored best on its own steps of the
    validation days; one record. selection: that BlockSelection, if already run."""
    blockwise = selection if selection is not None else BlockSelection(experiments, blocks, timezone=timezone).run()
    if verbose:
        blockwise.report()
    label = 'Block-wise selection (' + ', '.join(f'{c.block}: {blockwise.label(c.rank)}'
                                                for c in blockwise.choices) + ')'
    if verbose:
        print('\nTest forecasts, block-wise selection:')
    forecast = blockwise.rolling_forecasts(data, start, horizon, n_jobs=n_jobs, verbose=verbose)
    pool = blockwise.pool
    members = [results_log.single_settings(pool.members[p].experiment, pool.members[p].rank)
               for p in blockwise.winner_members()]
    settings = {'method': _methods(experiments),
                'blocks': [{'block': str(c.block), 'candidate': blockwise.label(c.rank)} for c in blockwise.choices],
                'fit_seconds': results_log.total_fit_seconds(members)}
    metric = next(iter(experiments.values())).metric
    return _record(label, ' + '.join(experiments), 'block-wise', forecast, metric, blockwise.score,
                   blockwise.pool.number_of_splits, settings, blockwise.validation_predictions)


def save(result: EvaluationResult, directory, not_saved_methods=(), notes: str = '') -> Path:
    """Save the run in `directory` (results_log.save_run). The singles and ensembles of the methods in
    not_saved_methods (e.g. ('Naive',)) are left out; combinations that include them are kept."""
    records = [r for r in result.records
               if not (r['kind'] in ('single', 'ensemble') and r['settings']['method'] in not_saved_methods)]
    experiment = next(iter(result.experiments.values()))
    # the validation days: the longest of the experiments' (all count back from the end of the training data)
    longest = max(result.experiments.values(), key=lambda e: e.number_of_splits)
    validation_dates, validation_actual = results_log.validation_period(longest)
    return results_log.save_run(list(result.experiments), result.test[experiment.time_variable],
                                result.test[experiment.outcome_variable], records, result.experiments,
                                notes=notes, directory=directory, validation_dates=validation_dates,
                                validation_actual=validation_actual)


def evaluate(experiments: dict[str, Callable[[pd.DataFrame], ForecastExperiment]], data: pd.DataFrame,
             test_days: int, steps_per_day: int, *, candidates: str = 'all', combine_methods: bool = True,
             day_blocks: dict | None = SIX_HOUR_BLOCKS, timezone: str | None = None, shift_days: int = 0,
             n_jobs: int = 1, save_to=None, not_saved_methods=(), notes: str = '',
             verbose: bool = True) -> EvaluationResult:
    """Search each experiment on the training data and forecast the test period with its candidates; with
    2+ experiments also their best combination (combine_methods); with day_blocks the block-wise selection.

    experiments: {name: builder}, each builder taking the training data and returning a ForecastExperiment
        (not yet run), e.g. lambda train: daily_experiment(train, 'RF', ...). The experiments must share
        the outcome, time column and forecasting period.
    data: every row, oldest first; the test period is its last test_days whole days (moved shift_days
        earlier), forecast one day (steps_per_day steps) at a time, refitted on all the data before each day.
    candidates: 'all' or 'best' (see forecast_candidates). timezone: the local time of day_blocks' hours.
    n_jobs: processes for the test forecasts (on Windows, call evaluate under `if __name__ == '__main__':`).
    save_to: the results folder (None: not saved). not_saved_methods, notes: see save."""
    data, train, test = split_train_test(data, test_days, steps_per_day, shift_days)
    if verbose:
        print(f'{len(data)} observations ({len(data) // steps_per_day} days); test period: the last '
              f'{test_days} days ({len(test)} observations)\n')
    result = EvaluationResult(test=test, experiments={})
    for name, build in experiments.items():
        if verbose:
            print(f'=== {name} ' + '=' * 60)
        started = time.perf_counter()
        experiment = build(train).run()
        result.experiments[name] = experiment
        if verbose:
            experiment.report()
        result.records += forecast_candidates(name, experiment, data, len(train), steps_per_day, candidates,
                                          n_jobs, verbose)
        if verbose:
            best = experiment.results_table(1).iloc[0].drop(experiment.metric)
            print('\nBest single: ' + ', '.join(f'{column}={value}' for column, value in best.items()))
            if experiment.ensemble_results:
                best_ensemble = experiment.ensemble_table(1).iloc[0]
                print(f'Best ensemble: {best_ensemble.aggregation} of {best_ensemble.members}')
            print(f'({name}: {time.perf_counter() - started:.0f}s including the test forecasts)\n')
    _check_compatible(result.experiments)

    # the candidates of both the combination across methods and the block-wise selection, ranked once
    # (by the block-wise selection, which scores its blocks in the same pass)
    pool = CombinedEnsembles(result.experiments)
    selection = BlockSelection(result.experiments, day_blocks, timezone=timezone, pool=pool).run() if day_blocks \
        else None
    if combine_methods and len(result.experiments) > 1:
        if verbose:
            print('=== All methods ' + '=' * 56)
        result.records.append(forecast_combined(result.experiments, data, len(train), steps_per_day, n_jobs, verbose,
                                                pool=pool))
    if day_blocks:
        if verbose:
            print('=== By part of the day ' + '=' * 50)
        result.records.append(forecast_blockwise(result.experiments, data, len(train), steps_per_day, day_blocks,
                                             timezone, n_jobs, verbose, selection=selection))
    result.forecasts = {r['model']: r['forecast'] for r in result.records}
    if verbose:
        print()
        print(result.table.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    if save_to is not None:
        result.folder = save(result, save_to, not_saved_methods, notes)
        if verbose:
            print(f'\nResults saved to {result.folder} (and appended to {Path(save_to) / "runs.csv"})')
    return result


def _check_compatible(experiments: dict[str, ForecastExperiment]) -> None:
    first = next(iter(experiments.values()))
    for name, experiment in experiments.items():
        for attribute in ('outcome_variable', 'time_variable', 'forecasting_period', 'metric'):
            if getattr(experiment, attribute) != getattr(first, attribute):
                raise ValueError(f'{name} has {attribute}={getattr(experiment, attribute)!r}, '
                                 f'unlike {getattr(first, attribute)!r}: the experiments must share it')
