"""Combining saved runs across methods, without refitting anything.

evaluate() combines the experiments it runs together (the best combination across methods and the
block-wise selection). combine_runs() does the same for experiments evaluated in separate runs - e.g. one
run per experiment, so that a long session that stops loses only the experiment it was running:

    result = combine_runs('results', ['2026-11-11_105410_tiny-RF', '2026-11-11_105435_tiny-LEAR'],
                          timezone='Europe/Bucharest')
    result.table     # each model's validation and test scores
    result.folder    # the new run saved in the same results folder

It needs only what save_run already writes for every run (results_log.py):
  - validation_forecasts.csv: every ranked single combination's forecasts of the validation days, from
    which the cross-method ensembles and the block-wise selection are chosen;
  - forecasts.csv: their test-period forecasts. A combination's test forecast is its members' test
    forecasts combined step by step, as in CombinedEnsembles.rolling_forecasts, so nothing is refitted.

Each saved experiment is rebuilt as a SavedExperiment (the few attributes CombinedEnsembles and
BlockSelection read), so the ranking, tie-breaking and block choices are exactly those of evaluate():
the runs combined here give the same result as one evaluate() of their experiments together.

The runs must share the test period (same timestamps and actual values), the metric, the score
aggregation and the forecasting period; their validation days are counted back from the end of the
training data, so they line up, and only the days all the experiments share are used.
Only single combinations are read: ensembles are rebuilt from them. A run saved with candidates='best'
offers only its best single; a method in not_saved_methods (e.g. Naive) has no singles to offer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from . import results_log
from .block_selection import BlockSelection
from .ensemble import CombinedEnsembles
from .evaluation import SIX_HOUR_BLOCKS, EvaluationResult, _record
from .grid_search import SearchResult

_SINGLE_RANK = re.compile(r': single #(\d+) \(')


@dataclass
class SavedExperiment:
    """One experiment of a saved run, as CombinedEnsembles and BlockSelection see a ForecastExperiment:
    its ranked single combinations (validation forecasts per split) and their saved test forecasts."""
    name: str
    run_id: str
    metric: str
    score_aggregation: str
    forecasting_period: int
    number_of_splits: int
    ensemble_top_n: int
    ensemble_aggregations: tuple
    data: pd.DataFrame                    # validation days shared by every experiment: date, actual
    results: list = field(default_factory=list)          # SearchResult per rank, best first
    labels: list = field(default_factory=list)           # their model labels in the run
    test_forecasts: list = field(default_factory=list)   # their test forecasts
    settings: list = field(default_factory=list)         # their settings (settings.json)
    outcome_variable: str = 'actual'
    time_variable: str = 'date'
    split_plan: object = True             # read by CombinedEnsembles only as "has been run"

    @property
    def method(self) -> str:
        return self.name

    def rolling_forecasts(self, data, start, horizon=None, rank: int = 1, ensemble: bool = False,
                          n_jobs=None, verbose=None) -> list[float]:
        """The saved test forecasts of the single combination at `rank` (data, start and horizon are
        those of the test period the run was saved with)."""
        if ensemble:
            raise ValueError('saved experiments offer their single combinations only')
        return list(self.test_forecasts[rank - 1])


def _read_run(directory: Path, run_id: str) -> dict:
    folder = directory / run_id
    missing = [name for name in ('forecasts.csv', 'validation_forecasts.csv', 'settings.json')
               if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f'{folder} has no {", ".join(missing)} (saved by an older version, or not a run)')
    with open(folder / 'settings.json', encoding='utf-8') as file:
        settings = json.load(file)
    return {'run_id': run_id, 'settings': settings,
            'forecasts': pd.read_csv(folder / 'forecasts.csv'),
            'validation': pd.read_csv(folder / 'validation_forecasts.csv')}


def _singles(run: dict, kinds: dict, experiment: str) -> list[tuple[int, str]]:
    """(rank, label) of the experiment's saved single combinations, best first."""
    singles = []
    for label, kind in kinds.items():
        if kind != 'single' or not label.startswith(f'{experiment}: '):
            continue
        match = _SINGLE_RANK.search(label[len(experiment):])
        if match:
            singles.append((int(match.group(1)), label))
        elif label == f'{experiment}: best single':  # saved with candidates='best'
            singles.append((1, label))
    return sorted(singles)


def _check_same(name: str, first, other, what: str) -> None:
    if len(first) != len(other) or not np.array_equal(np.asarray(first), np.asarray(other)):
        raise ValueError(f'{name} has a different {what} from the first run: runs can only be combined when '
                         f'they share it (same data file, test_days and shift_days)')


def load_saved_experiments(directory, run_ids) -> tuple[dict[str, SavedExperiment], pd.DataFrame]:
    """(experiments, test): the experiments of the saved runs `run_ids` (in the results folder `directory`)
    as SavedExperiments, keyed by experiment name (with ' [<run id>]' added when two runs have the same
    name), and the test period's rows (date, actual). Raises ValueError if the runs cannot be combined."""
    directory = Path(directory)
    if isinstance(run_ids, (str, Path)):
        run_ids = [run_ids]
    runs = [_read_run(directory, str(Path(run_id).name)) for run_id in run_ids]
    if not runs:
        raise ValueError('give at least 1 run')
    table = results_log.load_runs(directory)

    first = runs[0]
    test_dates, test_actual = first['forecasts']['date'].astype(str), first['forecasts']['actual']
    pieces = []
    for run in runs:
        _check_same(run['run_id'], test_dates, run['forecasts']['date'].astype(str), 'test period')
        _check_same(run['run_id'], test_actual.to_numpy(float), run['forecasts']['actual'].to_numpy(float),
                    'test actual values')
        pieces.append(run['validation'][['date', 'actual']])

    # the validation days of every run, aligned on their last day (all end with the training data)
    longest = max(pieces, key=len).reset_index(drop=True)
    for run, piece in zip(runs, pieces):
        tail = longest.iloc[len(longest) - len(piece):]
        _check_same(run['run_id'], tail['date'].astype(str), piece['date'].astype(str), 'validation days')
        _check_same(run['run_id'], tail['actual'].to_numpy(float), piece['actual'].to_numpy(float),
                    'validation actual values')

    experiments: dict[str, SavedExperiment] = {}
    for run in runs:
        rows = table[table['run_id'] == run['run_id']]
        kinds = dict(zip(rows['model'], rows['kind']))
        for name, spec in run['settings']['experiment_settings'].items():
            period, splits = int(spec['forecasting_period']), int(spec['validation_splits'])
            saved = SavedExperiment(name=name, run_id=run['run_id'], metric=spec['metric'],
                                    score_aggregation=spec['score_aggregation'], forecasting_period=period,
                                    number_of_splits=splits, ensemble_top_n=int(spec['ensemble_top_n']),
                                    ensemble_aggregations=tuple(spec['ensemble_aggregations']), data=longest)
            scorer = metrics.evaluator({'type': saved.metric})
            actual = longest['actual'].to_numpy(float)[-splits * period:].reshape(splits, period)
            singles = _singles(run, kinds, name)
            if [rank for rank, _ in singles] != list(range(1, len(singles) + 1)):
                raise ValueError(f'{run["run_id"]}: {name} has single combinations missing from the saved run '
                                 f'(ranks {[rank for rank, _ in singles]})')
            for rank, label in singles:
                values = run['validation'][label].to_numpy(float)[-splits * period:].reshape(splits, period)
                if np.isnan(values).all(axis=1).any():
                    raise ValueError(f'{run["run_id"]}: {label} has validation days without forecasts')
                predictions, actuals = [list(split) for split in values], [list(split) for split in actual]
                saved.results.append(SearchResult(index=rank, method_parameters={}, processing_parameters={},
                                                  split_scores=[scorer.score(p, a) for p, a in
                                                                zip(predictions, actuals)],
                                                  split_predictions=predictions, split_actuals=actuals))
                saved.labels.append(label)
                saved.test_forecasts.append(run['forecasts'][label].to_numpy(float))
                saved.settings.append(run['settings']['models'].get(label, {}))
            if not saved.results:
                print(f'{run["run_id"]}: {name} has no saved single combinations (not_saved_methods?): left out')
                continue
            key = name if name not in experiments else f'{name} [{run["run_id"]}]'
            experiments[key] = saved
    if not experiments:
        raise ValueError('none of the runs has saved single combinations to combine')
    test = pd.DataFrame({'date': test_dates.to_numpy(), 'actual': test_actual.to_numpy(float)})
    return experiments, test


def combine_runs(directory, run_ids, *, day_blocks: dict | None = SIX_HOUR_BLOCKS, timezone: str | None = None,
                 top_n: int | None = None, aggregations=None, include_members: bool = True, save: bool = True,
                 notes: str = '', verbose: bool = True) -> EvaluationResult:
    """The best combination across the experiments of the saved runs `run_ids` (folders in the results
    folder `directory`), and with day_blocks their block-wise selection, chosen on the validation days
    the experiments share and forecast over the test period from their saved forecasts (no refitting).

    day_blocks, timezone: as in evaluate (None or {}: no block-wise selection). top_n, aggregations: as in
    CombinedEnsembles (None: each experiment's own ensemble_top_n and aggregations). include_members: also
    keep each member alone (its saved test forecasts, scored on the shared validation days), so that the
    new run compares the combinations with their members. save: save it as a new run in `directory`
    (runs.csv and the dashboard treat it as any other run)."""
    experiments, test = load_saved_experiments(directory, run_ids)
    start, horizon = 0, len(test)
    names = list(experiments)
    method = ' + '.join(names)
    first = next(iter(experiments.values()))
    sources = list(dict.fromkeys(e.run_id for e in experiments.values()))
    if verbose:
        print(f'Combining {len(experiments)} experiments from {len(sources)} saved runs: {", ".join(names)}\n')

    pool = CombinedEnsembles(experiments, top_n=top_n, aggregations=aggregations)
    if day_blocks:  # the block-wise selection ranks the pool, scoring its blocks in the same pass
        blockwise = BlockSelection(experiments, day_blocks, timezone=timezone, pool=pool).run()
    else:
        pool.run()
    if verbose:
        pool.report()
    result = EvaluationResult(test=test, experiments=experiments)

    if include_members:
        for position, member in enumerate(pool.members):
            alone = next(r for r in pool.results if r.members == (position,))
            label = member.experiment.labels[member.rank - 1]
            result.records.append(_record(
                label, member.method, 'single', list(member.experiment.test_forecasts[member.rank - 1]),
                first.metric, alone.score, pool.number_of_splits, member.experiment.settings[member.rank - 1],
                results_log.pool_validation_forecast(pool, alone)))

    best = pool.results[0]
    settings = {'method': method, 'aggregation': best.aggregation, 'source_runs': sources,
                'members': [{'experiment': pool.members[p].method, 'rank': pool.members[p].rank,
                             'source_run': pool.members[p].experiment.run_id,
                             **pool.members[p].experiment.settings[pool.members[p].rank - 1]}
                            for p in best.members]}
    label = f'All methods: best ({pool.label(best)}{"" if best.size == 1 else ", " + best.aggregation})'
    result.records.append(_record(label, method, 'all methods', pool.rolling_forecasts(test, start, horizon),
                                  first.metric, pool.validation_score(), pool.number_of_splits, settings,
                                  results_log.pool_validation_forecast(pool, best)))

    if day_blocks:
        if verbose:
            print()
        if verbose:
            blockwise.report()
        label = 'Block-wise selection (' + ', '.join(f'{c.block}: {blockwise.label(c.rank)}'
                                                    for c in blockwise.choices) + ')'
        settings = {'method': method, 'source_runs': sources,
                    'blocks': [{'block': str(c.block), 'candidate': blockwise.label(c.rank)}
                               for c in blockwise.choices]}
        result.records.append(_record(label, method, 'block-wise', blockwise.rolling_forecasts(test, start, horizon),
                                      first.metric, blockwise.score, blockwise.pool.number_of_splits, settings,
                                      blockwise.validation_predictions))

    result.forecasts = {r['model']: r['forecast'] for r in result.records}
    if verbose:
        print()
        print(result.table.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    if save:
        n = pool.number_of_splits * first.forecasting_period
        shared = first.data.iloc[-n:]
        result.folder = results_log.save_run(
            ['combined', *names], test['date'], test['actual'], result.records,
            notes=notes or f'combine_runs of {", ".join(sources)}', directory=directory,
            validation_dates=shared['date'], validation_actual=shared['actual'],
            extra_experiment_settings={name: {'source_run': e.run_id} for name, e in experiments.items()})
        if verbose:
            print(f'\nResults saved to {result.folder} (and appended to {Path(directory) / "runs.csv"})')
    return result
