"""Separate model selection for different parts of the day.

The day is split into blocks of local hours (e.g. night 0-6, morning 7-12, afternoon 13-18,
evening 19-23). The candidates are those of CombinedEnsembles: every single combination among the
top members of each experiment, and every ensemble of them. Each block gets its own winner - the
candidate with the best score on that block's steps of the validation days - and the forecast of
a day is stitched together: each step comes from the winner of its block.

No refitting is needed for the selection: every candidate's validation forecasts are already
stored. The test forecasts need one rolling forecast per distinct winner (members already
forecast are reused through each experiment's cache). A CombinedEnsembles already run on the same
experiments can be passed as `pool`, so its candidates are not ranked a second time.

Blocks are in local time (timezone), so a block keeps its local hours across clock changes; every
local hour must belong to exactly one block. Scores within a block are always pooled (the metric
over every error of that block's steps on all validation days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .ensemble import ENSEMBLE_AGGREGATIONS, CombinedEnsembles, combine


@dataclass
class BlockChoice:
    block: str
    hours: tuple[int, ...]
    rank: int          # the winner's rank in CombinedEnsembles.results (1 = best over the whole day)
    score: float       # its score on this block
    overall_best_score: float  # the whole-day winner's score on this block, for comparison


class BlockSelection:
    """experiments: one ForecastExperiment, a list, or a dict {name: experiment} (as CombinedEnsembles).
    blocks: {name: local hours}, e.g. {'00-06': range(0, 7), '07-12': range(7, 13), ...}.
    timezone: the blocks' time zone (the time column is read as UTC); None → the times as they are.
    top_n, aggregations: passed to CombinedEnsembles.
    pool: a CombinedEnsembles of these experiments, run or not (then experiments, top_n and aggregations
    are not used): e.g. the one evaluate() ranked for the combination across methods."""

    def __init__(self, experiments, blocks: dict, timezone: str | None = None, top_n: int | None = None,
                 aggregations=None, pool: CombinedEnsembles | None = None):
        if pool is None:
            if not isinstance(experiments, (list, tuple, dict)):
                experiments = [experiments]
            pool = CombinedEnsembles(experiments, top_n=top_n, aggregations=aggregations)
        self.pool = pool
        self.blocks = {name: tuple(int(h) for h in hours) for name, hours in blocks.items()}
        hours = [h for block in self.blocks.values() for h in block]
        if sorted(hours) != list(range(24)):
            raise ValueError(f'every local hour 0-23 must be in exactly one block; got {self.blocks}')
        self.block_of_hour = {h: name for name, block in self.blocks.items() for h in block}
        self.timezone = timezone
        self.choices: list[BlockChoice] = []
        self.validation_predictions: np.ndarray | None = None
        self.score: float | None = None
        self.whole_day_score: float | None = None

    # main
    def run(self) -> 'BlockSelection':
        pool = self.pool
        self.metric = pool.metric
        score = metrics.METRICS[self.metric]
        # every candidate is scored on each block, its forecasts built one at a time from the members'
        # (with ~2^n candidates, holding all their forecasts at once would take gigabytes). A pool not yet
        # ranked is ranked here, its ensembles scored on the blocks while their forecasts are at hand
        visited, layout = {}, []

        def visit(members, aggregation, forecasts):
            if not layout:
                layout.append(self._layout(pool))
            steps, actuals = layout[0]
            candidate = forecasts.ravel()
            visited[members, aggregation] = [score(candidate[where], actuals[where]) for where in steps.values()]

        if not pool.results:
            pool.run(visit=visit)
        steps, actuals = self._layout(pool)
        members = np.array([member.result.split_predictions for member in pool.members], dtype=float)
        scores = {name: np.empty(len(pool.results)) for name in self.blocks}
        for position, result in enumerate(pool.results):
            block_scores = visited.get((result.members, result.aggregation))
            if block_scores is None:  # the members alone, or a pool ranked before
                candidate = self._predictions_of(result, members).ravel()
                block_scores = [score(candidate[where], actuals[where]) for where in steps.values()]
            for name, value in zip(steps, block_scores):
                scores[name][position] = value

        self.choices, stitched = [], np.full_like(actuals, np.nan)
        for name, hours in self.blocks.items():
            best = int(np.argmin(scores[name]))  # ties: the better whole-day rank
            self.choices.append(BlockChoice(name, hours, best + 1, float(scores[name][best]),
                                            float(scores[name][0])))
            winner = self._predictions_of(pool.results[best], members).ravel()
            stitched[steps[name]] = winner[steps[name]]
        self.validation_predictions = stitched  # the stitched validation forecasts, oldest first
        self.score = score(stitched, actuals)
        self.whole_day_score = score(self._predictions_of(pool.results[0], members).ravel(), actuals)
        return self

    def report(self) -> None:
        pool = self.pool
        print(f'Block-wise selection: {len(pool.results)} candidates ({len(pool.members)} single, '
              f'{len(pool.results) - len(pool.members)} ensembles), each block choosing on its own steps of '
              f'the {pool.number_of_splits} validation days (local hours, {self.timezone or "as given"})')
        rows = [{'block': c.block, 'hours': f'{c.hours[0]}-{c.hours[-1]}', 'winner': self.label(c.rank),
                 f'{self.metric} winner': c.score, f'{self.metric} whole-day best': c.overall_best_score,
                 'gain': c.overall_best_score - c.score} for c in self.choices]
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
        print(f'Whole day: block-wise {self.score:,.2f} vs the best whole-day candidate '
              f'({self.label(1)}) {self.whole_day_score:,.2f}')

    def label(self, rank: int) -> str:
        result = self.pool.results[rank - 1]
        return self.pool.label(result) + ('' if result.aggregation == 'single' else f' ({result.aggregation})')

    def rolling_forecasts(self, data: pd.DataFrame, start: int, horizon: int | None = None,
                          n_jobs: int | None = None, verbose: bool | None = None) -> list[float]:
        """Rolling forecasts from row `start` of `data` (see ForecastExperiment.rolling_forecasts):
        each distinct winner is forecast once, then each step is taken from its block's winner."""
        winners = {c.rank for c in self.choices}
        forecasts = {rank: np.asarray(self.pool.rolling_forecasts(data, start, horizon, rank=rank,
                                                                  n_jobs=n_jobs, verbose=verbose), dtype=float)
                     for rank in sorted(winners)}
        length = min(len(f) for f in forecasts.values())
        blocks = self._blocks_of(data[self.time_variable].iloc[start:start + length])
        stitched = np.full(length, np.nan)
        for choice in self.choices:
            mask = blocks == choice.block
            stitched[mask] = forecasts[choice.rank][:length][mask]
        return [float(value) for value in stitched]

    # subsidiary
    def _blocks_of(self, times) -> np.ndarray:
        """The block name of each timestamp, by its local hour."""
        stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(times), utc=self.timezone is not None))
        if self.timezone is not None:
            stamps = stamps.tz_convert(self.timezone)
        return np.array([self.block_of_hour[h] for h in stamps.hour])

    def _layout(self, pool) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Each block's positions among the validation steps of the pool (oldest first), and the
        actual values of those steps."""
        experiment = pool.experiments[0]
        self.time_variable = experiment.time_variable
        n, period = pool.number_of_splits, experiment.forecasting_period
        # the validation days are the last n forecasting periods of the training data, oldest first
        times = experiment.data[self.time_variable].iloc[-n * period:]
        blocks = self._blocks_of(times).reshape(n, period).ravel()
        actuals = np.array(pool.members[0].result.split_actuals, dtype=float).ravel()     # n * period
        steps = {}
        for name in self.blocks:
            steps[name] = np.flatnonzero(blocks == name)
            if not len(steps[name]):
                raise ValueError(f'block {name} has no steps in the validation days')
        return steps, actuals

    @staticmethod
    def _predictions_of(result, members: np.ndarray) -> np.ndarray:
        """The validation forecasts (splits x steps) of one pool result, from the members' (members x splits
        x steps): the member alone, or its members combined step by step."""
        if result.aggregation == 'single':
            return members[result.members[0]]
        return ENSEMBLE_AGGREGATIONS[result.aggregation](members[list(result.members)], axis=0)

    def _validation_predictions(self, result) -> np.ndarray:
        members = [self.pool.members[p].result.split_predictions for p in result.members]
        if result.aggregation == 'single':
            return np.array(members[0], dtype=float)
        return np.array([combine([m[split] for m in members], result.aggregation)
                         for split in range(self.pool.number_of_splits)], dtype=float)
