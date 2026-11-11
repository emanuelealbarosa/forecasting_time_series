"""Ensembles of the best single combinations.

Every ranked combination was scored on the same validation splits, so their validation
forecasts can be combined step by step and scored on those same splits - no refitting needed.

The members are the top n *distinct* combinations: a combination whose validation forecasts
are identical to those of a better-ranked one (e.g. the same settings with a training window
the method never looks beyond) is skipped, so an ensemble never averages a forecast with a copy
of itself. Members keep their rank in the full ranking (#1, #3, … can be the first two members).

With n members, every subset of 2 or more is tested with each aggregation.
For n = 3: #a+#b, #a+#c, #b+#c, #a+#b+#c. The median of two forecasts equals their mean,
so pairs are tested with the mean only when the mean is among the aggregations.

CombinedEnsembles does the same across several experiments (one per method): the top members of
each are pooled, and every member alone and every subset of 2 or more is scored on the validation
splits the experiments share.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from . import metrics
from .grid_search import aggregate_score

ENSEMBLE_AGGREGATIONS = {'mean': np.mean, 'median': np.median}


@dataclass
class EnsembleResult:
    members: tuple[int, ...]   # ranks of the single combinations combined (1 = best single)
    aggregation: str           # how their forecasts are combined at each step: 'mean' or 'median'
    split_scores: list[float] = field(default_factory=list)
    score: float | None = None

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def label(self) -> str:
        return ' + '.join(f'#{rank}' for rank in self.members)


def candidate_ensembles(number_of_members: int, aggregations) -> list[tuple[tuple[int, ...], str]]:
    """Every subset of 2+ ranks out of 1..number_of_members, with each aggregation."""
    candidates = []
    for size in range(2, number_of_members + 1):
        for members in itertools.combinations(range(1, number_of_members + 1), size):
            for aggregation in aggregations:
                if size == 2 and aggregation == 'median' and 'mean' in aggregations:
                    continue  # the median of two values is their mean
                candidates.append((members, aggregation))
    return candidates


def combine(forecasts, aggregation: str) -> np.ndarray:
    """Combine several forecasts step by step (cut to the shortest, if one stopped early)."""
    length = min(len(forecast) for forecast in forecasts)
    stacked = np.array([forecast[:length] for forecast in forecasts], dtype=float)
    return ENSEMBLE_AGGREGATIONS[aggregation](stacked, axis=0)


def same_forecasts(a, b) -> bool:
    """True if two results made the same validation forecasts on every split."""
    return len(a.split_predictions) == len(b.split_predictions) and all(
        len(p) == len(q) and np.allclose(p, q, rtol=1e-9, atol=1e-9)
        for p, q in zip(a.split_predictions, b.split_predictions))


def distinct_members(ranked, top_n: int) -> list[tuple[int, object]]:
    """The `top_n` best results with distinct validation forecasts, as (rank, result) pairs;
    a result identical to a better-ranked one is skipped."""
    chosen = []
    for rank, result in enumerate(ranked, start=1):
        if any(same_forecasts(result, other) for _, other in chosen):
            continue
        chosen.append((rank, result))
        if len(chosen) == top_n:
            break
    return chosen


def evaluate_ensembles(members_available, aggregations, metric: str,
                       score_aggregation: str) -> list[EnsembleResult]:
    """Score every ensemble of the given (rank, result) members; best first."""
    scorer = metrics.evaluator({'type': metric})
    number_of_splits = len(members_available[0][1].split_scores) if members_available else 0
    results = []
    for positions, aggregation in candidate_ensembles(len(members_available), aggregations):
        members = tuple(members_available[p - 1][0] for p in positions)  # ranks in the full ranking
        chosen = [members_available[p - 1][1] for p in positions]
        split_predictions, split_actuals = [], []
        for split in range(number_of_splits):
            combined = combine([m.split_predictions[split] for m in chosen], aggregation)
            split_predictions.append(list(combined))
            split_actuals.append(chosen[0].split_actuals[split][:len(combined)])
        split_scores = [scorer.score(p, a) for p, a in zip(split_predictions, split_actuals)]
        score = aggregate_score(score_aggregation, metric, split_scores, split_predictions, split_actuals)
        results.append(EnsembleResult(members, aggregation, split_scores, score))
    order = list(ENSEMBLE_AGGREGATIONS)
    # ties go to smaller ensembles, then to the mean, then to better-ranked members
    return sorted(results, key=lambda r: (float(f'{r.score:.10g}'), r.size,
                                          order.index(r.aggregation), r.members))


# ---------------------------------------------------------------------- ensembles across methods
@dataclass
class CombinedMember:
    """One single combination from one experiment, offered to the cross-method search."""
    method: str
    rank: int              # its rank in its own experiment
    result: object         # its SearchResult, cut to the validation splits shared by every experiment
    experiment: object     # the ForecastExperiment it comes from (used to forecast it)

    @property
    def label(self) -> str:
        return f'{self.method} #{self.rank}'


@dataclass
class CombinedResult:
    members: tuple[int, ...]   # positions in CombinedEnsembles.members
    aggregation: str           # 'single' (one member alone), 'mean' or 'median'
    split_scores: list[float] = field(default_factory=list)
    score: float | None = None

    @property
    def size(self) -> int:
        return len(self.members)


class CombinedEnsembles:
    """Ensembles of the best combinations of several experiments (one per method), e.g. Naive and LEAR.

    From each experiment, its top `ensemble_top_n` distinct combinations become members (a member
    with the same validation forecasts as one already chosen is skipped). Every member alone, and
    every subset of 2 or more members - within one method or across methods - combined with each
    aggregation, is scored on the validation splits that all the experiments share, and ranked.

    The experiments must forecast the same training data with the same forecast length, metric and
    score aggregation. Their validation splits are counted back from the end of the data, so they
    coincide: if one experiment used fewer splits, only the splits common to all are used, and
    single scores here can differ from the experiments' own.

    top_n: members per experiment (None → each experiment's ensemble_top_n; an int → the same for all).
    aggregations: None → those of the first experiment."""

    def __init__(self, experiments, top_n: int | None = None, aggregations=None):
        """experiments: a list (members are labelled by method) or a dict {name: experiment}
        (labelled by name, e.g. two LEAR experiments 'LEAR' and 'LEAR+HU')."""
        if isinstance(experiments, dict):
            self.names, self.experiments = list(experiments), list(experiments.values())
        else:
            self.experiments = list(experiments)
            self.names = [e.method for e in self.experiments]
        if len(set(self.names)) < len(self.names):  # same method twice in a list: number them
            self.names = [f'{name} ({i})' for i, name in enumerate(self.names, start=1)]
        if not self.experiments:  # one experiment is allowed: its members alone and their ensembles
            raise ValueError('give at least 1 experiment')
        self.top_n = top_n
        self.aggregations = tuple(aggregations or self.experiments[0].ensemble_aggregations)
        unknown = [a for a in self.aggregations if a not in ENSEMBLE_AGGREGATIONS]
        if unknown:
            raise ValueError(f'aggregations must be chosen from {list(ENSEMBLE_AGGREGATIONS)}; got {unknown}')
        first = self.experiments[0]
        self.metric, self.score_aggregation = first.metric, first.score_aggregation
        self.members: list[CombinedMember] = []
        self.results: list[CombinedResult] = []   # best first
        self.number_of_splits: int | None = None

    # main
    def run(self) -> 'CombinedEnsembles':
        self._check_compatible()
        self.number_of_splits = min(e.number_of_splits for e in self.experiments)
        n = self.number_of_splits
        self.members = []
        for name, experiment in zip(self.names, self.experiments):
            top_n = self.top_n if self.top_n is not None else experiment.ensemble_top_n
            for rank, result in distinct_members(experiment.results, top_n):
                trimmed = replace(result, split_scores=result.split_scores[-n:],
                                  split_predictions=result.split_predictions[-n:],
                                  split_actuals=result.split_actuals[-n:])
                if any(same_forecasts(trimmed, member.result) for member in self.members):
                    continue
                self.members.append(CombinedMember(name, rank, trimmed, experiment))
        self._check_same_actuals()

        results = [CombinedResult((position,), 'single', list(member.result.split_scores),
                                  aggregate_score(self.score_aggregation, self.metric, member.result.split_scores,
                                                  member.result.split_predictions, member.result.split_actuals))
                   for position, member in enumerate(self.members)]
        if len(self.members) >= 2:
            available = [(position, member.result) for position, member in enumerate(self.members)]
            results += [CombinedResult(e.members, e.aggregation, e.split_scores, e.score)
                        for e in evaluate_ensembles(available, self.aggregations, self.metric,
                                                    self.score_aggregation)]
        order = ['single'] + list(ENSEMBLE_AGGREGATIONS)
        # ties go to fewer members, then single → mean → median, then earlier members
        self.results = sorted(results, key=lambda r: (float(f'{r.score:.10g}'), r.size,
                                                      order.index(r.aggregation), r.members))
        return self

    def label(self, result: CombinedResult) -> str:
        return ' + '.join(self.members[p].label for p in result.members)

    def table(self, n: int | None = None):
        import pandas as pd
        rows = self.results[:n] if n is not None else self.results
        table = pd.DataFrame([{'members': self.label(r),
                               'methods': ' + '.join(dict.fromkeys(self.members[p].method for p in r.members)),
                               'size': r.size, 'aggregation': r.aggregation, self.metric: r.score}
                              for r in rows])
        table.index = pd.RangeIndex(1, len(table) + 1, name='rank')
        return table

    def report(self, n: int = 10) -> None:
        methods = ', '.join(self.names)
        members = ', '.join(m.label for m in self.members)
        ensembles = sum(r.size > 1 for r in self.results)
        print(f'Combinations across methods ({methods}): {len(self.members)} members ({members}); '
              f'{len(self.members)} alone and {ensembles} ensembles '
              f'({" and ".join(self.aggregations)}), on the {self.number_of_splits} validation splits '
              f'they share, ranked by {self.score_aggregation} {self.metric}')
        print(self.table(n).reset_index().to_string(index=False, formatters={self.metric: lambda x: f'{x:,.2f}'}))
        best = self.results[0]
        for method in dict.fromkeys(m.method for m in self.members):
            alone = next(r for r in self.results if r.size == 1 and self.members[r.members[0]].method == method)
            print(f'  best {method} alone: {self.label(alone)} {alone.score:,.2f}')
        print(f'Best overall: {self.label(best)} ({best.aggregation}) {best.score:,.2f}')

    def validation_score(self, rank: int = 1) -> float:
        return self._result_at(rank).score

    def rolling_forecasts(self, data, start: int, horizon: int | None = None, rank: int = 1,
                          n_jobs: int | None = None, verbose: bool | None = None) -> list[float]:
        """Rolling forecasts (see ForecastExperiment.rolling_forecasts) of the combination at `rank`:
        each member is forecast by its own experiment (reusing forecasts it already made), then
        the members are combined step by step."""
        result = self._result_at(rank)
        forecasts = [self.members[p].experiment.rolling_forecasts(data, start, horizon, rank=self.members[p].rank,
                                                                  n_jobs=n_jobs, verbose=verbose)
                     for p in result.members]
        if result.aggregation == 'single':
            return list(forecasts[0])
        return [float(value) for value in combine(forecasts, result.aggregation)]

    # subsidiary
    def _result_at(self, rank: int) -> CombinedResult:
        if not self.results:
            raise RuntimeError('no results: call run() first')
        if not 1 <= rank <= len(self.results):
            raise ValueError(f'rank must be between 1 and {len(self.results)}')
        return self.results[rank - 1]

    def _check_compatible(self) -> None:
        first = self.experiments[0]
        for experiment in self.experiments:
            if experiment.split_plan is None or not experiment.results:
                raise ValueError(f'the {experiment.method} experiment has no ranked results: run it first')
            for setting in ('metric', 'score_aggregation', 'forecasting_period', 'outcome_variable'):
                if getattr(experiment, setting) != getattr(first, setting):
                    raise ValueError(f'experiments must share {setting}: {first.method} has '
                                     f'{getattr(first, setting)!r}, {experiment.method} '
                                     f'{getattr(experiment, setting)!r}')
            same_data = len(experiment.data) == len(first.data) and np.array_equal(
                experiment.data[first.outcome_variable].to_numpy(dtype=float),
                first.data[first.outcome_variable].to_numpy(dtype=float), equal_nan=True)
            if not same_data:
                raise ValueError(f'experiments must use the same data: {first.method} and {experiment.method} differ')

    def _check_same_actuals(self) -> None:
        reference = self.members[0].result.split_actuals
        for member in self.members[1:]:
            if not all(np.allclose(a, b) for a, b in zip(member.result.split_actuals, reference)):
                raise ValueError(f'{member.label} was validated on different days from {self.members[0].label}')
