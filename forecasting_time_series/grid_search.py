"""Evaluates every parameter combination with walk-forward validation, optionally in parallel.

Every combination is scored on the same validation splits, so scores are comparable:
  - the number of splits is planned for the most demanding combination that can run at all
    (the one needing the most training history, including rows lost to differencing);
  - combinations that cannot run even with a single split are marked as failed up front;
  - a combination is ranked only if it succeeds on every split.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import grid_builder
from . import method as method_registry
from . import metrics
from . import processing
from . import validation

# How the validation splits are turned into one score:
#   'pooled' - the metric computed once over every error of every split (an RMSE stays a true RMSE)
#   'mean' / 'median' - the metric computed per split, then averaged
SCORE_AGGREGATIONS = ('pooled', 'mean', 'median')


def aggregate_score(aggregation, metric, split_scores, split_predictions, split_actuals) -> float:
    """One score for a set of validation splits."""
    if aggregation == 'pooled':
        predictions = [value for split in split_predictions for value in split]
        actual = [value for split in split_actuals for value in split]
        return metrics.evaluator({'type': metric}).score(predictions, actual)
    return float({'mean': np.mean, 'median': np.median}[aggregation](split_scores))


def training_window_label(training_window) -> str:
    return 'expanding' if training_window is None else str(training_window)


@dataclass
class SearchResult:
    """Outcome of one parameter combination."""
    index: int                    # position in the grid (last tie-breaker, keeps runs reproducible)
    method_parameters: dict
    processing_parameters: dict
    training_window: int | None = None  # None = expanding window; a number = fixed training length
    required_history: int = 0     # training rows needed: method's own need + rows lost to differencing
    split_scores: list[float] = field(default_factory=list)  # one per validation split that succeeded
    split_predictions: list[list[float]] = field(default_factory=list)  # their forecasts (for ensembles)
    split_actuals: list[list[float]] = field(default_factory=list)      # and the actual values
    split_seconds: list[float] = field(default_factory=list)  # wall seconds to fit and forecast each split
    score: float | None = None    # split_scores combined; None unless every split succeeded
    errors: list[str] = field(default_factory=list)          # why splits failed, if any did

    @property
    def fit_seconds(self) -> float | None:
        """Median wall seconds to fit and forecast one validation split (one forecasting period): what
        the combination costs each time it is refitted. None if no split was timed."""
        return float(np.median(self.split_seconds)) if self.split_seconds else None

    @property
    def is_valid(self) -> bool:
        return self.score is not None and bool(np.isfinite(self.score))

    @property
    def number_of_processing_steps(self) -> int:
        return processing.number_of_steps(self.processing_parameters)


@dataclass
class SplitPlan:
    """How the shared validation splits were chosen."""
    number_of_splits: int           # splits every combination is scored on
    allowed_by_training_share: int  # what validation_minimum_training_split alone would allow
    meets_training_share: bool      # False → even one split is below the minimum training share
    required_history: int           # training rows needed by the most demanding feasible combination
    constraining: list[dict] = field(default_factory=list)  # combinations that reduced the splits


class GridSearch:
    """Runs walk-forward validation on every combination of method parameters, processing
    settings and training window (expanding, or fixed lengths).

    n_jobs: 1 runs in this process; n > 1 uses n worker processes; -1 uses every CPU core."""

    def __init__(self, data, method, general_parameters, parameter_ranges,
                 score_aggregation='pooled', n_jobs=1, verbose=True):
        if score_aggregation not in SCORE_AGGREGATIONS:
            raise ValueError(f'score_aggregation must be one of {list(SCORE_AGGREGATIONS)}')
        self.data = data
        self.method = method
        self.general_parameters = general_parameters
        self.score_aggregation = score_aggregation
        self.grid = grid_builder.builder(method, parameter_ranges).get_grid_of_parameter_combinations()
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.plan: SplitPlan | None = None
        self.results: list[SearchResult] = []   # ranked, best first
        self.failed: list[SearchResult] = []    # not ranked, in grid order

    # main
    def run(self) -> GridSearch:
        candidates = [self._new_result(index, combination) for index, combination in enumerate(self.grid)]
        longest_training = len(self.data) - self.general_parameters['forecasting_period']
        for result in candidates:
            if result.errors:
                continue  # already ruled out (decomposition window too short for the method)
            if result.required_history > longest_training:
                result.errors = [f'needs more training history than the longest validation split has '
                                 f'({result.required_history} rows needed, {longest_training} available)']
        feasible = [r for r in candidates if not r.errors]
        infeasible = [r for r in candidates if r.errors]

        self.plan = self._plan_splits(feasible)
        evaluated = self._evaluate_all(feasible)
        self.results = sorted((r for r in evaluated if r.is_valid), key=self._ranking_key)
        self.failed = sorted([r for r in evaluated if not r.is_valid] + infeasible, key=lambda r: r.index)
        return self

    # subsidiary
    def _new_result(self, index, combination) -> SearchResult:
        method_parameters = dict(combination['method_parameters'])
        processing_parameters = dict(combination['processing_parameters'])
        # rows the method needs after preprocessing: its own history plus the rows differencing removes
        needed_after_decomposition = (method_registry.required_history(self.method, method_parameters)
                                      + processing.rows_lost_to_differencing(processing_parameters['differencing']))
        # MSTL needs more than two cycles of its longest period, and keeps only its window
        mstl_minimum, mstl_window = processing.decomposition_limits(processing_parameters.get('decomposition'))
        # a fixed training window: every split trains on exactly that many rows, so they must all exist
        training_window = combination.get('training_window')
        result = SearchResult(index, method_parameters, processing_parameters, training_window=training_window,
                              required_history=max(needed_after_decomposition, mstl_minimum, training_window or 0))
        if training_window is not None and max(needed_after_decomposition, mstl_minimum) > training_window:
            result.errors = [f'needs more history than the training window keeps '
                             f'({max(needed_after_decomposition, mstl_minimum)} rows needed, '
                             f'training window {training_window})']
        elif mstl_window is not None and needed_after_decomposition > min(mstl_window, training_window or mstl_window):
            result.errors = [f'needs more history than the decomposition window keeps '
                             f'({needed_after_decomposition} rows needed, window {mstl_window})']
        return result

    def _plan_splits(self, feasible: list[SearchResult]) -> SplitPlan:
        settings = (len(self.data), self.general_parameters['forecasting_period'],
                    self.general_parameters['validation_maximum_number_of_splits'],
                    self.general_parameters['validation_minimum_training_split'])
        allowed, meets_share = validation.plan_splits(*settings)
        required = max((r.required_history for r in feasible), default=0)
        number_of_splits, _ = validation.plan_splits(*settings, required_history=required)
        constraining = []
        if number_of_splits < allowed:
            constraining = [{**r.method_parameters, **processing.describe(**r.processing_parameters),
                             'training_window': training_window_label(r.training_window)}
                            for r in feasible if r.required_history == required]
        return SplitPlan(number_of_splits, allowed, meets_share, required, constraining)

    def _evaluate_all(self, results: list[SearchResult]) -> list[SearchResult]:
        context = {'data': self.data, 'method': self.method,
                   'general_parameters': {**self.general_parameters,
                                          'validation_number_of_splits': self.plan.number_of_splits},
                   'score_aggregation': self.score_aggregation}
        workers = self._number_of_workers(len(results))
        if workers == 1:
            return self._collect((_evaluate(context, r) for r in results), len(results))
        # each worker receives the data once (initializer), then only small tasks
        chunksize = max(1, len(results) // (workers * 4))
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(context,)) as pool:
            return self._collect(pool.map(_evaluate_in_worker, results, chunksize=chunksize), len(results))

    def _number_of_workers(self, number_of_tasks: int) -> int:
        workers = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        return max(1, min(workers, number_of_tasks))

    def _collect(self, outcomes, total: int) -> list[SearchResult]:
        collected = []
        report_every = max(1, total // 10)
        in_terminal = sys.stdout.isatty()  # overwrite one line in a terminal, else one line per update
        for done, outcome in enumerate(outcomes, start=1):
            collected.append(outcome)
            if self.verbose and (done % report_every == 0 or done == total):
                message = f'Evaluated {done}/{total} combinations'
                print(f'\r{message}' if in_terminal else message, end='' if in_terminal else '\n', flush=True)
        if self.verbose:
            print('\n' if in_terminal else '')
        return collected

    @staticmethod
    def _ranking_key(result: SearchResult):
        # scores equal to 10 significant digits count as ties; ties go to the simpler settings
        return (float(f'{result.score:.10g}'), result.number_of_processing_steps, result.index)


# ---------------------------------------------------------------------- evaluation of one combination
# Module-level functions, so worker processes can run them.

def _evaluate(context: dict, result: SearchResult) -> SearchResult:
    parameters = {'method': context['method'],
                  'general_parameters': context['general_parameters'],
                  'method_parameters': result.method_parameters,
                  'processing_parameters': result.processing_parameters,
                  'training_window': result.training_window}
    try:
        validator = validation.validation_agent(context['data'], parameters)
        validator.validate()
    except Exception as error:  # an unexpected failure ends this combination, not the whole search
        result.errors = [f'{type(error).__name__}: {error}']
        return result
    result.split_scores = [float(score) for score in validator.errors]
    result.split_predictions = validator.split_predictions
    result.split_actuals = validator.split_actuals
    result.split_seconds = validator.split_seconds
    result.errors = validator.failure_reasons
    if len(result.split_scores) == context['general_parameters']['validation_number_of_splits']:
        result.score = aggregate_score(context['score_aggregation'],
                                       context['general_parameters']['metrics']['type'],
                                       result.split_scores, result.split_predictions, result.split_actuals)
    return result


_WORKER_CONTEXT = None


def _init_worker(context: dict) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _evaluate_in_worker(result):
    return _evaluate(_WORKER_CONTEXT, result)
