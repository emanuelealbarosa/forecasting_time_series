"""Saves the results of every run, so that they are never lost, in a results folder given by the caller:

    <results folder>/
      runs.csv                      one row per model tested on the test period, appended after every run
      <run id>/forecasts.csv         the run's test-period forecasts: date, actual, one column per model
      <run id>/validation_forecasts.csv   the same over the validation days (the walk-forward splits the
                                     models were ranked on), so that models from different runs can be
                                     combined or selected afterwards without refitting them
      <run id>/settings.json         the full settings of every model in the run, and of each experiment

runs.csv has the same columns for every method, so rows from different runs line up: a few settings
are flat columns (blank where they do not apply), and every setting is in the `settings` JSON column.
Runs can be combined offline only when they share the validation days and the test period
(validation_start / validation_end and test_start / test_end in runs.csv)."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import ensemble as ensemble_module
from . import grid_search
from . import metrics
from . import processing

# Every function here takes the results folder as an argument: there is no default, since the library's
# own folder (site-packages, once installed) is no place for results.

# the folder holding this package's source, whose git state goes into runs.csv (see _git_state)
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

# settings shown as their own column in runs.csv (blank where a model does not have them)
FLAT_SETTINGS = ('training_window', 'transform', 'decomposition', 'differencing', 'normalisation',
                 'lag_days', 'lag_resolution', 'exog_lag_days', 'n_inputs_day_d', 'penalty',
                 'lstm_units', 'lstm_layers', 'hidden_layers', 'dropout', 'weight_decay', 'learning_rate',
                 'batch_size', 'n_seeds', 'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
                 'target_anchor')

COLUMNS = ('run_id', 'timestamp', 'experiment', 'model', 'kind', 'method',
           'validation_metric', 'validation_score', 'validation_days', 'test_rmse', 'test_mae',
           'test_start', 'test_end', 'test_days', 'fit_seconds', 'environment', *FLAT_SETTINGS,
           'settings', 'forecasts_file', 'git_commit', 'uncommitted_changes', 'notes',
           'validation_start', 'validation_end', 'validation_forecasts_file')


# ------------------------------------------------------------------ validation forecasts
def validation_period(experiment) -> tuple[pd.Series, pd.Series]:
    """(timestamps, actual values) of an experiment's validation days: the last
    number_of_splits forecasting periods of its training data, oldest first."""
    length = experiment.number_of_splits * experiment.forecasting_period
    rows = experiment.data.iloc[-length:]
    return rows[experiment.time_variable], rows[experiment.outcome_variable]


def validation_forecast(experiment, rank: int, ensemble: bool) -> np.ndarray:
    """The validation forecasts of the single combination (or ensemble) at `rank`, all splits in a row."""
    if ensemble:
        chosen = experiment.ensemble_results[rank - 1]
        members, aggregation = chosen.members, chosen.aggregation
    else:
        members, aggregation = (rank,), 'single'
    return _combined_splits([experiment.results[member - 1].split_predictions for member in members], aggregation)


def pool_validation_forecast(pool, result) -> np.ndarray:
    """The validation forecasts of one result of a CombinedEnsembles (on the splits its members share)."""
    return _combined_splits([pool.members[p].result.split_predictions for p in result.members], result.aggregation)


def _combined_splits(member_splits, aggregation) -> np.ndarray:
    if aggregation == 'single':
        return np.concatenate([np.asarray(split, dtype=float) for split in member_splits[0]])
    return np.concatenate([np.asarray(ensemble_module.combine(list(splits), aggregation), dtype=float)
                           for splits in zip(*member_splits)])


# ------------------------------------------------------------------ settings of one model
def single_settings(experiment, rank: int) -> dict:
    """Settings of the single combination at `rank`, with the inputs it uses."""
    chosen = experiment.best_parameters(rank)
    return {'method': experiment.method,
            'method_parameters': chosen['method_parameters'],
            'processing_parameters': chosen['processing_parameters'],
            'processing': processing.describe(**chosen['processing_parameters']),
            'training_window': chosen['training_window'],
            'inputs': _inputs(experiment),
            'fit_seconds': experiment.results[rank - 1].fit_seconds}


def total_fit_seconds(member_settings) -> float | None:
    """The fit seconds of a model made of several single combinations (an ensemble, a combination across
    methods, a block-wise selection): the sum over its distinct members, each refitted once per period.
    None if any member has no timing (saved by an older version)."""
    values = [settings.get('fit_seconds') for settings in member_settings]
    if not values or any(value is None for value in values):
        return None
    return float(sum(values))


def environment() -> str:
    """Where the run is made, for comparing fit seconds: timings are comparable within one environment."""
    place = 'Colab' if 'COLAB_RELEASE_TAG' in os.environ else platform.system()
    return f'{place}, {os.cpu_count()} CPUs'


def _inputs(experiment) -> list[str]:
    exog = experiment.exog_variables
    return [exog] if isinstance(exog, str) else list(exog)


def ensemble_settings(experiment, rank: int) -> dict:
    """Settings of the ensemble at `rank`: how its members are combined, and each member's settings."""
    chosen = experiment.ensemble_results[rank - 1]
    members = [single_settings(experiment, member) for member in chosen.members]
    return {'method': experiment.method, 'aggregation': chosen.aggregation,
            'member_ranks': list(chosen.members), 'members': members,
            'fit_seconds': total_fit_seconds(members)}


def candidate_settings(experiment, rank: int, ensemble: bool) -> dict:
    return ensemble_settings(experiment, rank) if ensemble else single_settings(experiment, rank)


def experiment_settings(experiment) -> dict:
    """What the experiment searched and how it validated: enough to rerun it."""
    train_dates = experiment.data[experiment.time_variable] if experiment.time_variable else None
    return {'method': experiment.method,
            'outcome_variable': experiment.outcome_variable,
            'inputs': _inputs(experiment),
            'method_parameter_ranges': experiment.method_parameter_ranges,
            'processing_ranges': {name: getattr(experiment, name) for name in
                                  ('transform_range', 'decomposition_range', 'differencing_range',
                                   'normalisation_range')},
            'training_window_range': list(experiment.training_window_range),
            'forecasting_period': experiment.forecasting_period,
            'metric': experiment.metric,
            'score_aggregation': experiment.score_aggregation,
            'validation_splits': experiment.number_of_splits,
            'validation_minimum_training_split': experiment.validation_minimum_training_split,
            'validation_maximum_number_of_splits': experiment.validation_maximum_number_of_splits,
            'ensembles': experiment.ensembles, 'ensemble_top_n': experiment.ensemble_top_n,
            'ensemble_aggregations': list(experiment.ensemble_aggregations),
            'training_data': {'rows': len(experiment.data),
                              'start': None if train_dates is None else str(train_dates.iloc[0]),
                              'end': None if train_dates is None else str(train_dates.iloc[-1])},
            'combinations_ranked': len(experiment.results), 'combinations_failed': len(experiment.failed),
            'n_jobs': getattr(experiment, 'n_jobs', None)}


# ------------------------------------------------------------------ saving a run
def save_run(experiment_names, dates, actual, records, experiments=None, notes: str = '', *,
             directory: Path | str, validation_dates=None, validation_actual=None,
             extra_experiment_settings: dict | None = None) -> Path:
    """Save one run in the results folder `directory`: its forecasts and settings in a new folder
    <directory>/<run id>, and one row per model appended to <directory>/runs.csv.

    experiment_names: the EXPERIMENTS keys run.
    dates, actual: the test period's timestamps and actual values.
    records: one dict per model tested, with keys model, experiment, kind ('single', 'ensemble', or
        e.g. 'all methods'), forecast (one value per test step), validation_metric, validation_score,
        validation_days, settings (a dict; see candidate_settings; its fit_seconds goes into runs.csv),
        optionally environment (where its models were fitted; default: here) and validation_forecast
        (see validation_forecast; it may cover only the last validation days, e.g. those shared across
        methods: it is aligned on the last day and left blank before).
    experiments: {name: ForecastExperiment}, whose search settings go into settings.json.
    validation_dates, validation_actual: the validation days' timestamps and actual values (the
        longest validation period of the run; see validation_period).
    extra_experiment_settings: {name: dict} saved in settings.json with those of `experiments`, for runs
        without ForecastExperiments (saved_runs.combine_runs: the run each experiment came from).
    Returns the run's folder."""
    directory = Path(directory)
    started = datetime.now()
    names = list(experiment_names)
    run_id = started.strftime('%Y-%m-%d_%H%M%S') + '_' + _slug('+'.join(names))
    run_directory = directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    dates, actual = list(dates), np.asarray(actual, dtype=float)
    labels = _unique_labels([record['model'] for record in records])
    columns = {'date': [str(date) for date in dates], 'actual': actual}
    for label, record in zip(labels, records):
        values = np.asarray(record['forecast'], dtype=float)
        if len(values) != len(actual):
            raise ValueError(f'{label}: {len(values)} forecasts for {len(actual)} test steps')
        columns[label] = values
    forecasts = pd.DataFrame(columns)  # built at once: one column at a time fragments a wide frame
    forecasts.to_csv(run_directory / 'forecasts.csv', index=False, encoding='utf-8')

    validation_span = {}  # model → (first, last) validation timestamp it has forecasts for
    if validation_dates is not None:
        validation_dates = [str(date) for date in validation_dates]
        length = len(validation_dates)
        columns = {'date': validation_dates, 'actual': np.asarray(validation_actual, dtype=float)}
        for label, record in zip(labels, records):
            values = record.get('validation_forecast')
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if len(values) > length:
                raise ValueError(f'{label}: {len(values)} validation forecasts for {length} validation steps')
            columns[label] = np.concatenate([np.full(length - len(values), np.nan), values])
            validation_span[label] = (validation_dates[length - len(values)], validation_dates[-1])
        pd.DataFrame(columns).to_csv(run_directory / 'validation_forecasts.csv', index=False, encoding='utf-8')

    commit, dirty = _git_state()
    settings = {'run_id': run_id, 'timestamp': started.isoformat(timespec='seconds'),
                'experiments': names, 'git_commit': commit, 'uncommitted_changes': dirty, 'notes': notes,
                'test_period': {'start': str(dates[0]), 'end': str(dates[-1]), 'steps': len(dates)},
                'environment': environment(),
                'experiment_settings': {**{name: experiment_settings(experiment)
                                           for name, experiment in (experiments or {}).items()},
                                        **(extra_experiment_settings or {})},
                'models': {label: record['settings'] for label, record in zip(labels, records)}}
    with open(run_directory / 'settings.json', 'w', encoding='utf-8') as file:
        json.dumps(settings, default=_jsonable)  # fail before writing a partial file
        json.dump(settings, file, indent=2, default=_jsonable)

    steps_per_day = _steps_per_day(dates)
    rows = []
    for label, record in zip(labels, records):
        flat = flat_settings(record['settings'])
        rows.append({'run_id': run_id, 'timestamp': settings['timestamp'],
                     'experiment': record.get('experiment', ''), 'model': label, 'kind': record['kind'],
                     'method': flat.pop('method', ''),
                     'validation_metric': record.get('validation_metric', ''),
                     'validation_score': record.get('validation_score'),
                     'validation_days': record.get('validation_days'),
                     'test_rmse': metrics.rmse(forecasts[label], actual),
                     'test_mae': metrics.mae(forecasts[label], actual),
                     'test_start': str(dates[0]), 'test_end': str(dates[-1]),
                     'test_days': len(dates) / steps_per_day if steps_per_day else None,
                     'fit_seconds': record['settings'].get('fit_seconds'),
                     'environment': record.get('environment') or settings['environment'],
                     **flat,
                     'settings': json.dumps(record['settings'], default=_jsonable),
                     'forecasts_file': f'{run_id}/forecasts.csv',
                     'git_commit': commit, 'uncommitted_changes': dirty, 'notes': notes,
                     **({'validation_start': validation_span[label][0], 'validation_end': validation_span[label][1],
                         'validation_forecasts_file': f'{run_id}/validation_forecasts.csv'}
                        if label in validation_span else {})})
    append_rows(rows, directory)
    return run_directory


def append_rows(rows: list[dict], directory: Path | str) -> None:
    """Append rows to <directory>/runs.csv (created with its header the first time), in the COLUMNS order.
    A runs.csv written with other columns (an older version) is rewritten with the current ones first."""
    path = Path(directory) / 'runs.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if path.exists() and tuple(pd.read_csv(path, nrows=0).columns) != COLUMNS:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        unknown = [c for c in existing.columns if c not in COLUMNS]
        if unknown:
            raise ValueError(f'{path} has columns this version does not know: {unknown}')
        existing.reindex(columns=COLUMNS).to_csv(path, index=False, encoding='utf-8')
    table.to_csv(path, mode='a', header=not path.exists(), index=False, encoding='utf-8')


def load_validation_forecasts(run_id: str, directory: Path | str) -> pd.DataFrame:
    """A run's validation forecasts (in the results folder `directory`): date, actual, one column per model
    (blank where a model has none)."""
    return pd.read_csv(Path(directory) / run_id / 'validation_forecasts.csv')


def load_runs(directory: Path | str) -> pd.DataFrame:
    """Every row of <directory>/runs.csv."""
    return pd.read_csv(Path(directory) / 'runs.csv')


def load_forecasts(run_id: str, directory: Path | str) -> pd.DataFrame:
    """A run's test-period forecasts (in the results folder `directory`): date, actual, one column per model."""
    return pd.read_csv(Path(directory) / run_id / 'forecasts.csv')


# ------------------------------------------------------------------ helpers
def flat_settings(settings: dict) -> dict:
    """The FLAT_SETTINGS of a model, as text. An ensemble shows each setting once if its members
    agree, else every member's value joined by ' | ' (e.g. training_window '16128 | 21504')."""
    if 'members' in settings:
        members = [flat_settings(member) for member in settings['members']]
        flat = {'method': settings.get('method', '')}
        for name in FLAT_SETTINGS:
            values = list(dict.fromkeys(member.get(name, '') for member in members))
            flat[name] = values[0] if len(values) == 1 else ' | '.join(str(value) for value in values)
        return flat
    if 'method_parameters' not in settings:  # e.g. a combination across methods: no common settings
        return {'method': settings.get('method', '')}
    parameters = settings['method_parameters']
    flat = {'method': settings['method'],
            'training_window': _window_label(settings['training_window'], parameters.get('steps_per_day')),
            **settings.get('processing', {})}
    for name in ('lag_days', 'lag_resolution', 'penalty', 'lstm_units', 'lstm_layers', 'hidden_layers', 'dropout',
                 'weight_decay', 'learning_rate', 'batch_size', 'n_seeds', 'n_estimators', 'max_depth',
                 'min_samples_leaf', 'max_features', 'target_anchor'):
        if name in parameters:
            flat[name] = _text(parameters[name])
    if parameters.get('exog_lag_days'):
        flat['exog_lag_days'] = '; '.join(f'{column}: {_text(lags)}'
                                          for column, lags in parameters['exog_lag_days'].items())
    if 'inputs' in settings:
        lagged = set(parameters.get('exog_lag_days') or {})
        flat['n_inputs_day_d'] = sum(column not in lagged for column in settings['inputs'])
    return flat


def _window_label(training_window, steps_per_day) -> str:
    """'expanding', '32w' when the window is whole weeks, else its length in rows."""
    if training_window is not None and steps_per_day and training_window % (7 * steps_per_day) == 0:
        return f'{training_window // (7 * steps_per_day)}w'
    return grid_search.training_window_label(training_window)


def _text(value) -> str:
    if isinstance(value, range):
        value = tuple(value)
    if isinstance(value, (list, tuple)):
        return '(' + ', '.join(str(item) for item in value) + ')'
    return str(value)


def _jsonable(value):
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def _slug(text: str, limit: int = 60) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-')[:limit] or 'run'


def _unique_labels(labels: list[str]) -> list[str]:
    seen, unique = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        unique.append(label if seen[label] == 1 else f'{label} ({seen[label]})')
    return unique


def _steps_per_day(dates) -> int | None:
    try:
        stamps = pd.to_datetime(pd.Series(dates))
        step = (stamps.iloc[1] - stamps.iloc[0])
        return int(pd.Timedelta(days=1) / step)
    except Exception:
        return None


def _github_desktop_git() -> str | None:
    """The git bundled with GitHub Desktop on Windows (when git itself is not on the PATH)."""
    local = os.environ.get('LOCALAPPDATA')
    if not local:
        return None
    found = sorted(Path(local, 'GitHubDesktop').glob('app-*/resources/app/git/cmd/git.exe'))
    return str(found[-1]) if found else None


def _git_state() -> tuple[str | None, bool | None]:
    """(commit hash, whether the code has uncommitted changes), read from .git (git itself is only
    needed for the second; None when it is not available). Changes under results/ do not count."""
    commit = None
    git_directory = PROJECT_DIRECTORY / '.git'
    try:
        head = (git_directory / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head[5:]
            ref_file = git_directory / ref
            if ref_file.exists():
                commit = ref_file.read_text().strip()
            else:
                packed = (git_directory / 'packed-refs').read_text().splitlines()
                commit = next((line.split()[0] for line in packed if line.endswith(' ' + ref)), None)
        else:
            commit = head
    except OSError:
        pass
    git = shutil.which('git') or _github_desktop_git()
    if git is None:
        return commit, None
    try:
        status = subprocess.run([git, 'status', '--porcelain'], cwd=PROJECT_DIRECTORY,
                                capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return commit, None
    changed = [line for line in status.splitlines() if line[3:].strip('"').split('/')[0] != 'results']
    return commit, bool(changed)
