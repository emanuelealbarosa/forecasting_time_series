"""Random forest - one multi-output forest forecasting the whole next day at once (scikit-learn).

On the MLP's inputs (forecasting_time_series/methods/daily_inputs.py): the outcome's profile on each lag day, those of
the lagged exogenous variables, and the whole day-d profile of each variable known in advance; with
lag_resolution 'hourly', every input profile is reduced to its 24 hourly averages.

One forest has one output per step of the day (96 for 15-minute data): each tree splits on the error
summed over every step, and each leaf holds a whole day's profile (the mean of the calibration days in
it). The forecast is therefore always an average of observed days - a forest cannot go beyond the
highest or lowest values it was trained on. Refitted on every forecast day (as LEAR and the MLP).
No scaling is needed: splits do not depend on the scale of the inputs.

Method parameters: those of daily_inputs.py (steps_per_day, lag_days, lag_resolution,
minimum_calibration_days, exog_lag_days, exog_lag_transform), and
  n_estimators     - trees in the forest
  max_depth        - deepest a tree may grow, None = until its leaves are pure or too small
  min_samples_leaf - fewest calibration days in a leaf (larger = smoother forecasts)
  max_features     - inputs considered at each split: a share (0 to 1), or 'sqrt'
  max_samples      - share of the calibration days drawn (with replacement) for each tree, None = all
  seed             - the random seed (makes the forest reproducible)
  n_jobs           - processes building the trees (-1 = every core)
  target_anchor    - None: the forest forecasts the price itself. Otherwise it forecasts the price minus
                     an anchor taken from the known past, and the anchor is added back; the inputs stay
                     price levels (unlike differencing in processing.py, which differences them too):
                       'previous_day'  the same step on d-1   (as differencing at lag steps_per_day)
                       'last_value'    the last step of d-1   (as differencing at lag 1)
                       'previous_mean' the mean of d-1        (the price relative to yesterday's level)
                       'previous_week' the same step on d-7   (as differencing at lag 7 x steps_per_day)
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import daily_inputs
from .daily_inputs import largest_lag, required_history, known_in_advance  # noqa: F401


TARGET_ANCHORS = (None, 'previous_day', 'last_value', 'previous_mean', 'previous_week')


def anchor(days, day, kind):
    """(len(day) x steps) the anchor of each day position in `day`, from the whole days before it."""
    day = np.asarray(day)
    if kind == 'previous_day':
        return days[day - 1]
    if kind == 'last_value':
        return np.repeat(days[day - 1][:, -1:], days.shape[1], axis=1)
    if kind == 'previous_mean':
        return np.repeat(days[day - 1].mean(axis=1, keepdims=True), days.shape[1], axis=1)
    if kind == 'previous_week':
        return days[day - 7]
    raise ValueError(f'unknown target_anchor {kind!r}')


def _whole(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def check_parameters(p):
    """Raise ValueError for a malformed random forest parameter combination."""
    daily_inputs.check_parameters(p)
    if not _whole(p.get('n_estimators'), 1):
        raise ValueError(f'n_estimators must be a whole number of at least 1, got {p.get("n_estimators")!r}')
    if p.get('max_depth') is not None and not _whole(p.get('max_depth'), 1):
        raise ValueError(f'max_depth must be None or a whole number of at least 1, got {p.get("max_depth")!r}')
    if not _whole(p.get('min_samples_leaf'), 1):
        raise ValueError(f'min_samples_leaf must be a whole number of at least 1, got {p.get("min_samples_leaf")!r}')
    features = p.get('max_features')
    if features != 'sqrt' and not (isinstance(features, (int, float)) and not isinstance(features, bool)
                                   and 0 < features <= 1):
        raise ValueError(f"max_features must be a share from above 0 to 1, or 'sqrt'; got {features!r}")
    samples = p.get('max_samples')
    if samples is not None and not (isinstance(samples, float) and 0 < samples <= 1):
        raise ValueError(f'max_samples must be None or a share from above 0 to 1, got {samples!r}')
    if not _whole(p.get('seed'), 0):
        raise ValueError(f'seed must be a whole number of at least 0, got {p.get("seed")!r}')
    jobs = p.get('n_jobs')
    if not (jobs == -1 or _whole(jobs, 1)):
        raise ValueError(f'n_jobs must be -1 (every core) or at least 1, got {jobs!r}')
    if p.get('target_anchor') not in TARGET_ANCHORS:
        raise ValueError(f'target_anchor must be one of {TARGET_ANCHORS}, got {p.get("target_anchor")!r}')
    if p.get('target_anchor') == 'previous_week' and largest_lag(p) < 7:
        raise ValueError('target_anchor previous_week needs lag days reaching back at least 7 days')


class method():

    def __init__(self, method_parameters):
        check_parameters(method_parameters)
        p = method_parameters
        self.inputs = daily_inputs.DailyInputBuilder(p, name='the random forest')
        self.steps_per_day = self.inputs.steps_per_day
        self.n_estimators = p['n_estimators']
        self.max_depth = p['max_depth']
        self.min_samples_leaf = p['min_samples_leaf']
        self.max_features = float(p['max_features']) if p['max_features'] != 'sqrt' else 'sqrt'
        self.max_samples = p['max_samples']
        self.seed = p['seed']
        self.n_jobs = p['n_jobs']
        self.target_anchor = p.get('target_anchor')
        self.n_inputs = None              # inputs of the last fit
        self.feature_importances = None   # each input's importance in the last fit

    # main
    def predict_horizon(self, data, outcome_variable, time_variable, horizon, exog_variables=(), future_exog=None):
        """Fit the forest on `data` (whole days) and forecast the first `horizon` steps of the next day.
        The variables known in advance are needed over the whole next day (their full profile is an input)."""
        same_day = known_in_advance(list(exog_variables or []), self.inputs.exog_lag_days)
        if same_day and horizon < self.steps_per_day:
            raise ValueError(f'the random forest uses the whole day-d profile of {same_day}: forecast whole '
                             f'days (horizon {horizon} < steps_per_day {self.steps_per_day})')
        d = self.inputs.build(data, outcome_variable, time_variable, horizon, exog_variables, future_exog)
        X, x_next = self.inputs.whole_day_design(d)
        forest = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                       max_samples=self.max_samples, bootstrap=True,
                                       random_state=self.seed, n_jobs=self.n_jobs)
        y, offset_next = d.y, 0.0
        if self.target_anchor:
            days = self.inputs.whole_days(data, [outcome_variable], time_variable)[:, :, 0]
            y = d.y - anchor(days, d.targets, self.target_anchor)
            offset_next = anchor(days, [len(days)], self.target_anchor)[0]
        forest.fit(X, y)
        self.n_inputs = X.shape[1]
        self.feature_importances = forest.feature_importances_
        forecast = np.asarray(forest.predict(x_next[None, :]), dtype=float)[0] + offset_next
        return [float(v) for v in forecast[:horizon]]
