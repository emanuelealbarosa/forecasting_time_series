"""LEAR - Lasso-Estimated AutoRegressive model (Lago, Marcjasz, De Schutter & Weron, 2021).

One linear model per step of the day (24 for hourly data, 96 for 15-minute data). The model
for step h forecasts step h of the next day (day d) from:
  - the full profile of the outcome on each lag day (default d-1, d-2, d-3 and d-7),
  - the exogenous variables known in advance, at step h of day d - such as the
    day-of-week dummies built by forecasting_time_series/calendar_features.py, and
  - the full profiles of lagged exogenous variables on their lag days (exog_lag_days) - variables
    known only up to the previous day, such as another zone's prices from the same auction.
These inputs are built by forecasting_time_series/methods/daily_inputs.py, shared with the MLP.
Each model is fitted with the Lasso, which keeps only the useful inputs; the penalty is chosen
per model (AIC by default). All steps of day d are forecast at once from data up to the end of
day d-1 (direct multi-step forecasting, not step-by-step recursion).

The input set is fixed by design: every candidate input goes in, and the Lasso decides which
ones matter. Only the penalty rule is searched (penalty_range).

The framework supplies the rest of the original benchmark:
  - asinh with median/MAD scaling       → transform_range=('asinh',)
  - the calibration window (in days)     → training_window_range, in observations (28 days = 28 * 96)
  - daily recalibration                  → walk-forward validation with forecasting_period = one day
  - averaging several calibration windows → ensembles

Requirements: the training data are whole, consecutive days (a leading partial day is dropped),
and a forecast covers at most one day, starting at the first step of the day.

Method parameters: those of daily_inputs.py (steps_per_day, lag_days, lag_resolution,
minimum_calibration_days, exog_lag_days, exog_lag_transform), and
  lag_interactions         - known-in-advance exogenous columns (e.g. the day-of-week dummies) whose
                             value at step h of day d multiplies each outcome lag input: with the
                             weekday dummies, the weight of each price lag can differ by weekday
                             (len(columns) × the outcome lag inputs more inputs)
  penalty (searched)       - 'aic' / 'bic': Lasso-LARS with that information criterion;
                             'cv': 5-fold time-ordered cross-validation (much slower);
                             a number: that fixed Lasso penalty (alpha)
"""
import warnings

import numpy as np
import scipy.linalg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.model_selection import TimeSeriesSplit

from . import daily_inputs
from .daily_inputs import largest_lag, required_history, known_in_advance  # noqa: F401

PENALTIES = ('aic', 'bic', 'cv')


def check_parameters(p):
    """Raise ValueError for a malformed LEAR parameter combination."""
    daily_inputs.check_parameters(p)
    penalty = p.get('penalty')
    if penalty not in PENALTIES and (isinstance(penalty, bool) or not isinstance(penalty, (int, float))
                                     or penalty <= 0):
        raise ValueError(f'penalty must be one of {PENALTIES} or a number above 0, got {penalty!r}')
    interactions = p.get('lag_interactions') or ()
    if isinstance(interactions, str) or not all(isinstance(c, str) for c in interactions):
        raise ValueError(f"lag_interactions must be a tuple of column names such as ('dow_mon', 'dow_tue'), "
                         f'got {interactions!r}')


class method():

    def __init__(self, method_parameters):
        check_parameters(method_parameters)
        self.inputs = daily_inputs.DailyInputBuilder(method_parameters, name='LEAR')
        self.steps_per_day = self.inputs.steps_per_day
        self.penalty = method_parameters['penalty']
        self.lag_interactions = tuple(method_parameters.get('lag_interactions') or ())
        self.inputs_kept = []  # inputs with a non-zero coefficient, per model (after the last fit)
        self.lars_fallbacks = 0  # fits refitted on an independent subset of inputs after LARS broke down

    # main
    def predict_horizon(self, data, outcome_variable, time_variable, horizon, exog_variables=(), future_exog=None):
        """Fit the per-step models on `data` (whole days) and forecast the first `horizon`
        steps of the next day. `future_exog` holds the exogenous variables known in advance
        over those steps; lagged-only variables (exog_lag_days without lag 0) are not used from it."""
        same_day = known_in_advance(list(exog_variables or []), self.inputs.exog_lag_days)
        not_known = [c for c in self.lag_interactions if c not in same_day]
        if not_known:
            raise ValueError(f'lag_interactions lists {not_known}, which are not exogenous variables known in advance')
        d = self.inputs.build(data, outcome_variable, time_variable, horizon, exog_variables, future_exog)
        interacting = [d.same_day_columns.index(c) for c in self.lag_interactions]
        n = len(d.targets)

        self.inputs_kept = []
        forecasts = []
        for step in range(horizon):
            # the lag inputs are shared by every step's model; the same-day inputs are those of its own step
            X = np.hstack([d.lags, d.same_day[:, step, :]])
            x_next = np.concatenate([d.lags_next, d.same_day_next[step]])
            if interacting:  # each outcome lag input times each interacting column at this step
                by = d.same_day[:, step][:, interacting]                              # (days, columns)
                X = np.hstack([X, (d.lags[:, :d.outcome_inputs, None] * by[:, None, :]).reshape(n, -1)])
                x_next = np.concatenate([x_next, np.outer(d.lags_next[:d.outcome_inputs],
                                                          d.same_day_next[step, interacting]).ravel()])
            # a small model for the noise variance: the same step on each lag day, plus the same-day inputs
            X_small = np.column_stack([d.small[:, step, :], d.same_day[:, step, :]])
            forecasts.append(self.fit_one(X, d.y[:, step], x_next, X_small))
        return forecasts

    # subsidiary
    @staticmethod
    def noise_variance(X_small, y):
        """Residual variance of a least-squares fit of y on a few inputs, per degree of freedom.
        Needed by AIC/BIC when there are fewer calibration days than inputs (scikit-learn cannot
        estimate it from the full model then). Inputs constant over the days are left out."""
        X_small = X_small[:, X_small.std(axis=0) > 0]
        X_small = np.column_stack([np.ones(len(y)), X_small])
        if len(y) - X_small.shape[1] < 2:  # too few days for the small model: the same step of day d-1 only
            X_small = X_small[:, :2]
        coefficients, *_ = np.linalg.lstsq(X_small, y, rcond=None)
        residuals = y - X_small @ coefficients
        degrees_of_freedom = len(y) - np.linalg.matrix_rank(X_small)
        return max(float(residuals @ residuals) / max(degrees_of_freedom, 1), 1e-12)

    @staticmethod
    def independent_columns(X, tolerance=1e-8):
        """Positions (in order) of a linearly independent subset of X's columns: QR with column
        pivoting, keeping the columns whose pivot is above `tolerance` times the largest."""
        _, R, pivots = scipy.linalg.qr(X, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(R))
        rank = int(np.sum(diagonal > tolerance * diagonal[0])) if len(diagonal) else 0
        return np.sort(pivots[:rank])

    def fit_one(self, X, y, x_next, X_small):
        """Fit the Lasso for one step of the day and forecast that step of the next day."""
        # an input constant over the calibration days (e.g. a weekday dummy never seen, or the scaled
        # solar actuals at night) is left out. Relative tolerance: a constant column's computed spread
        # can be rounding noise (~1e-16) rather than exactly 0, and standardising by it breaks the fit
        spread = X.std(axis=0)
        varying = np.isfinite(spread) & (spread > 1e-9 * (1 + np.abs(X).max(axis=0)))
        X, x_next = X[:, varying], x_next[varying]
        # standardise the inputs on the calibration days (the Lasso penalty is scale-dependent)
        mean, scale = X.mean(axis=0), X.std(axis=0)
        X, x_next = (X - mean) / scale, (x_next - mean) / scale
        # limited extrapolation: each forecast-day input may go beyond the range it had on the calibration
        # days by at most that range again. Without this, an input almost always equal in calibration (e.g.
        # the solar forecast at night, nearly always 0) turns a small new value into a huge standardised one
        low, high = X.min(axis=0), X.max(axis=0)
        x_next = np.clip(x_next, low - (high - low), high + (high - low))
        # identical inputs (e.g. several night-time solar forecasts, all 0 but on the same day) add no
        # information and make the LARS path degenerate: keep one of each
        _, first = np.unique(X.round(12), axis=1, return_index=True)
        keep = np.sort(first)
        X, x_next = X[:, keep], x_next[keep]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            if self.penalty in ('aic', 'bic'):
                options = {}
                if len(X) <= X.shape[1] + 1:
                    # fewer days than inputs: scikit-learn cannot estimate the noise variance itself
                    options['noise_variance'] = self.noise_variance(X_small, y)
                try:
                    model = LassoLarsIC(criterion=self.penalty, max_iter=2500, **options).fit(X, y)
                except ValueError:
                    # LARS can break down when some inputs are (almost) linear combinations of others
                    # ("shapes (161,7) and (6,) not aligned"): refit on a linearly independent subset
                    independent = self.independent_columns(X)
                    X, x_next = X[:, independent], x_next[independent]
                    self.lars_fallbacks += 1
                    model = LassoLarsIC(criterion=self.penalty, max_iter=2500, **options).fit(X, y)
            elif self.penalty == 'cv':
                folds = min(5, len(X) - 1)
                model = LassoCV(cv=TimeSeriesSplit(folds), max_iter=2500, n_alphas=30).fit(X, y)
            else:
                model = Lasso(alpha=float(self.penalty), max_iter=2500).fit(X, y)
        self.inputs_kept.append(int(np.count_nonzero(model.coef_)))
        return float(model.predict(x_next.reshape(1, -1))[0])
