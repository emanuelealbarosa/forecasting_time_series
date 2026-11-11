"""Preprocessing of the outcome before forecasting, undone on the predictions afterwards.

The order is fixed:   transform → decomposition → differencing → normalisation
and undone in reverse: de-normalise → undo differencing → add back trend and seasonality → undo transform.

Settings (one value of each per parameter combination):
    transform     : None, 'sqrt', 'log', 'asinh', ('power', p), ('asinh', centre)
    decomposition : None, or a dict of MSTL settings (see forecasting_time_series/decomposition.py)
    differencing  : 0 (none), a lag such as 1 or 12, or several lags such as (1, 12)
    normalisation : None, 'minmax', 'mean', 'median'

Every statistic (median, MAD, min, max, ...) is computed on the training data only.
A step that cannot be applied (log of values <= 0, a constant series, too few
observations) raises ValueError; the forecast then counts as failed.
"""
import numpy as np

from . import decomposition as mstl

MAD_TO_SD = 1.4826  # makes the MAD comparable to a standard deviation for normal data

PROCESSING_KEYS = ('transform', 'decomposition', 'differencing', 'normalisation')  # one per step, in order


def location_and_scale(y, kind):
    """Centre and spread used by the normalisations and the asinh transform."""
    if kind == 'median':
        location = np.nanmedian(y)
        scale = np.nanmedian(np.abs(y - location)) * MAD_TO_SD
    elif kind == 'mean':
        location, scale = np.nanmean(y), np.nanstd(y)
    elif kind == 'minmax':
        location = np.nanmin(y)
        scale = np.nanmax(y) - location
    else:
        raise ValueError(f'unknown centring {kind!r}')
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f'cannot scale by {kind}: the series has no spread')
    return location, scale


# ---------------------------------------------------------------------- transforms
class NoStep:
    label = 'none'

    def fit_transform(self, y):
        return y

    def inverse(self, y):
        return y


class PowerTransform:
    """Signed power: sign(y) * |y|**p, so negative values keep their sign."""

    def __init__(self, exponent):
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool) or exponent <= 0:
            raise ValueError(f'power exponent must be a number above 0, got {exponent!r}')
        self.exponent = exponent
        self.label = 'sqrt' if exponent == 0.5 else f'power({exponent:g})'

    def fit_transform(self, y):
        return np.sign(y) * np.abs(y) ** self.exponent

    def inverse(self, y):
        return np.sign(y) * np.abs(y) ** (1 / self.exponent)


class LogTransform:
    label = 'log'

    def fit_transform(self, y):
        if np.nanmin(y) <= 0:
            raise ValueError('log transform needs values above 0')
        return np.log(y)

    def inverse(self, y):
        return np.exp(y)


class AsinhTransform:
    """Variance-stabilising transform: asinh((y - centre) / spread).

    centre 'median' (default) uses median and MAD, 'mean' uses mean and standard
    deviation, None applies asinh to the values as they are."""

    def __init__(self, centre='median'):
        if centre not in ('median', 'mean', None):
            raise ValueError(f"asinh centre must be 'median', 'mean' or None, got {centre!r}")
        self.centre = centre
        self.label = 'asinh' if centre == 'median' else f'asinh({centre or "none"})'
        self.location, self.scale = 0.0, 1.0

    def fit_transform(self, y):
        if self.centre is not None:
            self.location, self.scale = location_and_scale(y, self.centre)
        return np.arcsinh((y - self.location) / self.scale)

    def inverse(self, y):
        return np.sinh(y) * self.scale + self.location


# ---------------------------------------------------------------------- differencing
class Differencing:
    """y[t] - y[t - lag], applied once per lag, in the order given."""

    def __init__(self, lags):
        self.lags = lags
        self.label = ', '.join(str(lag) for lag in lags)
        self.histories = []

    def fit_transform(self, y):
        self.histories = []
        for lag in self.lags:
            if len(y) <= lag:
                raise ValueError(f'too few observations to difference at lag {lag}')
            self.histories.append(y[-lag:].copy())  # last `lag` values before this step
            y = y[lag:] - y[:-lag]
        return y

    def inverse(self, d):
        """`d` continues the differenced series right after the training data."""
        for lag, history in zip(reversed(self.lags), reversed(self.histories)):
            levels = list(history)
            for value in d:
                levels.append(value + levels[-lag])
            d = np.array(levels[lag:], dtype=float)
        return d


# ---------------------------------------------------------------------- normalisation
class Normaliser:
    KINDS = (None, 'minmax', 'mean', 'median')

    def __init__(self, kind):
        if kind not in self.KINDS:
            raise ValueError(f'unknown normalisation {kind!r}; choose from {list(self.KINDS)}')
        self.kind = kind
        self.label = kind or 'none'
        self.location, self.scale = 0.0, 1.0

    def fit_transform(self, y):
        if self.kind is None:
            return y
        self.location, self.scale = location_and_scale(y, self.kind)
        return (y - self.location) / self.scale

    def inverse(self, y):
        return y * self.scale + self.location


# ---------------------------------------------------------------------- building from settings
def build_transform(spec):
    if isinstance(spec, list):
        spec = tuple(spec)
    if spec is None:
        return NoStep()
    if spec == 'sqrt':
        return PowerTransform(0.5)
    if spec == 'log':
        return LogTransform()
    if spec == 'asinh':
        return AsinhTransform('median')
    if isinstance(spec, tuple) and len(spec) == 2:
        name, value = spec
        if name == 'power':
            return PowerTransform(value)
        if name == 'asinh':
            return AsinhTransform(value)
    raise ValueError(f'unknown transform {spec!r}; use None, "sqrt", "log", "asinh", '
                     f'("power", p) or ("asinh", centre)')


def build_differencing(spec):
    if spec is None or spec == 0:
        return NoStep()
    lags = (spec,) if isinstance(spec, int) else tuple(spec)
    if not lags or not all(isinstance(lag, int) and not isinstance(lag, bool) and lag > 0 for lag in lags):
        raise ValueError(f'differencing must be 0, a positive lag or several lags such as (1, 12); got {spec!r}')
    return Differencing(lags)


def build_normaliser(spec):
    return Normaliser(spec)


def build_decomposition(spec):
    return NoStep() if spec is None else mstl.MSTLDecomposition(spec)


def describe(transform, differencing, normalisation, decomposition=None):
    """Readable labels for one combination of processing settings."""
    return {'transform': build_transform(transform).label,
            'decomposition': build_decomposition(decomposition).label,
            'differencing': build_differencing(differencing).label,
            'normalisation': build_normaliser(normalisation).label}


def decomposition_limits(spec):
    """(training rows MSTL needs, rows it keeps): (0, None) without a decomposition."""
    if spec is None:
        return 0, None
    settings = mstl.normalise_spec(spec)
    return mstl.minimum_rows(settings), settings['window']


def rows_lost_to_differencing(differencing):
    """Observations removed from the start of the series by the differencing setting."""
    step = build_differencing(differencing)
    return sum(step.lags) if isinstance(step, Differencing) else 0


def number_of_steps(settings):
    """How many of the steps actually do something (used to prefer simpler settings on ties)."""
    return sum(label != 'none' for label in describe(**settings).values())


# ---------------------------------------------------------------------- pipeline
class data_processer:
    """Applies the steps to the outcome column of a training set and undoes them on predictions."""

    def __init__(self, data, parameters):
        self.data = data.copy(deep=True)
        self.outcome_variable = parameters.get('general_parameters').get('outcome_variable')
        settings = parameters.get('processing_parameters')
        self.steps = [build_transform(settings.get('transform')),
                      build_decomposition(settings.get('decomposition')),
                      build_differencing(settings.get('differencing')),
                      build_normaliser(settings.get('normalisation'))]
        self.new_detransformed_data = None

    def transform(self):
        y = self.data[self.outcome_variable].to_numpy(dtype=float)
        for step in self.steps:
            y = step.fit_transform(y)
        dropped = len(self.data) - len(y)  # the decomposition window and differencing remove the first rows
        self.data = self.data.iloc[dropped:].copy()
        self.data[self.outcome_variable] = y

    def detransform(self, predictions):
        y = np.asarray(predictions, dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):  # overflow → inf → combination fails
            for step in reversed(self.steps):
                y = step.inverse(y)
        self.new_detransformed_data = y.tolist()
