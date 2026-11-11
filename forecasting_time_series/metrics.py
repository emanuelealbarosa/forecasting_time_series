"""Forecast error metrics. Lower is better for all of them.

    rmse : root mean squared error         - same units as the series, penalises large errors more
    mae  : mean absolute error             - same units as the series
    mape : mean absolute percentage error  - in percent (5.0 means 5%); undefined if an actual value is 0
"""
import numpy as np


def rmse(predictions, actual):
    return float(np.sqrt(np.mean((predictions - actual) ** 2)))


def mae(predictions, actual):
    return float(np.mean(np.abs(predictions - actual)))


def mape(predictions, actual):
    if np.any(actual == 0):
        raise ValueError('MAPE is undefined when an actual value is 0')
    return float(np.mean(np.abs((actual - predictions) / actual)) * 100)


METRICS = {'rmse': rmse, 'mae': mae, 'mape': mape}


class evaluator:

    def __init__(self, metrics_parameter):
        self.metrics_type = metrics_parameter.get('type')
        if self.metrics_type not in METRICS:
            raise ValueError(f'unknown metric {self.metrics_type!r}; choose from {list(METRICS)}')

    def score(self, predictions, values):
        """Score predictions against the actual values, pairing them in order. If there are
        fewer predictions than values (the forecast stopped early), only those are scored."""
        predictions = np.asarray(predictions, dtype=float)
        actual = np.asarray(values, dtype=float)[:len(predictions)]
        if len(predictions) == 0 or len(predictions) != len(actual):
            raise ValueError(f'cannot score {len(predictions)} predictions against {len(actual)} actual values')
        return METRICS[self.metrics_type](predictions, actual)
