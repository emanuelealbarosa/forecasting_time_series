"""LSTM - a recurrent network reading the past days in order, then forecasting the whole next day (PyTorch).

Where the MLP sees every input profile side by side, the LSTM reads the lag days one at a time,
oldest first: each lag day is one step of the sequence, and the network carries what it learnt from
one day to the next in its state. Its last state is then joined with what is known in advance about
day d (the forecasts and calendar dummies over day d), and dense layers map it to one output per step
of day d (96 for 15-minute data), as the MLP.

The inputs of the sequence step for lag day k (day d-k):
  - the outcome's profile on day d-k,
  - with sequence_known_inputs, the profiles of the variables known in advance on day d-k (e.g. the
    load forecast of that day: the network can relate the forecasts to the price that followed),
  - the profile of each lagged-only exogenous variable (exog_lag_days without lag 0) on day
    d-k-(m-1), where m is the smallest of its lags: a variable known up to d-2 (lags (2, 7)) enters
    step d-k with its value on day d-k-1. Only that smallest lag matters here - the sequence already
    covers the older days.
With lag_resolution 'hourly' every profile is reduced to its 24 hourly averages; a variable constant
within every day (e.g. a weekday dummy) contributes one value per day instead of a profile.

With sequence_steps 'intraday', the sequence is read one hour at a time instead (or one quarter-hour,
with lag_resolution 'full'): 7 lag days give 168 hourly steps, oldest first, each with the values of
the same variables at that hour - e.g. the price of that hour and the weekday dummies of that hour
(a single 0/1 per weekday, since an hour belongs to one local day). The lag days should then be
consecutive (e.g. 1 to 7), so that consecutive steps are consecutive hours.

Day d's known-in-advance inputs reach the output layer as their profile over day d (head_inputs
'profile'), or as one daily mean each ('daily_mean': e.g. the share of day d in each local weekday).

Training is the MLP's (forecasting_time_series/methods/mlp.py): standardised inputs and outputs, AdamW, mini-batches,
early stopping on the last validation_days calibration days, refit, n_seeds networks averaged; in
addition, the gradient norm is clipped at GRADIENT_CLIP (recurrent networks can have exploding gradients).
Inputs are standardised per feature (over every calibration day and sequence step), and day d's
inputs clipped to their calibration range widened by that range on each side.

Method parameters: those of daily_inputs.py (steps_per_day, lag_days - the days of the sequence,
lag_resolution, minimum_calibration_days, exog_lag_days, exog_lag_transform), those of the MLP
(hidden_layers - now the dense layers after the LSTM, () for none; activation, dropout, weight_decay,
learning_rate, batch_size, max_epochs, patience, validation_days, refit, n_seeds, seed), and
  lstm_units            - the size of the LSTM's state
  lstm_layers           - stacked LSTM layers (dropout is applied between them, and to the last state)
  sequence_known_inputs - True: the variables known in advance also enter each step, on their lag day
  sequence_steps        - 'day' (one step per lag day) or 'intraday' (one per hour, or quarter-hour)
  head_inputs           - 'profile' or 'daily_mean': how day d's known-in-advance inputs are given
"""
import numpy as np
import torch
from torch import nn

from . import daily_inputs, mlp
from .daily_inputs import known_in_advance

GRADIENT_CLIP = 1.0
SEQUENCE_STEPS = ('day', 'intraday')
HEAD_INPUTS = ('profile', 'daily_mean')


def _shift(lags):
    """Days by which a lagged variable is older than the lag day it is shown with (0 for lag 0 or 1)."""
    return max(min(lags) - 1, 0)


def largest_lag(method_parameters):
    """The furthest day back any input needs: the oldest lag day, plus the largest shift of a lagged variable."""
    shifts = [_shift(lags) for lags in (method_parameters.get('exog_lag_days') or {}).values()]
    return max(method_parameters['lag_days']) + max(shifts or [0])


def required_history(method_parameters):
    """Training observations needed: the largest lag plus the minimum calibration days, in steps."""
    days = largest_lag(method_parameters) + method_parameters['minimum_calibration_days']
    return days * method_parameters['steps_per_day']


def check_parameters(p):
    """Raise ValueError for a malformed LSTM parameter combination."""
    layers = p.get('hidden_layers')
    if isinstance(layers, int) or layers is None or not all(mlp._whole(n, 1) for n in layers):
        raise ValueError(f'hidden_layers must be a tuple of neurons per dense layer after the LSTM, '
                         f'() for none; got {layers!r}')
    # the MLP's checks, with at least one dense layer as the MLP needs (the LSTM may have none)
    mlp.check_parameters({**p, 'hidden_layers': tuple(layers) or (1,)})
    for name in ('lstm_units', 'lstm_layers'):
        if not mlp._whole(p.get(name), 1):
            raise ValueError(f'{name} must be a whole number of at least 1, got {p.get(name)!r}')
    if not isinstance(p.get('sequence_known_inputs'), bool):
        raise ValueError(f'sequence_known_inputs must be True or False, got {p.get("sequence_known_inputs")!r}')
    if p.get('sequence_steps', 'day') not in SEQUENCE_STEPS:
        raise ValueError(f'sequence_steps must be one of {SEQUENCE_STEPS}, got {p.get("sequence_steps")!r}')
    if p.get('head_inputs', 'profile') not in HEAD_INPUTS:
        raise ValueError(f'head_inputs must be one of {HEAD_INPUTS}, got {p.get("head_inputs")!r}')


class Network(nn.Module):
    """LSTM over the sequence, its last state joined with day d's inputs, dense layers, one output per step.
    The input is flat (sequence steps × features, then day d's inputs), so the MLP's training loop applies."""

    def __init__(self, n_steps, n_features, n_head, n_outputs, units, layers, hidden_layers, activation, dropout):
        super().__init__()
        self.n_steps, self.n_features = n_steps, n_features
        self.lstm = nn.LSTM(n_features, units, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.state_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        dense, width = [], units + n_head
        for neurons in hidden_layers:
            dense += [nn.Linear(width, neurons), mlp.ACTIVATIONS[activation]()]
            if dropout:
                dense.append(nn.Dropout(dropout))
            width = neurons
        dense.append(nn.Linear(width, n_outputs))
        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        cut = self.n_steps * self.n_features
        sequence = x[:, :cut].reshape(-1, self.n_steps, self.n_features)
        output, _ = self.lstm(sequence)
        state = self.state_dropout(output[:, -1])
        return self.dense(torch.cat([state, x[:, cut:]], dim=1))


class method(mlp.method):

    def __init__(self, method_parameters):
        check_parameters(method_parameters)
        p = method_parameters
        self.inputs = daily_inputs.DailyInputBuilder(p, name='LSTM')
        self.steps_per_day = self.inputs.steps_per_day
        self.lag_days = tuple(sorted(set(p['lag_days']), reverse=True))  # the sequence, oldest day first
        self.exog_lag_days = self.inputs.exog_lag_days
        self.largest_lag = largest_lag(p)
        self.lstm_units = p['lstm_units']
        self.lstm_layers = p['lstm_layers']
        self.sequence_known_inputs = p['sequence_known_inputs']
        self.sequence_steps = p.get('sequence_steps', 'day')
        self.head_inputs = p.get('head_inputs', 'profile')
        self.hidden_layers = tuple(p['hidden_layers'])
        self.activation = p['activation']
        self.dropout = float(p['dropout'])
        self.weight_decay = float(p['weight_decay'])
        self.learning_rate = float(p['learning_rate'])
        self.batch_size = p['batch_size']
        self.max_epochs = p['max_epochs']
        self.patience = p['patience']
        self.validation_days = p['validation_days']
        self.refit = p['refit']
        self.n_seeds = p['n_seeds']
        self.seed = p['seed']
        self.gradient_clip = GRADIENT_CLIP
        self.layout = None     # (sequence steps, features per step, day-d inputs), after the last fit
        self.epochs_used = []
        self.n_inputs = None   # features per step + day-d inputs, after the last fit

    # main
    def predict_horizon(self, data, outcome_variable, time_variable, horizon, exog_variables=(), future_exog=None):
        """Fit the networks on `data` (whole days) and forecast the first `horizon` steps of the next day."""
        same_day = known_in_advance(list(exog_variables or []), self.exog_lag_days)
        if same_day and horizon < self.steps_per_day:
            raise ValueError(f'the LSTM uses the whole day-d profile of {same_day}: forecast whole days '
                             f'(horizon {horizon} < steps_per_day {self.steps_per_day})')
        sequence, head, y, sequence_next, head_next = self.design(data, outcome_variable, time_variable,
                                                                  horizon, exog_variables, future_exog)
        X, x_next = self.prepare_inputs(sequence, head, sequence_next, head_next)
        y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
        y_scale[~(y_scale > 0)] = 1.0
        Y = (y - y_mean) / y_scale

        self.epochs_used = []
        forecasts = np.mean([self.fit_one(X, Y, x_next, self.seed + k) for k in range(self.n_seeds)], axis=0)
        return list(forecasts * y_scale + y_mean)[:horizon]

    # subsidiary
    def design(self, data, outcome_variable, time_variable, horizon, exog_variables, future_exog):
        """(sequences: calibration days × steps × features, day-d inputs: calibration days × inputs,
        outcome: calibration days × steps of the day, and the same two inputs for day d)."""
        b = self.inputs
        if horizon > self.steps_per_day:
            raise ValueError(f'the LSTM forecasts one day at a time: horizon {horizon} is longer than '
                             f'steps_per_day={self.steps_per_day}')
        exog_variables = list(exog_variables or [])
        unknown = [c for c in self.exog_lag_days if c not in exog_variables]
        if unknown:
            raise ValueError(f'exog_lag_days lists {unknown}, which are not among the exogenous variables')
        known = known_in_advance(exog_variables, self.exog_lag_days)
        lagged = [c for c in self.exog_lag_days if c not in known]
        values = b.whole_days(data, [outcome_variable] + known + lagged, time_variable)  # (days, steps, columns)
        outcome = values[:, :, 0]
        known_past = values[:, :, 1:1 + len(known)].copy()
        known_next = b.exog_of_forecast_day(future_exog, known, horizon).copy()          # (horizon, columns)
        for i, c in enumerate(known):
            if c in self.exog_lag_days:  # a lag-0 variable: on the scale of the lagged variables
                known_past[:, :, i], scale = b.scale_exog(known_past[:, :, i])
                known_next[:, i] = scale(known_next[:, i])
        lagged_past = [b.scale_exog(values[:, :, 1 + len(known) + j])[0] for j in range(len(lagged))]
        shifts = [_shift(self.exog_lag_days[c]) for c in lagged]
        # a variable constant within every day (e.g. a weekday dummy) gives one value per day
        daily = [bool(np.all(known_past[:, :, i] == known_past[:, :1, i])) and
                 bool(np.all(known_next[:, i] == known_next[0, i])) for i in range(len(known))]

        n_days = len(outcome)
        targets = np.arange(self.largest_lag, n_days)  # calibration days with every input available
        if len(targets) < b.minimum_calibration_days:
            raise ValueError(f'the LSTM has {len(targets)} calibration days with all lags available; '
                             f'minimum_calibration_days={b.minimum_calibration_days}')

        def profile(day_values, one_value):  # (days, steps) → (days, inputs)
            return day_values[:, :1] if one_value else b.lag_profile(day_values)

        def step(days):  # the inputs of the sequence step for each of `days` (day positions)
            parts = [profile(outcome[days], False)]
            if self.sequence_known_inputs:
                parts += [profile(known_past[days, :, i], daily[i]) for i in range(len(known))]
            parts += [profile(series[days - shift], False) for series, shift in zip(lagged_past, shifts)]
            return np.hstack(parts)

        def intraday_steps(days):  # (days, hours or quarter-hours, features) of each of `days`
            parts = [b.lag_profile(outcome[days])]
            if self.sequence_known_inputs:
                parts += [b.lag_profile(known_past[days, :, i]) for i in range(len(known))]
            parts += [b.lag_profile(series[days - shift]) for series, shift in zip(lagged_past, shifts)]
            return np.stack(parts, axis=-1)

        def sequences(days):  # (days, sequence steps, features): lag days oldest first
            if self.sequence_steps == 'intraday':
                return np.concatenate([intraday_steps(days - k) for k in self.lag_days], axis=1)
            return np.stack([step(days - k) for k in self.lag_days], axis=1)

        def head(known_day):  # day d's known-in-advance inputs, from (days, steps, columns)
            if self.head_inputs == 'daily_mean':
                parts = [known_day[:, :, i].mean(axis=1, keepdims=True) for i in range(len(known))]
            else:
                parts = [profile(known_day[:, :, i], daily[i]) for i in range(len(known))]
            return np.hstack(parts) if parts else np.zeros((len(known_day), 0))

        return (sequences(targets), head(known_past[targets]), outcome[targets],
                sequences(np.array([n_days]))[0], head(known_next[None, :, :])[0])

    def prepare_inputs(self, sequence, head, sequence_next, head_next):
        """Standardise each feature (over calibration days and sequence steps) and each day-d input, clip
        day d's to the calibration range widened by that range on each side, and flatten (MLP layout)."""
        def standardise(past, next_, axes):
            mean, scale = past.mean(axis=axes), past.std(axis=axes)
            scale[~(scale > 1e-9 * (1 + np.abs(mean)))] = 1.0  # constant inputs: centred to 0
            past, next_ = (past - mean) / scale, (next_ - mean) / scale
            low, high = past.min(axis=axes), past.max(axis=axes)
            return past, np.clip(next_, low - (high - low), high + (high - low))

        sequence, sequence_next = standardise(sequence, sequence_next, (0, 1))
        head, head_next = standardise(head, head_next, 0)
        n_days, n_steps, n_features = sequence.shape
        self.layout = (n_steps, n_features, head.shape[1])
        self.n_inputs = n_features + head.shape[1]
        return (np.hstack([sequence.reshape(n_days, -1), head]),
                np.concatenate([sequence_next.ravel(), head_next]))

    def network(self, n_inputs, n_outputs):
        n_steps, n_features, n_head = self.layout
        return Network(n_steps, n_features, n_head, n_outputs, self.lstm_units, self.lstm_layers,
                       self.hidden_layers, self.activation, self.dropout)
