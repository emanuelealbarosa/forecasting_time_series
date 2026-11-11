"""MLP - a feed-forward neural network forecasting the whole next day at once (PyTorch).

In the spirit of the DNN of Lago, De Ridder & De Schutter (2018) / Lago et al. (2021): one network
with one output per step of the day (96 for 15-minute data), so the steps share what they learn,
instead of LEAR's separate model per step. Its inputs are those of LEAR (forecasting_time_series/methods/daily_inputs.py):
  - the outcome's profile on each lag day, and those of the lagged exogenous variables,
  - the whole day-d profile of each exogenous variable known in advance (LEAR's model for step h
    sees only step h; the network sees every step, e.g. the full load forecast of day d).
With lag_resolution 'hourly', every profile (lags and day d) is reduced to its 24 hourly averages.

Training, on each fit (daily recalibration, as LEAR):
  - inputs constant over the calibration days, and duplicates, are dropped; the rest are standardised,
    and day d's inputs are clipped to their calibration range widened by that range on each side;
  - each output is standardised; the loss is the mean squared error (the framework scores RMSE);
  - Adam with decoupled weight decay (AdamW), mini-batches, dropout after each hidden layer;
  - early stopping on the last `validation_days` calibration days (the ones most like day d); with
    `refit`, the network is then trained again on every calibration day for the best number of epochs;
  - `n_seeds` networks from different random starts are averaged (a network's forecast depends on its
    starting weights; seeds seed, seed+1, ... make the result reproducible).

Method parameters: those of daily_inputs.py (steps_per_day, lag_days, lag_resolution,
minimum_calibration_days, exog_lag_days, exog_lag_transform), and
  hidden_layers   - neurons per hidden layer, e.g. (256, 128)
  activation      - 'relu', 'elu' or 'tanh'
  dropout         - share of each hidden layer's outputs dropped in training, 0 to below 1
  weight_decay    - AdamW weight decay (an L2-like penalty), at least 0
  learning_rate   - Adam's step size
  batch_size      - calibration days per mini-batch
  max_epochs      - most passes over the calibration days
  patience        - epochs without improvement on the held-out days before stopping
  validation_days - the last calibration days held out for early stopping
  refit           - True: retrain on every calibration day for the best number of epochs
  n_seeds         - networks averaged
  seed            - the first random seed
"""
import numpy as np
import torch
from torch import nn

from . import daily_inputs
from .daily_inputs import largest_lag, required_history, known_in_advance  # noqa: F401

ACTIVATIONS = {'relu': nn.ReLU, 'elu': nn.ELU, 'tanh': nn.Tanh}


def _whole(value, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value, low, high=None):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= low \
        and (high is None or value < high)


def check_parameters(p):
    """Raise ValueError for a malformed MLP parameter combination."""
    daily_inputs.check_parameters(p)
    layers = p.get('hidden_layers')
    if isinstance(layers, int) or not layers or not all(_whole(n, 1) for n in layers):
        raise ValueError(f'hidden_layers must be a tuple of neurons per layer, e.g. (256, 128); got {layers!r}')
    if p.get('activation') not in ACTIVATIONS:
        raise ValueError(f'activation must be one of {list(ACTIVATIONS)}, got {p.get("activation")!r}')
    if not _number(p.get('dropout'), 0, 1):
        raise ValueError(f'dropout must be from 0 to below 1, got {p.get("dropout")!r}')
    if not _number(p.get('weight_decay'), 0):
        raise ValueError(f'weight_decay must be at least 0, got {p.get("weight_decay")!r}')
    if not _number(p.get('learning_rate'), 0) or p['learning_rate'] == 0:
        raise ValueError(f'learning_rate must be above 0, got {p.get("learning_rate")!r}')
    for name, minimum in (('batch_size', 1), ('max_epochs', 1), ('patience', 1), ('validation_days', 1),
                          ('n_seeds', 1), ('seed', 0)):
        if not _whole(p.get(name), minimum):
            raise ValueError(f'{name} must be a whole number of at least {minimum}, got {p.get(name)!r}')
    if not isinstance(p.get('refit'), bool):
        raise ValueError(f'refit must be True or False, got {p.get("refit")!r}')
    if p['minimum_calibration_days'] <= p['validation_days'] + 1:
        raise ValueError(f'minimum_calibration_days ({p["minimum_calibration_days"]}) must exceed '
                         f'validation_days + 1 ({p["validation_days"] + 1}): early stopping holds out those days')


class method():

    def __init__(self, method_parameters):
        check_parameters(method_parameters)
        self.inputs = daily_inputs.DailyInputBuilder(method_parameters, name='MLP')
        self.steps_per_day = self.inputs.steps_per_day
        p = method_parameters
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
        self.gradient_clip = None  # largest gradient norm in training, None = no clipping
        self.epochs_used = []  # best number of epochs, per seed (after the last fit)
        self.n_inputs = None   # inputs left after dropping constant and duplicate ones (after the last fit)

    # main
    def predict_horizon(self, data, outcome_variable, time_variable, horizon, exog_variables=(), future_exog=None):
        """Fit the networks on `data` (whole days) and forecast the first `horizon` steps of the next day.
        The variables known in advance are needed over the whole next day (their full profile is an input)."""
        same_day = known_in_advance(list(exog_variables or []), self.inputs.exog_lag_days)
        if same_day and horizon < self.steps_per_day:
            raise ValueError(f'the MLP uses the whole day-d profile of {same_day}: forecast whole days '
                             f'(horizon {horizon} < steps_per_day {self.steps_per_day})')
        d = self.inputs.build(data, outcome_variable, time_variable, horizon, exog_variables, future_exog)
        X, x_next = self.design(d)
        X, x_next = self.prepare_inputs(X, x_next)
        y_mean, y_scale = d.y.mean(axis=0), d.y.std(axis=0)
        y_scale[~(y_scale > 0)] = 1.0
        Y = (d.y - y_mean) / y_scale

        self.epochs_used = []
        forecasts = np.mean([self.fit_one(X, Y, x_next, self.seed + k) for k in range(self.n_seeds)], axis=0)
        return list(forecasts * y_scale + y_mean)[:horizon]

    # subsidiary
    def design(self, d):
        """(calibration days × inputs, day d's inputs): the lag profiles, then each known-in-advance
        variable's day-d profile (hourly averages with lag_resolution 'hourly')."""
        return self.inputs.whole_day_design(d)

    def prepare_inputs(self, X, x_next):
        """Drop inputs constant over the calibration days and duplicates; standardise; clip day d's inputs
        to the calibration range widened by that range on each side (as LEAR)."""
        spread = X.std(axis=0)
        varying = np.isfinite(spread) & (spread > 1e-9 * (1 + np.abs(X).max(axis=0)))
        X, x_next = X[:, varying], x_next[varying]
        mean, scale = X.mean(axis=0), X.std(axis=0)
        X, x_next = (X - mean) / scale, (x_next - mean) / scale
        low, high = X.min(axis=0), X.max(axis=0)
        x_next = np.clip(x_next, low - (high - low), high + (high - low))
        _, first = np.unique(X.round(12), axis=1, return_index=True)
        keep = np.sort(first)
        self.n_inputs = len(keep)
        return X[:, keep], x_next[keep]

    def network(self, n_inputs, n_outputs):
        layers, width = [], n_inputs
        for neurons in self.hidden_layers:
            layers += [nn.Linear(width, neurons), ACTIVATIONS[self.activation]()]
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
            width = neurons
        layers.append(nn.Linear(width, n_outputs))
        return nn.Sequential(*layers)

    def train(self, X, Y, seed, epochs, X_held=None, Y_held=None):
        """Train a network from seed `seed` for up to `epochs` epochs. With held-out days: stop after
        `patience` epochs without improvement on them, and return (the best network, its epoch count);
        without: return (the network after `epochs` epochs, epochs)."""
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        net = self.network(X.shape[1], Y.shape[1])
        optimiser = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.MSELoss()
        X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
        held = X_held is not None
        if held:
            X_held = torch.as_tensor(X_held, dtype=torch.float32)
            Y_held = torch.as_tensor(Y_held, dtype=torch.float32)
        best_loss, best_epoch, best_state = np.inf, epochs, None
        for epoch in range(1, epochs + 1):
            net.train()
            for batch in torch.randperm(len(X), generator=generator).split(self.batch_size):
                optimiser.zero_grad()
                loss_function(net(X[batch]), Y[batch]).backward()
                if self.gradient_clip:  # set by the LSTM
                    nn.utils.clip_grad_norm_(net.parameters(), self.gradient_clip)
                optimiser.step()
            if held:
                net.eval()
                with torch.no_grad():
                    loss = float(loss_function(net(X_held), Y_held))
                if loss < best_loss:
                    best_loss, best_epoch = loss, epoch
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                elif epoch - best_epoch >= self.patience:
                    break
        if held:
            net.load_state_dict(best_state)
        net.eval()
        return net, best_epoch

    def fit_one(self, X, Y, x_next, seed):
        """One network's forecast of day d (standardised outputs)."""
        k = self.validation_days
        net, epochs = self.train(X[:-k], Y[:-k], seed, self.max_epochs, X[-k:], Y[-k:])
        if self.refit:  # the held-out days are the most recent: train again with them, as long as found best
            net, _ = self.train(X, Y, seed, epochs)
        self.epochs_used.append(epochs)
        with torch.no_grad():
            return net(torch.as_tensor(x_next[None, :], dtype=torch.float32)).numpy()[0].astype(float)
