# Forecasting Time Series

A small framework for testing and grid-searching a range of forecasting methods on a
**single univariate time series**. It wraps each method (Naive, ETS, ARIMA, Random
Forest, plus stubs for MLP/CNN/LSTM variants) in a common interface, runs walk-forward
validation across a grid of method and preprocessing parameters, and ranks the
combinations by forecast error.

## Status

This is a personal research/prototyping project, not a maintained package. All
methods wired into `classes/method.py` are now implemented:

- **Naive** (`classes/methods/naive.py`)
- **ETS** (`classes/methods/ets.py`)
- **ARIMA** (`classes/methods/arima.py`)
- **RF** (`classes/methods/rf.py`)
- **MLP** (`classes/methods/mlp.py`) — feedforward net over a lag window
- **CNN** (`classes/methods/cnn.py`) — 1D convolution over a lag window
- **LSTM** (`classes/methods/lstm.py`) — single-layer LSTM
- **CNN-LSTM** (`classes/methods/cnn_lstm.py`) — Conv1D reads subsequences, LSTM reads their summaries
- **ConvLSTM** (`classes/methods/convlstm.py`) — ConvLSTM2D over subsequences

The neural methods (MLP through twoDLSTM) all follow the same refit-every-step
pattern as ARIMA/RF: `predict()` builds a lagged supervised dataset from the raw
series, fits a fresh Keras model, and predicts one step ahead. This means a grid
search over these methods is significantly slower than over Naive/ARIMA/RF — keep
epoch counts and the parameter grid small unless you're prepared to wait, and
consider trimming `validation_maximum_number_of_splits` too, since each split refits
from scratch.


## Installation

Requires **Python 3.10–3.13**.

```bash
python3.12 -m venv venv       # use a Python 3.10-3.13 interpreter specifically
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this code was verified against:

```
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.9.0
statsmodels==0.15.0
tensorflow==2.16.2
```

**Note for Intel Macs:** TensorFlow stopped publishing `x86_64` macOS wheels after
`2.16.2` — that's the newest version installable on an Intel Mac, which is why it's
pinned here rather than a newer release. Apple Silicon (`arm64`) Macs, Linux, and
Windows can use newer TensorFlow versions if you prefer, but `numpy` needs to stay
below `2.0` to satisfy `tensorflow==2.16.2`'s own requirement (`numpy<2.0.0,>=1.26.0`)
— installing `numpy` before the rest, or with everything pinned as above in one
`pip install -r requirements.txt` call, avoids pip resolving a newer, incompatible
`numpy` on its own.

## Usage

`run.py` is the entry point. It builds a grid of method + preprocessing parameters,
runs walk-forward validation via `classes/grid_search.py`, and prints the
top 5 parameter combinations ranked by RMSE.

```bash
python run.py
```

Swap `forecasting_method_name` in `run.py` to (e.g.) `'Naive'`, `'ETS'`, `'ARIMA'`, or `'RF'` to test a different method (each block above it defines that method's parameter
search ranges).

### Expected data structure

The `data` DataFrame passed into `grid_search.search_agent` needs:

- **outcome column** (e.g. `crossings`): the numeric series to forecast.
- **time_sequence_variable column** (e.g. `date_sequence`): a grouping key used
  internally for differencing and lag/lead construction. For a single series, keep
  this constant across all rows.
- Rows sorted in ascending chronological order, with a clean default `RangeIndex`
  (0, 1, 2, …) — several steps slice the data positionally rather than by date.
- A `time_variable` column (e.g. `date`) is expected in the general parameters but
  isn't currently used by the forecasting logic itself.

`run.py` currently ships with a small simulated DataFrame in place of a real dataset
— swap that out for your own series before running anything meaningful.

## NB
- ARIMA/ETS fits will occasionally fail to converge or throw on some parameter
  combinations — these are caught and just excluded from the ranked output.

## Project structure

```
run.py                        # entry point / example usage
classes/
├── forecast.py                # single train/predict/evaluate cycle
├── grid_builder.py            # builds the parameter grid to search
├── grid_search.py             # runs the grid search, ranks results
├── method.py                  # dispatches to the selected forecasting method
├── metrics.py                 # scoring (currently RMSE only)
├── processing.py               # differencing / power transform / normalization
├── validation.py               # walk-forward validation splits
└── methods/                   # one file per forecasting method
    ├── naive.py, ets.py, arima.py, rf.py   # implemented
    └── mlp.py, cnn.py, lstm.py, cnn_lstm.py, convlstm.py, twodlstm.py  # stubs
```

## Ideas for future work

See `todos.txt` for a running list (additional error metrics, alternative validation
strategies, multistep walk-forward forecasting, unit tests, etc.).
