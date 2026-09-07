# Forecasting Time Series

A small framework for testing and grid-searching a range of forecasting methods on a
**single univariate time series**. It wraps each method (Naive, ETS, ARIMA, Random
Forest, plus stubs for MLP/CNN/LSTM variants) in a common interface, runs walk-forward
validation across a grid of method and preprocessing parameters, and ranks the
combinations by forecast error.

## Status

This is a personal research/prototyping project, not a maintained package. Of the
methods wired into `classes/method.py`, only these are actually implemented:

- **Naive** (`classes/methods/naive.py`)
- **ETS** (`classes/methods/ets.py`)
- **ARIMA** (`classes/methods/arima.py`)
- **RF** (`classes/methods/rf.py`)

`MLP`, `CNN`, `LSTM`, `CNN-LSTM`, `ConvLSTM`, and `twoDLSTM` are empty stub classes —
selecting one of these in `run.py` will raise an error once `.predict()` is called.

## Installation

Requires Python 3.9+ (tested on 3.12).

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this code was verified against
(Naive, ARIMA, and RF paths all run cleanly end-to-end on these):

```
numpy==2.5.3
pandas==2.2.3
scikit-learn==1.9.0
statsmodels==0.15.0
```

## Usage

`run.py` is the entry point. It builds a grid of method + preprocessing parameters,
runs walk-forward validation via `classes/grid_search.py`, and prints the
top 5 parameter combinations ranked by RMSE.

```bash
python run.py
```

Swap `forecasting_method_name` in `run.py` to `'Naive'`, `'ETS'`, `'ARIMA'`, or `'RF'`
to test a different method (each block above it defines that method's parameter
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

## Known gotchas

- The simulated-data line in `run.py`
  (`data['crossings'].values[i] = data['crossings'].values[i]*i`) relies on pandas'
  older copy-on-write-off behavior. It works with the pinned `pandas==2.2.3`, but will
  raise `ValueError: assignment destination is read-only` on pandas versions where
  copy-on-write is enabled by default (e.g. pandas 3.x). Rewrite that line with
  `.iloc` assignment if you upgrade pandas.
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
