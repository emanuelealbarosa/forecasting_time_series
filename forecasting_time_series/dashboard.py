"""Interactive dashboard (Dash) of one saved run: its test forecasts against the actual values.
Needs the optional dependencies dash and plotly: pip install "forecasting-time-series[dashboard]".

    from forecasting_time_series import dashboard
    dashboard.run_dashboard('results/2026-11-10_125129_LSTM-prices-32w', data, port=8050)
or from the command line (the latest run in a results folder by default):
    python -m forecasting_time_series.dashboard [<run folder>] --data FILE.csv [--results results]
        [--time-column date] [--value-column price] [--port 8050]

What it reads:
  - the run folder (forecasting_time_series/results_log.py): forecasts.csv - the actual and every model's
    test forecasts; ../runs.csv - each model's kind and validation score (without it, kinds are read
    from the labels);
  - the original data (a DataFrame or a CSV path, with a time column and a value column - by default
    'date' and 'price', as in the experiments): the values before the test period, shown as history.
Only the top_n single models and the top_n ensembles (every combination: ensembles, combinations
across methods, block-wise selections) are kept, ranked by their validation score - never by the test
score, which would pick the models on the data they are judged on. A combination identical to a
model already kept (e.g. 'All methods: best (X)', which is X) is dropped.

Two views:
  Forecast vs actual - the actual series and the chosen models over the test period (and some history):
    aggregated by day (daily means, the default), by hour (hourly means) or not at all (every
    quarter-hour); by hour and without aggregation, specific hours / quarter-hours can be picked
    (e.g. 18:00 and 19:00: one point per day each). The legend gives each model's RMSE over the
    quarter-hours shown.
  Accuracy by hour - the chosen metrics (RMSE, MAE, MAPE) of the chosen models at each hour (or
    quarter-hour) of the day over the test days, one chart per metric; below, every model kept, ranked
    by its metrics over the whole test period.
Models are coded S1-S5 (single models, best validation first) and E1-E5 (ensembles); each keeps its
colour and line in every view and selection: singles solid, ensembles dashed.
Times are UTC, as in the data.
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dcc, html
    from plotly.subplots import make_subplots
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(f'the dashboard needs dash and plotly: '
                              f'pip install "forecasting-time-series[dashboard]"', name=error.name) from error

# colours: the categorical palette's slots 1-6 in fixed order (light surface). Slot 1 is the actual
# series; models take slots 2-6. Singles and ensembles share the five hues (solid vs dashed), the
# ensembles shifted by one so that the best single (S1) and the best ensemble (E1) differ in hue too
ACTUAL_COLOUR = '#2a78d6'
MODEL_COLOURS = ('#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e1e0d9'
FONT = 'Segoe UI, Helvetica, Arial, sans-serif'

TOP_N = 5                      # single models and ensembles kept, each
LARGE_SERIES = 5000            # above this many points, draw with WebGL so the chart stays responsive
METRIC_NAMES = {'rmse': 'RMSE', 'mae': 'MAE', 'mape': 'MAPE (%)'}
AGGREGATIONS = {'day': 'By day', 'hour': 'By hour', 'none': 'None'}


# ---------------------------------------------------------------------- metrics
def score(metric, predictions, actual) -> float:
    """RMSE, MAE or MAPE (percent; over the non-zero actual values only), ignoring missing values."""
    predictions, actual = np.asarray(predictions, dtype=float), np.asarray(actual, dtype=float)
    keep = np.isfinite(predictions) & np.isfinite(actual)
    if metric == 'mape':
        keep &= actual != 0
    if not keep.any():
        return np.nan
    error = predictions[keep] - actual[keep]
    if metric == 'rmse':
        return float(np.sqrt(np.mean(error ** 2)))
    if metric == 'mae':
        return float(np.mean(np.abs(error)))
    return float(np.mean(np.abs(error / actual[keep])) * 100)


# ---------------------------------------------------------------------- loading a run
@dataclass
class RunData:
    run_id: str
    frame: pd.DataFrame     # time, actual, then one column per model code (NaN before the test period)
    test_start: int         # first row of the test period in frame
    models: pd.DataFrame    # one row per model code (S1.., E1..): label, name, kind, validation_score, colour, dash
    validation_metric: str  # the metric of the validation scores (e.g. 'rmse')


def load_history(data, time_column: str = 'date', value_column: str = 'price') -> pd.DataFrame:
    """The original series as (time, actual), time in UTC without a time zone. `data`: a DataFrame or a
    CSV path, with the columns time_column and value_column."""
    if isinstance(data, pd.DataFrame):
        raw = data[[time_column, value_column]]
    else:
        raw = pd.read_csv(data, usecols=[time_column, value_column])
    time = pd.to_datetime(raw[time_column], utc=True).dt.tz_localize(None)
    return pd.DataFrame({'time': time.to_numpy(), 'actual': raw[value_column].to_numpy(dtype=float)}) \
        .dropna().sort_values('time').reset_index(drop=True)


def load_run(results_folder, data, time_column: str = 'date', value_column: str = 'price',
             top_n: int = TOP_N) -> RunData:
    """The run saved in `results_folder` (with the history from the original `data`, see load_history),
    keeping the top_n single models and the top_n ensembles by validation score."""
    folder = Path(results_folder)
    forecasts = pd.read_csv(folder / 'forecasts.csv')
    labels = [c for c in forecasts.columns if c not in ('date', 'actual')]
    info = pd.DataFrame({'label': labels})
    validation_metric = 'rmse'
    runs_file = folder.parent / 'runs.csv'
    if runs_file.exists():
        runs = pd.read_csv(runs_file)
        runs = runs[runs['run_id'] == folder.name].drop_duplicates('model')
        info = info.merge(runs[['model', 'kind', 'validation_score']], left_on='label', right_on='model',
                          how='left').drop(columns='model')
        if runs['validation_metric'].notna().any():
            validation_metric = str(runs['validation_metric'].dropna().iloc[0])
    else:
        info['kind'], info['validation_score'] = np.nan, np.nan
    guessed = np.where(info['label'].str.contains(r': single #\d+'), 'single', 'ensemble')
    info['kind'] = info['kind'].fillna(pd.Series(guessed, index=info.index))
    info['group'] = np.where(info['kind'] == 'single', 'single', 'ensemble')
    info['order'] = np.arange(len(info))  # the file's order, for models without a validation score

    # best validation first (singles before combinations on ties); drop forecasts equal to one already kept
    info = info.sort_values(['validation_score', 'group', 'order'], na_position='last',
                            key=lambda c: c.map({'single': 0, 'ensemble': 1}) if c.name == 'group' else c)
    kept, values = [], []
    for row in info.itertuples():
        forecast = forecasts[row.label].to_numpy(dtype=float)
        if any(np.allclose(forecast, other, equal_nan=True) for other in values):
            continue
        kept.append(row.Index)
        values.append(forecast)
    info = info.loc[kept]
    rows = []
    for group, prefix, shift in (('single', 'S', 0), ('ensemble', 'E', 1)):
        chosen = info[info['group'] == group].head(top_n)
        for rank, row in enumerate(chosen.itertuples(), start=1):
            rows.append({'code': f'{prefix}{rank}', 'label': row.label, 'name': short_name(row.label, row.kind),
                         'kind': row.kind, 'validation_score': row.validation_score,
                         'colour': MODEL_COLOURS[(rank - 1 + shift) % len(MODEL_COLOURS)],
                         'dash': 'solid' if group == 'single' else 'dash'})
    if not rows:
        raise ValueError(f'{folder / "forecasts.csv"} has no forecasts')
    models = pd.DataFrame(rows).set_index('code')

    # history from the original data up to the end of the test period; the test actuals from the run
    test = pd.DataFrame({'time': pd.to_datetime(forecasts['date']).to_numpy(),
                         'actual': forecasts['actual'].to_numpy(dtype=float)})
    values = load_history(data, time_column, value_column)
    history = values[values['time'] < test['time'].iloc[0]]
    frame = pd.concat([history, test], ignore_index=True)
    test_start = len(history)
    for code, label in models['label'].items():
        frame[code] = np.nan
        frame.loc[test_start:, code] = forecasts[label].to_numpy(dtype=float)
    return RunData(folder.name, frame, test_start, models, validation_metric)


def short_name(label: str, kind: str) -> str:
    """A label short enough for a legend; the full one stays in the ranking's tooltips."""
    if kind == 'block-wise':
        return 'Block-wise selection'
    name = re.sub(r' \(window \d+\)$', '', label)
    return name if len(name) <= 70 else name[:69] + '…'


# ---------------------------------------------------------------------- the app
def build_app(results_folder, data, time_column: str = 'date', value_column: str = 'price', top_n: int = TOP_N,
              title: str | None = None, ylabel: str | None = None) -> Dash:
    """The dashboard of the run in `results_folder`, with the history before the test period from `data`
    (the original series: a DataFrame or a CSV path, see load_history). ylabel: the values' unit."""
    run = load_run(results_folder, data, time_column, value_column, top_n)
    frame, models, test_start = run.frame, run.models, run.test_start
    test = frame.iloc[test_start:]
    codes = list(models.index)
    defaults = [c for c in ('S1', 'E1') if c in models.index] or codes[:1]
    model_options = [{'label': f'{code}  {row["name"]}', 'value': code} for code, row in models.iterrows()]
    test_days = test['time'].dt.normalize().nunique()
    title = title or f'{value_column}: run {run.run_id}'

    label_style = {'fontSize': '13px', 'color': TEXT_SECONDARY, 'marginBottom': '4px'}
    row_style = {'display': 'flex', 'gap': '32px', 'flexWrap': 'wrap', 'alignItems': 'flex-end', 'marginTop': '16px'}
    tab_style = {'padding': '8px 16px', 'fontSize': '14px', 'backgroundColor': SURFACE,
                 'border': 'none', 'borderBottom': f'1px solid {GRID}', 'color': TEXT_SECONDARY}
    selected_tab_style = {**tab_style, 'color': TEXT, 'fontWeight': 600, 'borderBottom': f'2px solid {TEXT}'}

    def control(label, component, label_id=None):
        return html.Div([html.Div(label, id=label_id, style=label_style) if label_id else
                         html.Div(label, style=label_style), component])

    forecast_view = html.Div([
        html.Div(style=row_style, children=[
            control('Models', dcc.Dropdown(id='forecast-models', options=model_options, value=defaults, multi=True,
                                           style={'width': '560px', 'maxWidth': '90vw'})),
        ]),
        html.Div(style=row_style, children=[
            control('Aggregation', dcc.RadioItems(id='aggregation', value='day', inline=True,
                                                  options=[{'label': f' {v}', 'value': k}
                                                           for k, v in AGGREGATIONS.items()],
                                                  inputStyle={'marginLeft': '12px'})),
            control('Hours (UTC)', dcc.Dropdown(id='hours', multi=True, placeholder='All',
                                                style={'width': '320px'}), label_id='hours-label'),
            control('History before the test period',
                    dcc.RadioItems(id='history', value=7, inline=True, inputStyle={'marginLeft': '12px'},
                                   options=[{'label': ' None', 'value': 0}, {'label': ' 1 week', 'value': 7},
                                            {'label': ' 4 weeks', 'value': 28}, {'label': ' 12 weeks', 'value': 84},
                                            {'label': ' All', 'value': 'all'}])),
        ]),
        dcc.Graph(id='forecast-chart', style={'height': '65vh', 'marginTop': '12px'}, config={'displaylogo': False}),
        html.Div(id='forecast-summary', style={'fontSize': '13px', 'color': TEXT_SECONDARY}),
    ])

    accuracy_view = html.Div([
        html.Div(style=row_style, children=[
            control('Metrics', dcc.Checklist(id='metrics', value=['rmse'], inline=True,
                                             options=[{'label': f' {v}', 'value': k} for k, v in METRIC_NAMES.items()],
                                             inputStyle={'marginLeft': '12px'})),
            control('Resolution', dcc.RadioItems(id='resolution', value='hour', inline=True,
                                                 options=[{'label': ' Hour', 'value': 'hour'},
                                                          {'label': ' Quarter-hour', 'value': 'quarter'}],
                                                 inputStyle={'marginLeft': '12px'})),
            control('Models', dcc.Dropdown(id='accuracy-models', options=model_options, value=defaults, multi=True,
                                           style={'width': '560px', 'maxWidth': '90vw'})),
        ]),
        dcc.Graph(id='accuracy-chart', style={'marginTop': '12px'}, config={'displaylogo': False}),
        html.Div(f'Ranking over the whole test period ({test_days} days), sorted by the first metric chosen; '
                 f'validation: {run.validation_metric.upper()} on the validation days (the models are ranked '
                 f'S1-S{top_n} and E1-E{top_n} by it)', style={**label_style, 'marginTop': '20px'}),
        html.Div(id='ranking'),
        html.Details([html.Summary('Table of the values in the chart', style={'cursor': 'pointer', 'fontSize': '13px',
                                                                              'marginTop': '16px'}),
                      html.Div(id='accuracy-table')]),
    ])

    app = Dash(__name__, title='Forecast dashboard')
    app.layout = html.Div(style={'fontFamily': FONT, 'backgroundColor': SURFACE, 'color': TEXT,
                                 'padding': '16px 24px', 'minHeight': '100vh', 'boxSizing': 'border-box'},
        children=[
            html.H2(title, style={'fontSize': '18px', 'fontWeight': 600, 'margin': '0 0 4px'}),
            html.Div(f'Test period {test["time"].iloc[0]:%Y-%m-%d} to {test["time"].iloc[-1]:%Y-%m-%d} '
                     f'({test_days} days) · the {top_n} best single models (S, solid) and {top_n} best ensembles '
                     f'(E, dashed) on validation', style=label_style),
            dcc.Tabs(value='forecast', children=[
                dcc.Tab(forecast_view, label='Forecast vs actual', value='forecast',
                        style=tab_style, selected_style=selected_tab_style),
                dcc.Tab(accuracy_view, label='Accuracy by hour', value='accuracy',
                        style=tab_style, selected_style=selected_tab_style),
            ]),
        ])

    day_steps = steps_per_day(frame)
    hour_labels = sorted(frame['time'].dt.strftime('%H:00').unique())
    quarter_labels = sorted(frame['time'].dt.strftime('%H:%M').unique())

    @app.callback(Output('hours', 'options'), Output('hours', 'value'), Output('hours', 'disabled'),
                  Output('hours-label', 'children'), Input('aggregation', 'value'))
    def hour_picker(aggregation):
        if aggregation == 'day':
            return [], [], True, 'Hours (UTC): whole days'
        labels = hour_labels if aggregation == 'hour' else quarter_labels
        return ([{'label': label, 'value': label} for label in labels], [], False,
                'Hours (UTC)' if aggregation == 'hour' else 'Quarter-hours (UTC)')

    @app.callback(Output('forecast-chart', 'figure'), Output('forecast-summary', 'children'),
                  Input('forecast-models', 'value'), Input('aggregation', 'value'), Input('hours', 'value'),
                  Input('history', 'value'))
    def forecast_chart(chosen, aggregation, hours, history_days):
        chosen = [c for c in codes if c in (chosen or [])]
        start = 0 if history_days == 'all' else max(0, test_start - history_days * day_steps)
        view = frame.iloc[start:]
        hours = hours or []
        if aggregation == 'hour' and hours:
            view = view[view['time'].dt.strftime('%H:00').isin(hours)]
        elif aggregation == 'none' and hours:
            view = view[view['time'].dt.strftime('%H:%M').isin(hours)]
        scored = view[view.index >= test_start]  # the quarter-hours behind what is shown
        shown = aggregate(view, ['actual'] + chosen, aggregation)
        sparse = aggregation != 'day' and bool(hours)  # a few points per day: draw the points
        trace = go.Scattergl if len(shown) > LARGE_SERIES else go.Scatter
        mode = 'lines+markers' if sparse else 'lines'

        figure = go.Figure()
        figure.add_vrect(x0=frame['time'].iloc[test_start], x1=frame['time'].iloc[-1],
                         fillcolor=GRID, opacity=0.5, line_width=0, layer='below',
                         annotation_text='test period', annotation_position='top left',
                         annotation_font=dict(size=11, color=TEXT_SECONDARY))
        figure.add_trace(trace(x=shown['time'], y=shown['actual'], name='Actual', mode=mode,
                               line=dict(color=ACTUAL_COLOUR, width=2), marker=dict(size=8)))
        texts = []
        test_shown = shown[shown['time'] >= frame['time'].iloc[test_start].floor('D' if aggregation == 'day' else 'h')]
        for code in chosen:
            row = models.loc[code]
            rmse = score('rmse', scored[code], scored['actual'])
            mae = score('mae', scored[code], scored['actual'])
            texts.append(f'{code} RMSE {rmse:,.2f}, MAE {mae:,.2f}')
            figure.add_trace(trace(x=test_shown['time'], y=test_shown[code], mode=mode,
                                   name=f'{code} {row["name"]} (RMSE {rmse:,.1f})',
                                   line=dict(color=row['colour'], width=2, dash=row['dash']), marker=dict(size=8)))
        style_figure(figure, ylabel)
        figure.update_layout(legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom'))
        what = {'day': 'daily means', 'hour': 'hourly means', 'none': 'every quarter-hour'}[aggregation]
        if hours:
            what += f' at {", ".join(sorted(hours))} UTC'
        summary = (f'Shown: {what}. Scores over the {len(scored)} test quarter-hours behind them'
                   + (': ' + ' · '.join(texts) if texts else ' (no model chosen).'))
        return figure, summary

    @app.callback(Output('accuracy-chart', 'figure'), Output('accuracy-chart', 'style'),
                  Output('ranking', 'children'), Output('accuracy-table', 'children'),
                  Input('metrics', 'value'), Input('accuracy-models', 'value'), Input('resolution', 'value'))
    def accuracy_chart(metrics, chosen, resolution):
        metrics = [m for m in METRIC_NAMES if m in (metrics or [])]
        chosen = [c for c in codes if c in (chosen or [])]
        ranking = ranking_table(test, models, metrics[0] if metrics else 'rmse', chosen, run.validation_metric)
        if not metrics or not chosen:
            figure = go.Figure()
            figure.add_annotation(text='Choose at least one metric and one model', showarrow=False,
                                  font=dict(color=TEXT_SECONDARY, size=14))
            style_figure(figure, '')
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            return figure, {'height': '300px'}, ranking, None
        scores = scores_by_time_of_day(test, chosen, metrics, resolution)
        figure = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.06)
        for i, metric in enumerate(metrics, start=1):
            for code in chosen:
                row = models.loc[code]
                figure.add_trace(go.Scatter(x=scores.index, y=scores[(metric, code)], mode='lines',
                                            name=f'{code} {row["name"]}', legendgroup=code, showlegend=i == 1,
                                            line=dict(color=row['colour'], width=2, dash=row['dash']),
                                            hovertemplate=f'%{{y:,.2f}}<extra>{code}</extra>'), row=i, col=1)
            figure.update_yaxes(title_text=METRIC_NAMES[metric], rangemode='tozero', row=i, col=1)
        style_figure(figure, None)
        ticks = [label for label in scores.index if label.endswith(':00')][::1 if resolution == 'hour' else 2]
        figure.update_xaxes(type='category', tickmode='array', tickvals=ticks, ticktext=ticks)
        figure.update_xaxes(title_text='Hour of the day (UTC)' if resolution == 'hour' else 'Quarter-hour (UTC)',
                            row=len(metrics), col=1)
        figure.update_layout(legend=dict(orientation='h', x=0, y=1.02, yanchor='bottom'))
        height = 120 + 300 * len(metrics)
        return figure, {'height': f'{height}px', 'marginTop': '12px'}, ranking, values_table(scores, models)

    return app


def run_dashboard(results_folder, data, time_column: str = 'date', value_column: str = 'price', port: int = 8050,
                  top_n: int = TOP_N, **kwargs):
    """Build the dashboard of `results_folder` and serve it at http://127.0.0.1:<port> (until stopped).
    kwargs: build_app's title and ylabel."""
    app = build_app(results_folder, data, time_column, value_column, top_n, **kwargs)
    print(f'\nStarting the dashboard: open http://127.0.0.1:{port} (stop with Ctrl+C)', flush=True)
    app.run(port=port, debug=False)


# ---------------------------------------------------------------------- helpers
def steps_per_day(frame: pd.DataFrame) -> int:
    return int(frame['time'].dt.normalize().value_counts().max())


def aggregate(view: pd.DataFrame, columns: list[str], aggregation: str) -> pd.DataFrame:
    """Daily or hourly means of `columns` (a forecast is shown where it covers its whole day or hour), or
    the view as it is."""
    if aggregation == 'none':
        return view[['time'] + columns]
    period = view['time'].dt.floor('D' if aggregation == 'day' else 'h')
    grouped = view[columns].groupby(period.to_numpy())
    means = grouped.mean().where(grouped.count() == grouped.size().to_numpy()[:, None])
    return means.rename_axis('time').reset_index()


def scores_by_time_of_day(test: pd.DataFrame, codes: list[str], metrics: list[str], resolution: str) -> pd.DataFrame:
    """One row per hour (or quarter-hour) of the day, one column per (metric, model): the metric over the
    test days at that time of day."""
    key = test['time'].dt.strftime('%H:00' if resolution == 'hour' else '%H:%M')
    rows = {label: {(metric, code): score(metric, group[code], group['actual']) for metric in metrics
                    for code in codes}
            for label, group in test.groupby(key, sort=True)}
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def ranking_table(test: pd.DataFrame, models: pd.DataFrame, sort_metric: str, chosen: list[str],
                  validation_metric: str) -> html.Table:
    """Every model kept, with its metrics over the whole test period, best first by `sort_metric`;
    the models in the chart are marked with their line."""
    rows = [{'code': code, 'name': row['name'], 'label': row['label'], 'kind': row['kind'],
             'validation': row['validation_score'],
             **{metric: score(metric, test[code], test['actual']) for metric in METRIC_NAMES}}
            for code, row in models.iterrows()]
    rows.sort(key=lambda r: (np.isnan(r[sort_metric]), r[sort_metric]))
    cell = {'padding': '4px 12px', 'textAlign': 'right', 'fontVariantNumeric': 'tabular-nums',
            'borderBottom': f'1px solid {GRID}'}
    left = {**cell, 'textAlign': 'left'}
    number = lambda v: '–' if pd.isna(v) else f'{v:,.2f}'
    header = html.Tr([html.Th('#', style=cell), html.Th('', style=left), html.Th('Model', style=left),
                      html.Th('Kind', style=left), html.Th(f'Validation {validation_metric.upper()}', style=cell)]
                     + [html.Th(f'Test {name}', style={**cell, 'fontWeight': 700 if m == sort_metric else 400})
                        for m, name in METRIC_NAMES.items()])
    body = []
    for place, r in enumerate(rows, start=1):
        row = models.loc[r['code']]
        swatch = html.Span(style={'display': 'inline-block', 'width': '24px', 'verticalAlign': 'middle',
                                  'borderTop': f'2px {"solid" if row["dash"] == "solid" else "dashed"} '
                                               f'{row["colour"]}'}) if r['code'] in chosen else ''
        body.append(html.Tr([html.Td(place, style=cell), html.Td(swatch, style=left),
                             html.Td([html.B(r['code']), f'  {r["name"]}'], title=r['label'], style=left),
                             html.Td(r['kind'], style=left), html.Td(number(r['validation']), style=cell)]
                            + [html.Td(number(r[m]), style={**cell, 'fontWeight': 700 if m == sort_metric else 400})
                               for m in METRIC_NAMES],
                            style={'color': TEXT if r['code'] in chosen else TEXT_SECONDARY}))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={'borderCollapse': 'collapse', 'fontSize': '13px', 'marginTop': '4px'})


def values_table(scores: pd.DataFrame, models: pd.DataFrame) -> html.Table:
    cell = {'padding': '2px 12px', 'textAlign': 'right', 'fontVariantNumeric': 'tabular-nums',
            'borderBottom': f'1px solid {GRID}'}
    header = html.Tr([html.Th('Time (UTC)', style={**cell, 'textAlign': 'left'})]
                     + [html.Th(f'{code} {METRIC_NAMES[metric]}', title=models.loc[code, 'label'], style=cell)
                        for metric, code in scores.columns])
    rows = [html.Tr([html.Td(label, style={**cell, 'textAlign': 'left'})]
                    + [html.Td('–' if np.isnan(value) else f'{value:,.2f}', style=cell) for value in values])
            for label, values in zip(scores.index, scores.to_numpy())]
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={'borderCollapse': 'collapse', 'fontSize': '13px', 'marginTop': '8px'})


def style_figure(figure: go.Figure, ylabel: str | None) -> None:
    figure.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=TEXT),
                         margin=dict(l=60, r=20, t=40, b=40), hovermode='x unified',
                         legend=dict(bgcolor='rgba(0,0,0,0)'))
    figure.update_yaxes(gridcolor=GRID, zeroline=False, color=TEXT_SECONDARY)
    figure.update_xaxes(showgrid=False, linecolor=GRID, color=TEXT_SECONDARY)
    if ylabel:
        figure.update_yaxes(title_text=ylabel)


def latest_run(directory) -> Path:
    """The most recent run folder (with a forecasts.csv) in the results folder `directory`."""
    folders = sorted(p for p in Path(directory).iterdir() if (p / 'forecasts.csv').exists())
    if not folders:
        raise FileNotFoundError(f'no saved runs in {directory}')
    return folders[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dashboard of a saved run (forecasting_time_series.results_log)')
    parser.add_argument('folder', nargs='?', help='the run folder (default: the latest in --results)')
    parser.add_argument('--data', required=True, help='the original series (CSV)')
    parser.add_argument('--results', default='results', help='the results folder, for the latest run')
    parser.add_argument('--time-column', default='date')
    parser.add_argument('--value-column', default='price')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--top', type=int, default=TOP_N, help='single models and ensembles kept, each')
    parser.add_argument('--ylabel', default=None, help="the values' unit, e.g. EUR/MWh")
    arguments = parser.parse_args()
    run_dashboard(arguments.folder or latest_run(arguments.results), arguments.data, arguments.time_column,
                  arguments.value_column, arguments.port, arguments.top, ylabel=arguments.ylabel)
