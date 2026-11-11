"""Calendar covariates: deterministic, so their future values are always known.

Build them from the time column and add them to the data as exogenous variables, e.g.

    features = calendar_features(data['date'], ('day_of_week',), timezone='Europe/Bucharest')
    data = data.join(features)
    ForecastExperiment(data, ..., exog_variables=list(features.columns))

The same call on future dates gives their future values (see future_dates()).

Features (one or more):
  'day_of_week' - 7 dummies dow_mon … dow_sun
  'weekend'     - 1 on Saturday and Sunday
  'day_of_month' - 31 dummies dom_01 … dom_31
  'month'       - 12 dummies month_01 … month_12
  'holiday'     - 1 on the dates listed in `holidays`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ('day_of_week', 'weekend', 'day_of_month', 'month', 'holiday')
WEEKDAYS = ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')


def calendar_features(dates, features=('day_of_week',), timezone: str | None = None,
                      holidays=()) -> pd.DataFrame:
    """One row per date, one column per calendar covariate (0/1 floats), same index as `dates`.

    timezone: the dates are read as UTC and converted to this time zone first, so that, e.g.,
              the weekday changes at local midnight (None → the dates are used as they are)."""
    if isinstance(features, str):
        features = (features,)
    unknown = [f for f in features if f not in FEATURES]
    if unknown or not features:
        raise ValueError(f'calendar features must be chosen from {list(FEATURES)}; got {list(features)}')
    index = dates.index if isinstance(dates, pd.Series) else pd.RangeIndex(len(dates))
    stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates), utc=timezone is not None))
    if timezone is not None:
        stamps = stamps.tz_convert(timezone)

    columns = {}
    if 'day_of_week' in features:
        for number, name in enumerate(WEEKDAYS):
            columns[f'dow_{name}'] = stamps.weekday == number
    if 'weekend' in features:
        columns['weekend'] = stamps.weekday >= 5
    if 'day_of_month' in features:
        for day in range(1, 32):
            columns[f'dom_{day:02d}'] = stamps.day == day
    if 'month' in features:
        for month in range(1, 13):
            columns[f'month_{month:02d}'] = stamps.month == month
    if 'holiday' in features:
        holiday_dates = pd.to_datetime(list(holidays)).normalize()
        columns['holiday'] = stamps.tz_localize(None).normalize().isin(holiday_dates)
    return pd.DataFrame({name: np.asarray(values, dtype=float) for name, values in columns.items()}, index=index)


def future_dates(last_date, periods: int, freq: str, date_format: str | None = None) -> pd.Series:
    """The `periods` timestamps after `last_date` (e.g. freq='15min'), to build future calendar values;
    with date_format (e.g. '%Y-%m-%d %H:%M') they are returned as strings, like the data's time column."""
    stamps = pd.date_range(pd.Timestamp(last_date), periods=periods + 1, freq=freq)[1:]
    return pd.Series(stamps.strftime(date_format) if date_format else stamps)
