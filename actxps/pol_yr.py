import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from dateutil.relativedelta import relativedelta
from datetime import datetime
from actxps.tools import document, arg_match

_pol_doc = """
# Calculate policy durations

Given a vector of dates and a vector of issue dates, calculate
policy years (`pol_yr()`), quarters (`pol_qtr()`), months (`pol_mth()`), 
or weeks (`pol_wk()`).

These functions assume the first day of each policy year is the
anniversary date (or issue date in the first year). The last day of each
policy year is the day before the next anniversary date. Analogous rules
are used for policy quarters, policy months, and policy weeks.

## Parameters
`dates`: str or datetime or DatetimeIndex 
    Date(s)
`issue_date`: str or datetime or DatetimeIndex
    Issue date(s)
`dur_length`: str
    Duration length. Only applies to `pol_interval()`. Must be 'year', 
    'quarter', 'month', or 'week'

## Returns 

Vector of integers
"""


@document(_pol_doc)
def pol_interval(dates: str | datetime | DatetimeIndex | pd.Series,
                 issue_date: str | datetime | DatetimeIndex | pd.Series,
                 dur_length: str) -> np.ndarray:

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    dates = _convert_date(dates)
    issue_date = _convert_date(issue_date)

    dat = pd.DataFrame({
        'issue_date': issue_date,
        'dates': dates
    }, index=np.arange(max(len2(dates), len2(issue_date))))

    if dur_length == "year":
        res = [relativedelta(a, b).years for a, b in
               zip(dat.dates, dat.issue_date)]

    elif dur_length in ["month", "quarter"]:
        def mth_calc(a, b):
            delta = relativedelta(a, b)
            return 12 * delta.years + delta.months

        if dur_length == "quarter":
            res = [mth_calc(a, b) // 3 for a, b
                   in zip(dat.dates, dat.issue_date)]
        else:
            res = [mth_calc(a, b) for a, b in zip(dat.dates, dat.issue_date)]

    else:
        res = (dat.dates - dat.issue_date).dt.days // 7

    return np.array(res) + 1


@document(_pol_doc)
def pol_yr(dates: str | datetime | DatetimeIndex | pd.Series,
           issue_date: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return pol_interval(dates, issue_date, 'year')


@document(_pol_doc)
def pol_mth(dates: str | datetime | DatetimeIndex | pd.Series,
            issue_date: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return pol_interval(dates, issue_date, 'month')


@document(_pol_doc)
def pol_qtr(dates: str | datetime | DatetimeIndex | pd.Series,
            issue_date: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return pol_interval(dates, issue_date, 'quarter')


@document(_pol_doc)
def pol_wk(dates: str | datetime | DatetimeIndex | pd.Series,
           issue_date: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return pol_interval(dates, issue_date, 'week')


def len2(x) -> int:
    """Length function with non-iterables and strings returning 1

    ## Parameters

    `x`: Any

    ## Returns

    An integer equal to 1 if `x` is not iterable or if `x` is a string.
    Otherwise, the length of `x`.
    """
    if isinstance(x, str):
        return 1
    try:
        return len(x)
    except TypeError:
        return 1


_frac_doc = """
# Calculate fractional durations

Given vectors of start and end dates, calculate elapsed years (`frac_yr()`), 
quarters (`frac_qtr()`), months (`frac_mth()`), or weeks (`frac_wk()`).

## Parameters

`start`: str or datetime or DatetimeIndex 
    Start dates
`end`: str or datetime or DatetimeIndex
    End dates
`dur_length`: str
    Duration length. Only applies to `frac_interval()`. Must be 'year', 
    'quarter', 'month', or 'week'

## Returns 

Vector of floats
"""


@document(_frac_doc)
def frac_interval(start: str | datetime | DatetimeIndex | pd.Series,
                  end: str | datetime | DatetimeIndex | pd.Series,
                  dur_length: str) -> np.ndarray:

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    start = _convert_date(start)
    end = _convert_date(end)

    dat = pd.DataFrame({
        'start': start,
        'end': end
    }, index=np.arange(max(len2(start), len2(end))))

    res = [_delta_frac(a, b, dur_length) for a, b
           in zip(dat.start, dat.end)]

    return np.array(res)


def _delta_frac(start: datetime, end: datetime, dur_length: str) -> float:
    """
    Internal function for calculating one set of fractional durations.

    This function is used by `frac_interval()` and is not meant to be called
    directly.

    ## Parameters

    `start`: datetime
        Start date
    `end`: datetime
        end date
    `dur_length`: str
    Duration length. Only applies to `frac_interval()`. Must be 'year', 
    'quarter', 'month', or 'week'

    ## Returns

    Float
    """

    if dur_length == 'week':
        return (end - start).days / 7

    delta = relativedelta(end, start)

    if dur_length == 'year':
        dt = (start + relativedelta(years=delta.years))
        numer = (end - dt).days
        denom = ((dt + relativedelta(years=1)) - dt).days
        res = delta.years + numer / denom
    elif dur_length == 'quarter':
        dt = (start + relativedelta(years=delta.years,
                                    months=3 * (delta.months // 3)))
        numer = (end - dt).days
        denom = ((dt + relativedelta(months=3)) - dt).days
        res = delta.years * 4 + delta.months // 3 + numer / denom
    else:
        dt = (start + relativedelta(years=delta.years,
                                    months=delta.months))
        numer = (end - dt).days
        denom = ((dt + relativedelta(months=1)) - dt).days
        res = delta.years * 12 + delta.months + numer / denom

    return res


@document(_frac_doc)
def frac_yr(start: str | datetime | DatetimeIndex | pd.Series,
            end: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return frac_interval(start, end, 'year')


@document(_frac_doc)
def frac_mth(start: str | datetime | DatetimeIndex | pd.Series,
             end: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return frac_interval(start, end, 'month')


@document(_frac_doc)
def frac_qtr(start: str | datetime | DatetimeIndex | pd.Series,
             end: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return frac_interval(start, end, 'quarter')


@document(_frac_doc)
def frac_wk(start: str | datetime | DatetimeIndex | pd.Series,
            end: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    return frac_interval(start, end, 'week')


def _convert_date(x):
    """
    Helper function to convert inputs to dates. 
    This function should not be called directly.
    """
    if isinstance(x, pd.Series):
        x = x.values
    else:
        if not isinstance(x, DatetimeIndex):
            x = pd.to_datetime(x)
    return x


_add_doc = """
# Add time periods of varying lengths to a vector of dates

Given a vector of dates and an integer vector, add years (`add_yr()`), 
quarters (`add_qtr()`), months, (`add_mth()`), or weeks (`add_wk()`).

## Parameters

`dates`: str | datetime | DatetimeIndex | pd.Series
    Dates
`x`
`end`: str or datetime or DatetimeIndex
    End dates
`dur_length`: str
    Duration length. Only applies to `frac_interval()`. Must be 'year', 
    'quarter', 'month', or 'week'

## Returns 

Vector of floats
"""


@document(_add_doc)
def add_interval(dates: str | datetime | DatetimeIndex | pd.Series,
                 x: pd.Series | np.ndarray,
                 dur_length: str) -> np.ndarray:

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    dates = pd.Series(_convert_date(dates))
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(x, list):
        x = np.array(x)

    y = dates.dt.year.values
    m = dates.dt.month.values
    d = dates.dt.day.values

    if dur_length == 'year':
        y += x
    elif dur_length == 'month':
        y += (m - 1 + x) // 12
        m = (m + x) % 12
        m = np.where(m == 0, 12, m)
    elif dur_length == 'quarter':
        q = (m - 1) // 3 + 1
        y += (q - 1 + x) // 4
        m = (m + x * 3) % 12
        m = np.where(m == 0, 12, m)
    else:
        return (dates + np.timedelta64(7, 'D') * x).values

    max_days = pd.to_datetime(pd.DataFrame({
        'year': y,
        'month': m,
        'day': 1
    })).dt.days_in_month.values

    res = pd.to_datetime(pd.DataFrame({
        'year': y,
        'month': m,
        'day': np.minimum(d, max_days)
    }))

    return res.values


@document(_add_doc)
def add_yr(dates: str | datetime | DatetimeIndex | pd.Series,
           x: pd.Series | np.ndarray) -> np.ndarray:
    return add_interval(dates, x, 'year')


@document(_add_doc)
def add_qtr(dates: str | datetime | DatetimeIndex | pd.Series,
            x: pd.Series | np.ndarray) -> np.ndarray:
    return add_interval(dates, x, 'quarter')


@document(_add_doc)
def add_mth(dates: str | datetime | DatetimeIndex | pd.Series,
            x: pd.Series | np.ndarray) -> np.ndarray:
    return add_interval(dates, x, 'month')


@document(_add_doc)
def add_wk(dates: str | datetime | DatetimeIndex | pd.Series,
           x: pd.Series | np.ndarray) -> np.ndarray:
    return add_interval(dates, x, 'week')
