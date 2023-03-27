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
`x`: str or datetime or DatetimeIndex 
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
def pol_interval(x: str | datetime | DatetimeIndex,
                 issue_date: str | datetime | DatetimeIndex,
                 dur_length: str) -> np.ndarray:

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    if not isinstance(x, DatetimeIndex):
        x = pd.to_datetime(x)
    if not isinstance(issue_date, DatetimeIndex):
        issue_date = pd.to_datetime(issue_date)

    dat = pd.DataFrame({
        'issue_date': issue_date,
        'x': x
    }, index=np.arange(max(len2(x), len2(issue_date))))

    if dur_length == "year":
        res = [relativedelta(a, b).years for a, b in
               zip(dat.x, dat.issue_date)]

    elif dur_length in ["month", "quarter"]:
        def mth_calc(a, b):
            delta = relativedelta(a, b)
            return 12 * delta.years + delta.months

        if dur_length == "quarter":
            res = [mth_calc(a, b) // 3 for a, b in zip(dat.x, dat.issue_date)]
        else:
            res = [mth_calc(a, b) for a, b in zip(dat.x, dat.issue_date)]

    else:
        res = (dat.x - dat.issue_date).dt.days // 7

    return np.array(res) + 1


@document(_pol_doc)
def pol_yr(x: str | datetime | DatetimeIndex,
           issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
    return pol_interval(x, issue_date, 'year')


@document(_pol_doc)
def pol_mth(x: str | datetime | DatetimeIndex,
            issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
    return pol_interval(x, issue_date, 'month')


@document(_pol_doc)
def pol_qtr(x: str | datetime | DatetimeIndex,
            issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
    return pol_interval(x, issue_date, 'quarter')


@document(_pol_doc)
def pol_wk(x: str | datetime | DatetimeIndex,
           issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
    return pol_interval(x, issue_date, 'week')


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
def frac_interval(start: str | datetime | DatetimeIndex,
                  end: str | datetime | DatetimeIndex,
                  dur_length: str) -> np.ndarray:

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    if not isinstance(start, DatetimeIndex):
        start = pd.to_datetime(start)
    if not isinstance(end, DatetimeIndex):
        end = pd.to_datetime(end)

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
        denom = ((start + relativedelta(years=delta.years + 1)) -
                 (start + relativedelta(years=delta.years))).days
        res = delta.years + delta.months / 12 + delta.days / denom
    elif dur_length == 'quarter':
        denom = ((start + relativedelta(years=delta.years,
                                        months=delta.months + 1)) -
                 (start + relativedelta(years=delta.years,
                                        months=delta.months))).days
        res = delta.years * 4 + (delta.months + delta.days / denom) / 3
    else:
        denom = ((start + relativedelta(years=delta.years,
                                        months=delta.months + 1)) -
                 (start + relativedelta(years=delta.years,
                                        months=delta.months))).days
        res = delta.years * 12 + delta.months + delta.days / denom

    return res


@document(_frac_doc)
def frac_yr(start: str | datetime | DatetimeIndex,
            end: str | datetime | DatetimeIndex) -> np.ndarray:
    return frac_interval(start, end, 'year')


@document(_frac_doc)
def frac_mth(start: str | datetime | DatetimeIndex,
             end: str | datetime | DatetimeIndex) -> np.ndarray:
    return frac_interval(start, end, 'month')


@document(_frac_doc)
def frac_qtr(start: str | datetime | DatetimeIndex,
             end: str | datetime | DatetimeIndex) -> np.ndarray:
    return frac_interval(start, end, 'quarter')


@document(_frac_doc)
def frac_wk(start: str | datetime | DatetimeIndex,
            end: str | datetime | DatetimeIndex) -> np.ndarray:
    return frac_interval(start, end, 'week')
