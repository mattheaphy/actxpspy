import polars as pl
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from dateutil.relativedelta import relativedelta
from datetime import datetime, date
from actxps.tools import arg_match


def pol_interval(dates: str | datetime | DatetimeIndex | pd.Series,
                 issue_date: str | datetime | DatetimeIndex | pd.Series,
                 dur_length: str) -> np.ndarray:
    """
    Calculate policy durations in years, quarters, months, or weeks

    This function assumes the first day of each policy year is the
    anniversary date (or issue date in the first year). The last day of each
    policy year is the day before the next anniversary date. Analogous rules
    are used for policy quarters, policy months, and policy weeks.

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex 
        Date(s)
    issue_date : str | datetime | DatetimeIndex
        Issue date(s)
    dur_length : {'year', 'quarter', 'month', 'week'}
        Policy duration length

    Returns 
    ----------
    np.ndarray
        A vector of integers

    See Also
    ----------
    pol_yr, pol_qtr, pol_mth, pol_wk

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.pol_interval(['2024-05-01', '2024-02-01'], 
                    ['2008-03-14', '2008-03-14'], 
                    'year')
    ```
    """
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


def pol_yr(dates: str | datetime | DatetimeIndex | pd.Series,
           issue_date: str | datetime | DatetimeIndex | pd.Series) -> np.ndarray:
    """
    Calculate policy years

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex 
        Date(s)
    issue_date : str | datetime | DatetimeIndex
        Issue date(s)

    Returns 
    ----------
    np.ndarray
        A vector of integers

    See Also
    ----------
    pol_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.pol_yr(['2024-05-01', '2024-02-01'], 
              ['2008-03-14', '2008-03-14'])
    ```    
    """
    return pol_interval(dates, issue_date, 'year')


def pol_mth(dates: str | datetime | DatetimeIndex | pd.Series,
            issue_date: str | datetime | DatetimeIndex | pd.Series) -> \
        np.ndarray:
    """
    Calculate policy months

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex 
        Date(s)
    issue_date : str | datetime | DatetimeIndex
        Issue date(s)

    Returns 
    ----------
    np.ndarray
        A vector of integers

    See Also
    ----------
    pol_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.pol_mth(['2024-05-01', '2024-02-01'], 
               ['2008-03-14', '2008-03-14'])
    ```        
    """
    return pol_interval(dates, issue_date, 'month')


def pol_qtr(dates: str | datetime | DatetimeIndex | pd.Series,
            issue_date: str | datetime | DatetimeIndex | pd.Series) -> \
        np.ndarray:
    """
    Calculate policy quarters

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex 
        Date(s)
    issue_date : str | datetime | DatetimeIndex
        Issue date(s)

    Returns 
    ----------
    np.ndarray
        A vector of integers

    See Also
    ----------
    pol_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.pol_qtr(['2024-05-01', '2024-02-01'], 
               ['2008-03-14', '2008-03-14'])
    ```            
    """
    return pol_interval(dates, issue_date, 'quarter')


def pol_wk(dates: str | datetime | DatetimeIndex | pd.Series,
           issue_date: str | datetime | DatetimeIndex | pd.Series) -> \
        np.ndarray:
    """
    Calculate policy weeks

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex
        Date(s)
    issue_date : str | datetime | DatetimeIndex
        Issue date(s)

    Returns 
    ----------
    np.ndarray
        A vector of integers

    See Also
    ----------
    pol_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.pol_wk(['2024-05-01', '2024-02-01'], 
              ['2008-03-14', '2008-03-14'])
    ```            
    """
    return pol_interval(dates, issue_date, 'week')


def len2(x, default: int = 1) -> int:
    """
    Length function with non-iterables and strings returning 1

    Parameters
    ----------
    x : Any
    default : int
        Default value to return when x has no length

    Returns
    ----------
    int
        `default` if `x` is not iterable or if `x` is a string. Otherwise, 
        the length of `x`.
    """
    if isinstance(x, str):
        return default
    try:
        return len(x)
    except TypeError:
        return default


def frac_interval(start: str | date | list | pl.Series,
                  end: str | date | list | pl.Series,
                  dur_length: str) -> pl.Series:
    """
    Calculate fractional years, quarters, months, or weeks between two dates

    Parameters
    ----------
    start : str | date | list | pl.Series
        Start dates
    end : str | date | list | pl.Series
        End dates
    dur_length : {'year', 'quarter', 'month', 'week'}
        Duration length

    Returns 
    ----------
    pl.Series
        A series of floats

    See Also
    ----------
    frac_yr, frac_qtr, frac_mth, frac_wk

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.frac_interval(['2008-03-14', '2008-03-14'],
                     ['2024-05-01', '2024-02-01'], 
                     'year')
    ```            
    """

    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    start = _convert_date(start)
    end = _convert_date(end)

    return _delta_frac(start, end, dur_length)


def _delta_frac(start: pl.Series, end: pl.Series, dur_length: str) -> pl.Series:
    """
    Internal function for calculating one set of fractional durations.

    This function is used by `frac_interval()` and is not meant to be called
    directly.

    Parameters
    ----------
    start : pl.Series
        Start date
    end : pl.Series
        End date
    dur_length : str
        Duration length. Only applies to `frac_interval()`. Must be 'year', 
        'quarter', 'month', or 'week'

    Returns
    ----------
    pl.Series
    """
    if dur_length == 'week':
        return (end - start).dt.total_days() / 7

    if dur_length == 'year':
        interval = '1y'
    elif dur_length == 'quarter':
        interval = '1q'
    else:
        interval = '1mo'

    # create ranges of dates from start to <end
    ranges = pl.date_ranges(start, end, interval, eager=True)
    # left bounding date
    l = ranges.list.last()
    # right bounding date
    r = l.dt.offset_by(interval)
    # number of complete periods
    n = ranges.list.len() - 1
    # complete periods + fractional period
    res = (n + (end - l).dt.total_days() / (r - l).dt.total_days())

    return res


def frac_yr(start: str | date | list | pl.Series,
            end: str | date | list | pl.Series) -> pl.Series:
    """
    Calculate fractional years between two dates

    Parameters
    ----------
    start :  str | date | list | pl.Series
        Start dates
    end :  str | date | list | pl.Series
        End dates

    Returns 
    ----------
    pl.Series

    See Also
    ----------
    frac_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.frac_yr(['2008-03-14', '2008-03-14'],
               ['2024-05-01', '2024-02-01'])
    ```                
    """
    return frac_interval(start, end, 'year')


def frac_mth(start: str | date | list | pl.Series,
             end: str | date | list | pl.Series) -> pl.Series:
    """
    Calculate fractional months between two dates

    Parameters
    ----------
    start :  str | date | list | pl.Series
        Start dates
    end :  str | date | list | pl.Series
        End dates

    Returns 
    ----------
    pl.Series

    See Also
    ----------
    frac_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.frac_mth(['2008-03-14', '2008-03-14'],
                ['2024-05-01', '2024-02-01'])
    ```                    
    """
    return frac_interval(start, end, 'month')


def frac_qtr(start: str | date | list | pl.Series,
             end: str | date | list | pl.Series) -> pl.Series:
    """
    Calculate fractional quarters between two dates

    Parameters
    ----------
    start : str | date | list | pl.Series
        Start dates
    end : str | date | list | pl.Series
        End dates

    Returns 
    ----------
    pl.Series

    See Also
    ----------
    frac_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.frac_qtr(['2008-03-14', '2008-03-14'],
                ['2024-05-01', '2024-02-01'])
    ```                        
    """
    return frac_interval(start, end, 'quarter')


def frac_wk(start: str | date | list | pl.Series,
            end: str | date | list | pl.Series) -> pl.Series:
    """
    Calculate fractional weeks between two dates

    Parameters
    ----------
    start : str | date | list | pl.Series
        Start dates
    end : str | date | list | pl.Series
        End dates

    Returns 
    ----------
    pl.Series

    See Also
    ----------
    frac_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.frac_wk(['2008-03-14', '2008-03-14'],
               ['2024-05-01', '2024-02-01'])
    ```                        
    """
    return frac_interval(start, end, 'week')


def _convert_date(x):
    """
    Helper function to convert inputs to dates. 
    This function should not be called directly.
    """
    if not isinstance(x, pl.Series):
        if len2(x, None) is None:
            x = [x]
        x = pl.Series(x)
    if not isinstance(x.dtype, pl.Date):
        x = x.str.to_date('%Y-%m-%d')
    return x


def add_interval(dates: str | datetime | DatetimeIndex | pd.Series,
                 x: pd.Series | np.ndarray,
                 dur_length: str) -> np.ndarray:
    """
    Add years, quarters, months, or weeks to a vector of dates

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex | pd.Series
        Dates
    x : int | pd.Series | np.ndarray
        Number of periods to add
    dur_length : {'year', 'quarter', 'month', 'week'}
        Duration length

    Returns 
    ----------
    np.ndarray

    See Also
    ----------
    add_yr, add_qtr, add_mth, add_wk

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.add_interval(['2008-03-14', '2008-03-14'], [16, 15], 'year')
    ```                        
    """
    arg_match('dur_length', dur_length, ['year', 'quarter', 'month', 'week'])

    dates = _convert_date(dates)

    dat = pd.DataFrame({
        'dates': dates,
        'x': x
    }, index=np.arange(max(len2(dates), len2(x))))
    dates = dat.dates

    x = dat.x.values
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


def add_yr(dates: str | datetime | DatetimeIndex | pd.Series,
           x: pd.Series | np.ndarray) -> np.ndarray:
    """
    Add years to a vector of dates

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex | pd.Series
        Dates
    x : int | pd.Series | np.ndarray
        Number of periods to add

    Returns 
    ----------
    np.ndarray

    See Also
    ----------
    add_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.add_yr(['2008-03-14', '2008-03-14'], [16, 15])
    ```                            
    """
    return add_interval(dates, x, 'year')


def add_qtr(dates: str | datetime | DatetimeIndex | pd.Series,
            x: pd.Series | np.ndarray) -> np.ndarray:
    """
    Add quarters to a vector of dates

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex | pd.Series
        Dates
    x : int | pd.Series | np.ndarray
        Number of periods to add

    Returns 
    ----------
    np.ndarray

    See Also
    ----------
    add_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.add_qtr(['2008-03-14', '2008-03-14'], [64, 60])
    ```                                
    """
    return add_interval(dates, x, 'quarter')


def add_mth(dates: str | datetime | DatetimeIndex | pd.Series,
            x: pd.Series | np.ndarray) -> np.ndarray:
    """
    Add months to a vector of dates

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex | pd.Series
        Dates
    x : int | pd.Series | np.ndarray
        Number of periods to add

    Returns 
    ----------
    np.ndarray

    See Also
    ----------
    add_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.add_mth(['2008-03-14', '2008-03-14'], [192, 180])
    ```                                
    """
    return add_interval(dates, x, 'month')


def add_wk(dates: str | datetime | DatetimeIndex | pd.Series,
           x: pd.Series | np.ndarray) -> np.ndarray:
    """
    Add weeks to a vector of dates

    Parameters
    ----------
    dates : str | datetime | DatetimeIndex | pd.Series
        Dates
    x : int | pd.Series | np.ndarray
        Number of periods to add

    Returns 
    ----------
    np.ndarray

    See Also
    ----------
    add_interval

    Examples
    ----------
    ```{python}
    import actxps as xp
    xp.add_wk(['2008-03-14', '2008-03-14'], [835, 783])
    ```                                
    """
    return add_interval(dates, x, 'week')


def _date_str(x: date | str, x_name: str = "x") -> date:
    """
    Internal function for converting strings to dates if necessary.

    Parameters
    ----------
    x : date | str
        A date object or a string in %Y-%m-%d format.
    x_name : str, default="x"
        An optional variable name to print for error messages.

    Returns
    -------
    date
    """
    if isinstance(x, date):
        return x
    assert isinstance(x, str), f"{x_name} must be a date or string " \
        "in %Y-%m-%d format."
    return datetime.strptime(x, '%Y-%m-%d').date()
