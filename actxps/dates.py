import polars as pl
from datetime import datetime, date
from actxps.tools import arg_match


def pol_interval(dates: str | date | list | pl.Series,
                 issue_date:  str | date | list | pl.Series,
                 dur_length: str) -> pl.Series:
    """
    Calculate policy durations in years, quarters, months, or weeks

    This function assumes the first day of each policy year is the
    anniversary date (or issue date in the first year). The last day of each
    policy year is the day before the next anniversary date. Analogous rules
    are used for policy quarters, policy months, and policy weeks.

    Parameters
    ----------
    dates : str | date | list | pl.Series
        Date(s)
    issue_date : str | date | list | pl.Series
        Issue date(s)
    dur_length : {'year', 'quarter', 'month', 'week'}
        Policy duration length

    Returns 
    ----------
    pl.Series

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

    if dur_length == 'year':
        interval = '1y'
    elif dur_length == 'quarter':
        interval = '1q'
    elif dur_length == 'month':
        interval = '1mo'
    else:
        interval = '1w'

    # create ranges of dates from start to <end
    return pl.date_ranges(issue_date, dates, interval, eager=True).list.len()


def pol_yr(dates: str | date | list | pl.Series,
           issue_date: str | date | list | pl.Series) -> pl.Series:
    """
    Calculate policy years

    Parameters
    ----------
    dates : str | date | list | pl.Series
        Date(s)
    issue_date : str | date | list | pl.Series
        Issue date(s)

    Returns 
    ----------
    pl.Series

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


def pol_mth(dates: str | date | list | pl.Series,
            issue_date: str | date | list | pl.Series) -> \
        pl.Series:
    """
    Calculate policy months

    Parameters
    ----------
    dates : str | date | list | pl.Series
        Date(s)
    issue_date : str | date | list | pl.Series
        Issue date(s)

    Returns 
    ----------
    pl.Series

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


def pol_qtr(dates: str | date | list | pl.Series,
            issue_date: str | date | list | pl.Series) -> \
        pl.Series:
    """
    Calculate policy quarters

    Parameters
    ----------
    dates : str | date | list | pl.Series
        Date(s)
    issue_date : str | date | list | pl.Series
        Issue date(s)

    Returns 
    ----------
    pl.Series

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


def pol_wk(dates: str | date | list | pl.Series,
           issue_date: str | date | list | pl.Series) -> \
        pl.Series:
    """
    Calculate policy weeks

    Parameters
    ----------
    dates : str | date | list | pl.Series
        Date(s)
    issue_date : str | date | list | pl.Series
        Issue date(s)

    Returns 
    ----------
    pl.Series

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


def _len2(x, default: int = 1) -> int:
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


def _delta_frac(start: pl.Series, end: pl.Series, dur_length: str,
                cal_periods: bool = False) -> pl.Series:
    """
    Internal function for calculating fractional durations.

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
    cal_periods : bool, default=False
        If `True`, then fractional year, quarter, or month periods are
        determined based on calendar periods.

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
    # number of complete periods
    n = ranges.list.len() - 1
    # left bounding date
    l = ranges.list.last()
    # right bounding date. Note that we're not simply adding a year to l because
    #   doing so would create a difference of 1 day when `l` == 2/28 and the
    #   following year is a leap year and `start` is a leap day.
    r = ranges.list.first().dt.offset_by((n + 1).cast(str) + interval[1:])
    if cal_periods:
        l = l.dt.month_end()
        r = r.dt.month_end()
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


def _convert_date(x) -> pl.Series:
    """
    Helper function to convert inputs to dates. 
    This function should not be called directly.
    """
    if not isinstance(x, pl.Series):
        if _len2(x, None) is None:
            x = [x]
        x = pl.Series(x)
    if not isinstance(x.dtype, pl.Date):
        x = x.str.to_date('%Y-%m-%d')
    return x


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
