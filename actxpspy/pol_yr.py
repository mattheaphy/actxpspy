import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from dateutil.relativedelta import relativedelta
from datetime import datetime
from actxpspy.tools import document

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
  Duration length. Only applies to `pol_duration()`. Must be 'Y' (year), 
  'Q' (quarter), 'M' (month), or 'W' (week)

## Returns 

An integer vector

"""

@document(_pol_doc)
def pol_interval(x: str | datetime | DatetimeIndex, 
                 issue_date: str | datetime | DatetimeIndex,
                 dur_length: str) -> np.ndarray:
  
  
  assert dur_length in list('YQMW'), \
    "`dur_length` must be 'Y' (year), 'Q' (quarter), 'M' (month), or 'W' (week)"
  
  if not isinstance(x, DatetimeIndex):
    x = pd.to_datetime(x)
  if not isinstance(issue_date, DatetimeIndex):
    issue_date = pd.to_datetime(issue_date)
    
  dat = pd.DataFrame({
    'issue_date': issue_date,
    'x': x
  }, index=np.arange(max(len2(x), len2(issue_date))))
  
  if dur_length == "Y":
    res = [relativedelta(a, b).years for a, b in zip(dat.x, dat.issue_date)]
    
  elif dur_length in ["M", "Q"]:
    def mth_calc(a, b):
      delta = relativedelta(a, b)
      return 12 * delta.years + delta.months

    if dur_length == "Q":
      res = [mth_calc(a, b) // 3 for a, b in zip(dat.x, dat.issue_date)]
    else:
      res = [mth_calc(a, b) for a, b in zip(dat.x, dat.issue_date)]
      
  else:
    res = (dat.x - dat.issue_date).dt.days // 7
  
  return np.array(res) + 1
  
@document(_pol_doc)
def pol_yr(x: str | datetime | DatetimeIndex, 
           issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
  return pol_interval(x, issue_date, 'Y')

@document(_pol_doc)
def pol_mth(x: str | datetime | DatetimeIndex, 
            issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
  return pol_interval(x, issue_date, 'M')

@document(_pol_doc)
def pol_qtr(x: str | datetime | DatetimeIndex, 
            issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
  return pol_interval(x, issue_date, 'Q')

@document(_pol_doc)
def pol_wk(x: str | datetime | DatetimeIndex, 
           issue_date: str | datetime | DatetimeIndex) -> np.ndarray:
  return pol_interval(x, issue_date, 'W')

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