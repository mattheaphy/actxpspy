from warnings import warn
from actxps.expose import ExposedDF
from actxps.dates import add_yr
from actxps.tools import (
    relocate,
    _verify_exposed_df
)
import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np


class SplitExposedDF(ExposedDF):
    """
    Split calendar exposures by policy year

    Split calendar period exposures that cross a policy anniversary
    into a pre-anniversary record and a post-anniversary record.

    Parameters
    ----------
    expo : ExposedDF
        An exposure data frame class with calendar-based exposure periods

    Notes
    ----------
    The `ExposedDF` must have calendar year, quarter, month, or week exposure
    records. Calendar year exposures are created by passing `cal_expo = True` to 
    `ExposedDF` (or alternatively, with the class methods 
    `ExposedDF.expose_cy()`, `ExposedDF.expose_cq()`, `ExposedDF.expose_cm()`, 
    and `ExposedDF.expose_cw()`).

    After splitting, the resulting data will contain both calendar exposures and 
    policy year exposures. These columns will be named 'exposure_cal' and
    'exposure_pol', respectively. Calendar exposures will be in the original 
    units passed to `SplitExposedDF()`. Policy exposures will always be 
    expressed in years. Downstream functions like `exp_stats()` and 
    `exp_shiny()` will require clarification as to which exposure basis should 
    be used to summarize results.    

    After splitting, the column 'pol_yr' will contain policy years.

    Examples
    ----------
    ```{python}
    import actxps as xp
    toy_census = xp.load_toy_census()
    expo = xp.ExposedDF.expose_cy(toy_census, "2022-12-31")
    xp.SplitExposedDF(expo)
    ```

    See Also
    ----------
    `ExposedDF()` for more information on exposure data.
    """

    def __init__(self, expo: ExposedDF):

        _verify_exposed_df(expo)

        if not expo.cal_expo:
            raise AssertionError(
                '`SplitExposedDF()` can only be used with an ' +
                '`ExposedDF` object that has calendar exposures.' +
                ' Hint: Try creating an `ExposedDF` object with ' +
                'the argument `cal_expo` set to `True` before ' +
                'calling `SplitExposedDF()`.')

        if len(expo.trx_types) > 0:
            warn('Transactions have already been attached to this ' +
                 '`ExposedDF`. This will lead to duplication of transactions ' +
                 'after exposures are split. The appropriate order of ' +
                 'operations is to call `add_transactions()` after ' +
                 '`SplitExposedDF()`.')

        target_status = expo.target_status
        default_status = expo.default_status
        date_cols = expo.date_cols
        expo_length = expo.expo_length

        def pol_frac(x, start, end, y=None):

            if y is None:
                return (x - start + Day(1)) / (end - start + Day(1))
            else:
                return (x - y) / (end - start + Day(1))

        def cal_frac(x):
            return pol_frac(x, data.cal_b, data.cal_e)

        # time fractions
        # h = yearfrac from boy to anniv
        # v = yearfrac from boy to term

        data = expo.data
        # temporary generic date column names
        data = data.rename(columns={date_cols[0]: 'cal_b',
                                    date_cols[1]: 'cal_e'})
        data['anniv'] = add_yr(data.issue_date,
                               data.cal_b.dt.year - data.issue_date.dt.year)
        data['split'] = data.anniv.between(data.cal_b, data.cal_e)
        data['h'] = cal_frac(data.anniv - Day(1))
        data['v'] = cal_frac(data.term_date)

        pre_anniv = data.copy()[data.split]
        pre_anniv['piece'] = 1
        pre_anniv['cal_b'] = np.maximum(pre_anniv.issue_date, pre_anniv.cal_b)
        pre_anniv['cal_e'] = pre_anniv.anniv - Day(1)
        pre_anniv['exposure'] = pre_anniv.h
        pre_anniv['exposure_pol'] = 1 - pol_frac(pre_anniv.cal_b - Day(1),
                                                 add_yr(pre_anniv.anniv, -1),
                                                 pre_anniv.anniv - Day(1))

        post_anniv = data.copy()
        post_anniv['piece'] = 2
        post_anniv['cal_b'] = np.where(post_anniv.split,
                                       post_anniv.anniv, post_anniv.cal_b)
        post_anniv['exposure'] = np.where(
            post_anniv.split, 1 - post_anniv.h, 1)
        post_anniv['anniv'] = np.where(post_anniv.anniv > post_anniv.cal_e,
                                       add_yr(post_anniv.anniv, -1),
                                       post_anniv.anniv)
        post_anniv['exposure_pol'] = pol_frac(post_anniv.cal_e,
                                              post_anniv.anniv,
                                              add_yr(post_anniv.anniv, 1) -
                                              np.timedelta64(1, 'D'),
                                              post_anniv.cal_b - Day(1))

        data = pd.concat((pre_anniv, post_anniv))
        data = data[(data.cal_b <= data.cal_e) &
                    (pd.isna(data.term_date) |
                    (data.term_date >= data.cal_b))]
        data['term_date'] = pd.to_datetime(
            np.where(data.term_date.between(data.cal_b, data.cal_e),
                     data.term_date, pd.NA))
        data['pol_yr'] = data.anniv.dt.year - data.issue_date.dt.year + \
            data.piece - 1
        data['status'] = np.where(pd.isna(data.term_date),
                                  pd.Categorical([default_status],
                                                 data.status.cat.categories),
                                  data.status)
        data['claims'] = data.status.isin(target_status)
        data['exposure_cal'] = np.select(
            [data.claims, pd.isna(data.term_date), data.piece == 1],
            [np.where((data.piece == 1) | (data.cal_b == data.issue_date),
                      1, 1 - data.h),
             data.exposure, data.v],
            default=data.v - data.h
        )
        data['exposure_pol'] = np.select(
            [data.claims, pd.isna(data.term_date), data.piece == 1],
            [np.select([data.piece == 1, data.split],
                       [data.exposure_pol, 1],
                       default=1 - pol_frac(data.cal_b - Day(1),
                                            data.anniv,
                                            add_yr(data.anniv, 1) -
                                            np.timedelta64(1, 'D'))),
             data.exposure_pol,
             pol_frac(data.term_date,
                      add_yr(data.anniv, -1),
                      data.anniv - Day(1)) - (1 - data.exposure_pol)],
            default=pol_frac(data.term_date, data.anniv,
                             add_yr(data.anniv, 1) - np.timedelta64(1, 'D')))

        data.sort_values(['pol_num', 'cal_b', 'piece'], inplace=True)
        data.drop(columns={'h', 'v', 'split', 'anniv', 'claims',
                           'exposure', 'piece'}, inplace=True)
        data = relocate(data, 'pol_yr', after='cal_e')
        # # restore date column names
        data.rename(columns={'cal_b': date_cols[0],
                             'cal_e': date_cols[1]}, inplace=True)

        self._finalize(data, expo.end_date, expo.start_date, 
                       target_status, True, expo_length,
                       expo.trx_types, default_status, True)

        return None
