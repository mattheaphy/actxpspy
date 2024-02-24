from warnings import warn
from actxps import ExposedDF
from actxps.tools import (
    relocate,
    _verify_exposed_df
)
import polars as pl


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
    records. Calendar year exposures are created by passing `cal_expo=True` to 
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

        target_status = pl.Series(expo.target_status, dtype=str)
        default_status = expo.default_status
        date_cols = expo.date_cols
        expo_length = expo.expo_length

        def pol_frac(x: pl.Expr, start: str | pl.Expr,
                     end: str | pl.Expr, y: pl.Expr = None):

            if isinstance(start, str):
                start = pl.col(start)
            if isinstance(end, str):
                end = pl.col(end)
            if y is None:
                y = start

            return (((x - y).dt.total_days() + 1) /
                    ((end - start).dt.total_days() + 1))

        def cal_frac(x: pl.Expr):
            return pol_frac(x, 'cal_b', 'cal_e')

        def add_yr(x: pl.Expr, n: pl.Expr):
            return (x.dt.offset_by(pl.format('{}y', n)))

        # time fractions
        # h = yearfrac from boy to anniv
        # v = yearfrac from boy to term

        data = expo.data.clone().lazy()
        # temporary generic date column names
        data = (
            data.rename({date_cols[0]: 'cal_b',
                         date_cols[1]: 'cal_e'}).
            with_columns(
                anniv=add_yr(pl.col('issue_date'),
                             pl.col('cal_b').dt.year() -
                             pl.col('issue_date').dt.year())
            ).
            with_columns(
                split=pl.col('anniv').is_between(pl.col('cal_b'),
                                                 pl.col('cal_e')),
                h=cal_frac(pl.col('anniv').dt.offset_by('-1d')),
                v=cal_frac(pl.col('term_date'))
            ).collect())

        pre_anniv = (
            data.clone().lazy().
            filter(pl.col('split')).
            with_columns(
                piece=1,
                cal_b=pl.max_horizontal(pl.col('issue_date'),
                                        pl.col('cal_b')),
                cal_e=pl.col('anniv').dt.offset_by('-1d'),
                exposure=pl.col('h'),
                exposure_pol=1 - pol_frac(
                    pl.col('cal_b').dt.offset_by('-1d'),
                    add_yr(pl.col('anniv'), pl.lit(-1)),
                    pl.col('anniv').dt.offset_by('-1d')
                )
            )
        )

        post_anniv = (
            data.clone().lazy().
            with_columns(
                piece=2,
                cal_b=(pl.when(pl.col('split')).
                       then(pl.col('anniv')).
                       otherwise(pl.col('cal_b'))),
                exposure=(pl.when(pl.col('split')).
                          then(1 - pl.col('h')).
                          otherwise(1)),
                anniv=(pl.when(pl.col('anniv') > pl.col('cal_e')).
                       then(add_yr(pl.col('anniv'), pl.lit(-1))).
                       otherwise(pl.col('anniv')))).
            with_columns(
                exposure_pol=pol_frac(
                    pl.col('cal_e'),
                    'anniv',
                    add_yr(pl.col('anniv'), pl.lit(1)).dt.offset_by('-1d'),
                    pl.col('cal_b'))
            )
        )

        data = (
            pl.concat([pre_anniv, post_anniv], how='vertical').
            filter((pl.col('cal_b') <= pl.col('cal_e')) &
                   (pl.col('term_date').is_null() |
                    (pl.col('term_date') >= pl.col('cal_b')))).
            with_columns(
                term_date=(pl.when(pl.col('term_date').
                                   is_between(pl.col('cal_b'),
                                              pl.col('cal_e'))).
                           then(pl.col('term_date')).
                           otherwise(None)),
                pol_yr=(pl.col('anniv').dt.year() -
                        pl.col('issue_date').dt.year() + pl.col('piece') - 1)).
            with_columns(
                status=(pl.when(pl.col('term_date').is_null()).
                        then(pl.lit(default_status)).
                        otherwise(pl.col('status')))).
            with_columns(
                claims=pl.col('status').is_in(target_status)
            ).
            with_columns(
                exposure_cal=(pl.when(pl.col('claims')).
                              then(
                                  pl.when(
                                      (pl.col('piece') == 1) |
                                      (pl.col('cal_b') == pl.col('issue_date'))
                                  ).
                                  then(1).
                                  otherwise(1 - pl.col('h'))).
                              when(pl.col('term_date').is_null()).
                              then(pl.col('exposure')).
                              when(pl.col('piece') == 1).
                              then(pl.col('v')).
                              otherwise(pl.col('v') - pl.col('h'))),

                exposure_pol=(pl.when(pl.col('claims')).
                              then(
                                  pl.when(pl.col('piece') == 1).
                                  then(pl.col('exposure_pol')).
                                  when(pl.col('split')).
                                  then(1).
                                  otherwise(1 - pol_frac(
                                      pl.col('cal_b').dt.offset_by('-1d'),
                                      'anniv',
                                      (add_yr(pl.col('anniv'), pl.lit(1)).
                                       dt.offset_by('-1d'))
                                  ))).
                              when(pl.col('term_date').is_null()).
                              then(pl.col('exposure_pol')).
                              when(pl.col('piece') == 1).
                              then(pol_frac(
                                  pl.col('term_date'),
                                  add_yr(pl.col('anniv'), pl.lit(-1)),
                                  pl.col('anniv').dt.offset_by('-1d')) -
                    (1 - pl.col('exposure_pol'))).
                    otherwise(pol_frac(
                        pl.col('term_date'),
                        'anniv',
                        (add_yr(pl.col('anniv'), pl.lit(1)).
                         dt.offset_by('-1d'))
                    ))
                )
            ).
            sort('pol_num', 'cal_b', 'piece').
            drop(['h', 'v', 'split', 'anniv', 'claims', 'exposure', 'piece'])
        )

        data = (relocate(data, 'pol_yr', after='cal_e').
                # restore date column names
                rename({'cal_b': date_cols[0],
                        'cal_e': date_cols[1]}).
                collect())

        self._finalize(data, expo.end_date, expo.start_date,
                       target_status, True, expo_length,
                       expo.trx_types, default_status, True)

        return None


def _check_split_expose_basis(obj, col_exposure):
    """
    This internal function sends an error if a `SplitExposedDF` is passed
    without clarifying which exposure basis should be used.
    """
    if isinstance(obj, SplitExposedDF):
        assert col_exposure in ["exposure_cal", "exposure_pol"], \
            'A `SplitExposedDF` was passed without clarifying which ' + \
            'exposure basis should be used to summarize results. Hint: ' + \
            'Pass "exposure_pol" to `col_exposure` for policy year ' + \
            'exposures pass "exposure_cal" to `col_exposure` for calendar ' + \
            'exposures.'
