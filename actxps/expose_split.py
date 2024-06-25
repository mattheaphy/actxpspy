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
        start_date = expo.start_date
        end_date = expo.end_date
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
            return pol_frac(x, 'hold_cal_b', 'hold_cal_e')

        def add_yr(x: pl.Expr, n: pl.Expr):
            return (x.dt.offset_by(pl.format('{}y', n)))

        # time fractions
        # b = fraction from boy to cal_b
        #     - usually zero except for new contracts and a truncated start date
        # h = fraction from boy to anniv
        # v = fraction from boy to the earlier of termination and cal_e

        data = expo.data.lazy()
        # temporary generic date column names
        data = (
            data.rename({date_cols[0]: 'cal_b',
                         date_cols[1]: 'cal_e'}).
            with_columns(
                pol_yr=(pl.col('cal_b').dt.year() -
                        pl.col('issue_date').dt.year())
            ).
            with_columns(
                anniv=add_yr(pl.col('issue_date'), pl.col('pol_yr'))
            ).
            with_columns(
                hold_cal_b=pl.col('cal_b'),
                hold_cal_e=pl.col('cal_e'),
                split=pl.col('anniv').is_between(pl.col('cal_b'),
                                                 pl.col('cal_e')),
                cal_b=pl.max_horizontal(start_date, 'issue_date', 'cal_b'),
                cal_e=pl.min_horizontal(end_date, 'cal_e')
            ).with_columns(
                b=cal_frac(pl.col('cal_b').dt.offset_by('-1d')),
                h=(pl.when(pl.col('split')).
                   then(cal_frac(pl.col('anniv').dt.offset_by('-1d'))).
                   otherwise(0)),
                v=cal_frac(pl.when(pl.col('term_date').is_null()).
                           then(pl.col('cal_e')).
                           otherwise(pl.col('term_date')))
            ).collect())

        pre_anniv = (
            data.lazy().
            filter(pl.col('split')).
            with_columns(
                piece=1,
                next_anniv=pl.col('anniv')
            ).with_columns(
                cal_e=pl.min_horizontal(
                    end_date, pl.col('next_anniv').dt.offset_by('-1d')),
                exposure=pl.min_horizontal('h', 'v') - pl.col('b')
            )
        )

        post_anniv = (
            data.lazy().
            with_columns(
                piece=2,
                cal_b=(pl.when(pl.col('split')).
                       then(pl.max_horizontal('anniv', start_date)).
                       otherwise(pl.col('cal_b')))
            ).with_columns(
                pol_yr=pl.col('pol_yr') + (pl.col('cal_b') >= pl.col('anniv'))
            ).with_columns(
                exposure=pl.col('v') - pl.max_horizontal('h', 'b'),
                next_anniv=add_yr(pl.col('issue_date'), pl.col('pol_yr'))
            )
        )

        data = (
            pl.concat([pre_anniv, post_anniv], how='vertical').
            filter((pl.col('cal_b') <= pl.col('cal_e')) &
                   (pl.col('term_date').is_null() |
                    (pl.col('term_date') >= pl.col('cal_b')))).
            with_columns(
                anniv=add_yr(pl.col('issue_date'), pl.col('pol_yr') - 1),
                term_date=(pl.when(pl.col('term_date').
                                   is_between(pl.col('cal_b'),
                                              pl.col('cal_e'))).
                           then(pl.col('term_date')).
                           otherwise(None))).
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
                                      (pl.col('cal_b') == pl.col('issue_date')) |
                                      (pl.col('cal_b') == start_date)
                                  ).
                                  then(1).
                                  otherwise(1 - (pl.col('h') - pl.col('b')))).
                              when(pl.col('term_date').is_null()).
                              then(pl.col('exposure')).
                              when(pl.col('piece') == 1).
                              then(pl.col('v') - pl.col('b')).
                              otherwise(pl.col('v') -
                                        pl.max_horizontal('h', 'b'))),

                exposure_pol=(
                    pl.when(pl.col('claims')).
                    then((1 - pol_frac(
                        pl.col('cal_b').dt.offset_by('-1d'),
                        'anniv',
                        pl.col('next_anniv').dt.offset_by('-1d'))
                    )).
                    otherwise(pol_frac(
                        pl.min_horizontal('cal_e', 'term_date'),
                        'anniv',
                        pl.col('next_anniv').dt.offset_by('-1d'),
                        pl.col('cal_b')#.dt.offset_by('-1d')
                    ))
                )
            ).
            sort('pol_num', 'cal_b', 'piece').
            drop(['b', 'h', 'v', 'split', 'anniv', 'next_anniv', 'claims',
                  'exposure', 'piece', 'hold_cal_b', 'hold_cal_e'])
        )

        data = (relocate(data, 'pol_yr', after='cal_e').
                # restore date column names
                rename({'cal_b': date_cols[0],
                        'cal_e': date_cols[1]}).
                collect())

        self._finalize(data, expo.end_date, expo.start_date,
                       target_status.to_numpy(), True, expo_length,
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
