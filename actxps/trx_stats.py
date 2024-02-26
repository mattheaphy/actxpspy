import polars as pl
import pandas as pd
import numpy as np
from datetime import date
from functools import singledispatchmethod
from itertools import product
from actxps.expose import ExposedDF
from actxps.tools import (
    _plot_experience,
    _pivot_plot_special,
    _verify_exposed_df,
    _conf_int_warning,
    _verify_col_names,
    _date_str,
    _qbinom,
    _qnorm,
    _check_convert_df
)
from actxps.col_select import (
    col_matches,
    col_starts_with
)
from actxps.expose_split import _check_split_expose_basis
from actxps.dates import _len2
from plotnine import aes
from great_tables import (
    GT,
    pct,
    md
)
from scipy.stats import binom


class TrxStats():
    """
    Transactions study summary class

    Create a summary of transaction counts, amounts, and utilization rates
    (a `TrxStats` object).

    Typically, the `TrxStats` class constructor should not be called directly.
    The preferred method for creating a `TrxStats` object is to call the
    `trx_stats()` method on an `ExposedDF` object.    

    Parameters
    ----------
    expo : ExposedDF
        An exposed data frame class
    trx_types : list or str, default=None
        A list of transaction types to include in the output. If `None` is
        provided, all available transaction types in the `trx_types` 
        property of `expo` will be used.
    percent_of : list | str, default=None
        A optional list containing column names in the `data` property of `expo`
        to use as denominators in the calculation of utilization rates or
        actual-to-expected ratios.
    combine_trx : bool, default=False
        If `False`, the results will contain output rows for each 
        transaction type. If `True`, the results will contain aggregated
        experience across all transaction types.
    full_exposures_only : bool, default=True
        If `True`, partially exposed records will be ignored 
        in the results.
    conf_int : bool, default=False 
        If `True`, the output will include confidence intervals around the
        observed utilization rate and any `percent_of` output columns.
    conf_level : float, default=0.95 
        Confidence level for confidence intervals
    col_exposure : str, default='exposure'
        Name of the column in the `data` property of `expo` containing exposures        


    Attributes
    ----------

    data : pl.DataFrame
        A data framethat includes columns for any grouping variables and
        transaction types, plus the following: `trx_n` (the number of unique 
        transactions), `trx_amt` (total transaction amount), `trx_flag` (the 
        number of observation periods with non-zero transaction amounts), 
        `exposure` (total exposures), `avg_trx` (mean transaction amount 
        {`trx_amt / trx_flag`}), `avg_all` (mean transaction amount over all 
        records {`trx_amt / exposure`}), `trx_freq` (transaction frequency when 
        a transaction occurs {`trx_n / trx_flag`}), `trx_utilization` 
        (transaction utilization per observation period 
        {`trx_flag / exposure`}). If `percent_of` is provided, the results will 
        also include the sum of any columns passed to `percent_of` with 
        non-zero transactions (these columns include the suffix `_w_trx`.
        - The sum of any columns passed to `percent_of`), `pct_of_{*}_w_trx` 
        (total transactions as a percentage of column `{*}_w_trx`), 
        `pct_of_{*}_all` (total transactions as a percentage of column `{*}`).


    Notes
    ----------
    If the `ExposedDF` object is grouped (see the `group_by()` method), the
    returned `TrxStats` object's data will contain one row per group.

    Any number of transaction types can be passed to the `trx_types` 
    argument, however each transaction type **must** appear in the 
    `trx_types` property of the `ExposedDF` object. In addition, 
    `trx_stats()` expects to see columns named `trx_n_{*}`
    (for transaction counts) and `trx_amt_{*}` for (transaction amounts) 
    for each transaction type. To ensure `.data` is in the appropriate 
    format, use the class method `ExposedDF.from_DataFrame()` to convert 
    an existing data frame with transactions or use `add_transactions()` 
    to attach transactions to an existing `ExposedDF` object.

    **"Percentage of" calculations**

    The `percent_of` argument is optional. If provided, this argument must
    be list with values corresponding to columns in the `data` property of 
    `expo` containing values to use as denominators in the calculation of 
    utilization rates or actual-to-expected ratios. Example usage:

    - In a study of partial withdrawal transactions, if `percent_of` refers
    to account values, observed withdrawal rates can be determined.
    - In a study of recurring claims, if `percent_of` refers to a column
    containing a maximum benefit amount, utilization rates can be 
    determined.

    **Confidence intervals**

    If `conf_int` is set to `True`, the output will contain lower and upper
    confidence interval limits for the observed utilization rate and any
    `percent_of` output columns. The confidence level is dictated
    by `conf_level`.

    - Intervals for the utilization rate (`trx_util`) assume a binomial
    distribution.
    - Intervals for transactions as a percentage of another column with
    non-zero transactions (`pct_of_{*}_w_trx`) are constructed using a normal
    distribution
    - Intervals for transactions as a percentage of another column
    regardless of transaction utilization (`pct_of_{*}_all`) are calculated
    assuming that the aggregate distribution is normal with a mean equal to
    observed transactions and a variance equal to:

        `Var(S) = E(N) * Var(X) + E(X)**2 * Var(N)`,

    Where `S` is the aggregate transactions random variable, `X` is an 
    individual transaction amount assumed to follow a normal distribution, and 
    `N` is a binomial random variable for transaction utilization.

    **Default removal of partial exposures**

    As a default, partial exposures are removed from `data` before 
    summarizing results. This is done to avoid complexity associated with a 
    lopsided skew in the timing of transactions. For example, if
    transactions can occur on a monthly basis or annually at the beginning 
    of each policy year, partial exposures may not be appropriate. If a
    policy had an exposure of 0.5 years and was taking withdrawals annually 
    at the beginning of the year, an argument could be made that the 
    exposure should instead be 1 complete year. If the same policy was 
    expected to take withdrawals 9 months into the year, it's not clear if
    the exposure should be 0.5 years or 0.5 / 0.75 years. To override this 
    treatment, set `full_exposures_only` to `False`.

    **Alternative class constructor**

    `TrxStats.from_DataFrame()` can be used to coerce a data frame containing 
    pre-aggregated experience into a `TrxStats` object. This is most useful
    for working with industry study data where individual exposure records are
    not available.    
    """
    @singledispatchmethod
    def __init__(self,
                 expo: ExposedDF,
                 trx_types: list | str = None,
                 percent_of: list | str = None,
                 combine_trx: bool = False,
                 full_exposures_only: bool = True,
                 conf_int: bool = False,
                 conf_level: float = 0.95,
                 col_exposure: str = 'exposure'):

        _verify_exposed_df(expo)
        self.data = None

        assert len(expo.trx_types) > 0, \
            ("No transactions have been attached. Add transaction data using "
             "`add_transactions()` before calling `trx_stats()`.")

        if trx_types is None:
            trx_types = expo.trx_types
        else:
            trx_types = np.atleast_1d(trx_types).tolist()
            unmatched = set(trx_types).difference(set(expo.trx_types))
            assert len(unmatched) == 0, \
                ("The following transactions do not exist in `expo`: " +
                 ", ".join(unmatched))

        _check_split_expose_basis(expo, col_exposure)

        start_date = expo.start_date
        end_date = expo.end_date
        data = expo.data.lazy().rename({col_exposure: 'exposure'})
        groups = expo.groups
        if groups is None:
            groups = []

        # remove partial exposures
        if full_exposures_only:
            data = data.filter((pl.col('exposure') - 1).abs() <=
                               np.finfo(float).eps ** 0.5)

        trx_cols = col_matches(data, f'trx_(n|amt)_({"|".join(trx_types)})')

        if combine_trx:
            trx_n_cols = [x for x in trx_cols if "_n_" in x]
            trx_amt_cols = [x for x in trx_cols if "_amt_" in x]
            data = data.with_columns(
                trx_n_All=pl.sum_horizontal(trx_n_cols),
                trx_amt_All=pl.sum_horizontal(trx_amt_cols),
            )
            trx_cols = ["trx_n_All", "trx_amt_All"]

        if percent_of is None:
            percent_of = []
        else:
            # coerce to list in case percent_of is str
            percent_of = np.atleast_1d(percent_of).tolist()

        # subset columns
        id_vars = ['index', 'pol_num', 'exposure'] + groups + percent_of
        data = (
            data.
            with_row_index('index').
            select(id_vars + trx_cols).
            # pivot longer
            melt(id_vars=id_vars).
            with_columns(
                pl.col('variable').str.replace('^trx_', '')).
            # split transaction types from kinds
            with_columns(
                pl.col('variable').str.split_exact('_', 1).
                struct.rename_fields(['kind', 'trx_type'])
            ).unnest('variable').
            collect().
            # pivot wider
            pivot(index=id_vars + ['trx_type'],
                  values='value', columns='kind').
            lazy().
            rename({'n': 'trx_n', 'amt': 'trx_amt'}).
            # fill in missing values
            with_columns(pl.col('trx_n', 'trx_amt').fill_null(0)).
            with_columns(trx_flag=pl.col('trx_n').abs() > 0).
            drop('index')
        )

        if conf_int:
            data = data.with_columns(trx_amt_sq=pl.col('trx_amt') ** 2)

        for x in percent_of:
            data = data.with_columns(
                (pl.col(percent_of) * pl.col('trx_flag')).name.map(
                    lambda x: x + '_w_trx'))

        xp_params = {'conf_level': conf_level,
                     'conf_int': conf_int}

        self._finalize(data, trx_types, percent_of,
                       groups, start_date, end_date, xp_params)

    def _finalize(self,
                  data: pl.LazyFrame | pl.DataFrame,
                  trx_types,
                  percent_of,
                  groups,
                  start_date,
                  end_date,
                  xp_params,
                  agg: bool = True):
        """
        Internal method for finalizing transaction study summary objects
        """

        # set up properties
        self.groups = groups
        self.trx_types = trx_types
        self.percent_of = percent_of
        self.start_date = _date_str(start_date)
        self.end_date = _date_str(end_date)
        self.xp_params = xp_params

        # finish trx stats
        if agg:
            self.data = self._calc(data)
        else:
            self.data = data

        return None

    def _calc(self, data:  pl.LazyFrame):
        """
        Support function for summarizing data for one group
        """

        conf_level = self.xp_params['conf_level']
        conf_int = self.xp_params['conf_int']

        if self.groups is None:
            groups = ['trx_type']
        else:
            groups = self.groups + ['trx_type']

        # dictionary of summarized values
        fields = {'trx_n': pl.col('trx_n').sum(),
                  'trx_flag': pl.col('trx_flag').sum(),
                  'trx_amt': pl.col('trx_amt').sum(),
                  'exposure': pl.col('exposure').sum()}

        if len(self.percent_of) > 0:
            for x in self.percent_of:
                xw = x + "_w_trx"
                fields[x] = pl.col(x).sum()
                fields[xw] = pl.col(xw).sum()
            if conf_int:
                fields['trx_amt_sq'] = pl.col('trx_amt_sq').sum()

        # apply summary fields
        data = (data.group_by(groups).agg(**fields).sort(groups).
                with_columns(
                    avg_trx=pl.col('trx_amt') / pl.col('trx_flag'),
                    avg_all=pl.col('trx_amt') / pl.col('exposure'),
                    trx_freq=pl.col('trx_n') / pl.col('trx_flag'),
                    trx_util=pl.col('trx_flag') / pl.col('exposure'))
                )

        if len(self.percent_of) > 0:
            pct_cols = {}
            for x in self.percent_of:
                xw = x + "_w_trx"
                pct_cols[f"pct_of_{x}_all"] = pl.col('trx_amt') / pl.col(x)
                pct_cols[f"pct_of_{xw}"] = pl.col('trx_amt') / pl.col(xw)

            data = data.with_columns(**pct_cols)

        # confidence interval formulas
        if conf_int:

            p = [(1 - conf_level) / 2, 1 - (1 - conf_level) / 2]

            data = data.with_columns(
                trx_util_lower=_qbinom(p[0], prob='trx_util'),
                trx_util_upper=_qbinom(p[1], prob='trx_util'),
            )

            if len(self.percent_of) > 0:
                # standard deviations
                # For binomial N
                # Var(S) = n * p * (Var(X) + E(X)**2 * (1 - p))
                data = (data.with_columns(
                    sd_trx=((pl.col('trx_amt_sq') / pl.col('trx_flag')) -
                            pl.col('avg_trx') ** 2) ** 0.5
                ).with_columns(
                    sd_all=(pl.col('trx_flag') *
                              (pl.col('sd_trx') ** 2 + pl.col('avg_trx') ** 2 *
                               (1 - pl.col('trx_util')))) ** 0.5
                ))

            ci_cols = {}

            for x in self.percent_of:

                xw = x + "_w_trx"

                # confidence intervals with transactions
                ci_cols[f'pct_of_{xw}_lower'] = \
                    _qnorm(p[0], 'trx_amt', pl.col('sd_trx') *
                           pl.col('trx_flag') ** 0.5) / pl.col(xw)
                ci_cols[f'pct_of_{xw}_upper'] = \
                    _qnorm(p[1], 'trx_amt', pl.col('sd_trx') *
                           pl.col('trx_flag') ** 0.5) / pl.col(xw)
                # confidence intervals across all records
                ci_cols[f'pct_of_{x}_all_lower'] = \
                    _qnorm(p[0], 'trx_amt', 'sd_all') / pl.col(x)
                ci_cols[f'pct_of_{x}_all_upper'] = \
                    _qnorm(p[1], 'trx_amt', 'sd_all') / pl.col(x)

            data = data.with_columns(**ci_cols)
            if conf_int and (len(self.percent_of) > 0):
                data = data.drop('sd_all', 'sd_trx')

        return data.collect()

    @classmethod
    def from_DataFrame(cls,
                       data: pl.DataFrame | pd.DataFrame,
                       conf_int: bool = False,
                       conf_level: float = 0.95,
                       col_trx_amt: str = 'trx_amt',
                       col_trx_n: str = 'trx_n',
                       col_trx_flag: str = 'trx_flag',
                       col_exposure: str = "exposure",
                       col_percent_of: str = None,
                       col_percent_of_w_trx: str = None,
                       col_trx_amt_sq: str = "trx_amt_sq",
                       start_date: date | str = date(1900, 1, 1),
                       end_date: date | str = None):
        """
        Convert a data frame containing aggregate transaction experience study
        results to the `TrxStats` class.

        `from_DataFrame()` is most useful for working with aggregate summaries 
        of experience that were not created by actxps where individual policy
        information is not available. After converting the data to the 
        `TrxStats` class, `summary()` can be used to summarize data by any 
        grouping variables, and `plot()` and `table()` are available for 
        reporting.

        Parameters
        ----------
        data : pl.DataFrame | pd.DataFrame
            A DataFrame containing aggregate transaction study results. See the 
            Notes section for required columns that must be present.
        conf_int : bool, default=False
            If `True`, future calls to `summary()` will include confidence 
            intervals around the observed utilization rates and any
            `percent_of` output columns.
        conf_level : float, default=0.95
            Confidence level used for the Limited Fluctuation credibility method
            and confidence intervals.
        col_trx_amt : str, default='trx_amt'
            Name of the column in `data` containing transaction amounts.
        col_trx_n : str, default='trx_n'
            Name of the column in `data` containing transaction counts.
        col_trx_flag : str, default='trx_flag'
            Name of the column in `data` containing the number of exposure records
            with transactions.
        col_exposure : str, default='exposure'
            Name of the column in `data` containing exposures.
        col_percent_of : str, default=None
            Name of the column in `data` containing a numeric variable to use 
            in "percent of" calculations.
        col_percent_of_w_trx : str, default=None
            Name of the column in `data` containing a numeric variable to use 
            in "percent of" calculations with transactions.
        col_trx_amt_sq : str, default='trx_amt_sq'
            Only required when `col_percent_of` is passed and `conf_int` is 
            `True`. Name of the column in `data` containing squared transaction 
            amounts.
        start_date : date | str, default=date(1900, 1, 1)
            Transaction study start date
        end_date : date | str, optional
            Transaction study end date

        Returns
        -------
        TrxStats
            A `TrxStats` object


        Notes
        ----------
        At a minimum, the following columns are required:

        - Transaction amounts (`trx_amt`)
        - Transaction counts (`trx_n`)
        - The number of exposure records with transactions (`trx_flag`). 
        This number is not necessarily equal to transaction counts. If multiple 
        transactions are allowed per exposure period, `trx_flag` will be less 
        than `trx_n`.
        - Exposures (`exposure`)

        If transaction amounts should be expressed as a percentage of another
        variable (i.e. to calculate utilization rates or actual-to-expected 
        ratios), additional columns are required:

        - A denominator "percent of" column. For example, the sum of account 
        values.
        - A denominator "percent of" column for exposure records with
        transactions. For example, the sum of account values across all records
        with non-zero transaction amounts.

        If confidence intervals are desired and "percent of" columns are passed,
        an additional column for the sum of squared transaction amounts 
        (`trx_amt_sq`) is also required.

        The names in parentheses above are expected column names. If the data
        frame passed to `from_DataFrame()` uses different column names, these 
        can be specified using the `col_*` arguments.

        `start_date`, and `end_date` are optional arguments that are
        only used for printing the resulting `TrxStats` object.

        Unlike `ExposedDF.trx_stats()`, `from_DataFrame()` only permits a 
        single transaction type and a single `percent_of` column.

        Examples
        ----------
        ```{python}
        # convert pre-aggregated experience into a TrxStats object
        import actxps as xp

        agg_sim_dat = xp.load_agg_sim_dat()
        dat = xp.TrxStats.from_DataFrame(
            agg_sim_dat,
            col_exposure="n",
            col_trx_amt="wd",
            col_trx_n="wd_n",
            col_trx_flag="wd_flag",
            col_percent_of="av",
            col_percent_of_w_trx="av_w_wd",
            col_trx_amt_sq="wd_sq",
            start_date=2005, end_date=2019,
            conf_int=True)
        dat

        # summary by policy year
        dat.summary('pol_yr')
        ```

        See Also
        ----------
        `ExposedDF.trx_stats()` for information on how `TrxStats` objects are 
        typically created from individual exposure records.
        """

        # convert data to polars dataframe if necessary
        data = _check_convert_df(data)

        # column name alignment
        rename_dict = {col_trx_amt: 'trx_amt',
                       col_trx_n: 'trx_n',
                       col_trx_flag: 'trx_flag',
                       col_exposure: "exposure"}

        req_names = {"exposure", "trx_amt", "trx_n", "trx_flag"}

        if conf_int and col_percent_of is not None:
            req_names.update(["trx_amt_sq"])
            rename_dict.update({col_trx_amt_sq: "trx_amt_sq"})

        if col_percent_of is not None:
            req_names.update([col_percent_of, col_percent_of + "_w_trx"])

        if col_percent_of_w_trx is not None:
            assert col_percent_of is not None, \
                "`col_percent_of_w_trx` was supplied without passing " + \
                "anything to `col_percent_of`"
            rename_dict.update(
                {col_percent_of_w_trx: col_percent_of + "_w_trx"})

        data = (data.rename(rename_dict).
                with_columns(trx_type=pl.lit(col_trx_amt)))

        # check required columns
        _verify_col_names(data.columns, req_names)

        if col_percent_of is None:
            col_percent_of = []
        else:
            col_percent_of = [col_percent_of]

        return TrxStats(data,
                        trx_types=[col_trx_amt],
                        percent_of=col_percent_of,
                        groups=[],
                        start_date=start_date,
                        end_date=end_date,
                        xp_params={'conf_int': conf_int,
                                   'conf_level': conf_level},
                        agg=False)

    @ __init__.register(pl.DataFrame)
    def _special_init(self, data: pl.DataFrame, **kwargs):
        """
        Special constructor for the TrxStats class. This constructor is used
        by the `from_DataFrame()` class method to create new TrxStats objects 
        from pre-aggregated data frames.
        """
        self._finalize(data, **kwargs)

    def summary(self, *by):
        """
        Re-summarize transaction experience data

        Re-summarize the data while retaining any grouping variables passed to
        the `*by` argument.

        Parameters
        ----------
        *by :
            Column names in `data` that will be used as grouping variables in
            the re-summarized object. Passing nothing is acceptable and will
            produce a 1-row experience summary.

        Returns
        ----------
        TrxStats
            A new `TrxStats` object with rows for all the unique groups in `*by`

        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status="Surrender")
        expo.add_transactions(withdrawals)

        trx_res = (expo.group_by('inc_guar', 'pol_yr').
                   trx_stats(percent_of='premium'))
        trx_res.summary('inc_guar')
        ```            
        """
        return TrxStats(by, self)

    @ __init__.register(tuple)
    def _special_init(self, by: tuple, old_self):
        """
        Special constructor for the TrxStats class. This constructor is used
        by the `summary()` class method to create new summarized instances.
        """

        by = list(by)

        if len(by) > 0:
            assert all(pl.Series(by).is_in(old_self.data.columns)), \
                "All grouping variables passed to `*by` must be in the " + \
                "`.data` property."

        self._finalize(old_self.data.lazy(),
                       old_self.trx_types, old_self.percent_of,
                       by, old_self.start_date, old_self.end_date,
                       old_self.xp_params)

    def __repr__(self):
        repr = "Transaction study results\n\n"

        if len(self.groups) > 0:
            repr += f"Groups: {', '.join([str(i) for i in self.groups])}\n"

        repr += f"Study range: {self.start_date} to {self.end_date}\n"

        repr += f"Transaction types: {', '.join([str(i) for i in self.trx_types])}\n"

        if len(self.percent_of) > 0:
            repr += f"Transactions as % of: {', '.join([str(i) for i in self.percent_of])}\n"

        if self.data is not None:
            repr = repr + f'\n{self.data}'

        return repr

    def plot(self,
             x: str = None,
             y: str = "trx_util",
             color: str = None,
             facets: list | str = None,
             mapping: aes = None,
             scales: str = "fixed",
             geoms: str = "lines",
             y_labels: callable = lambda l: [f"{v * 100:.1f}%" for v in l],
             y_log10: bool = False,
             conf_int_bars: bool = False):
        """
        Plot transaction study results

        Parameters
        ----------
        x : str, default=None
            A column name in `data` to use as the `x` variable. If `None`,
            `x` will default to the first grouping variable. If there are no
            grouping variables, `x` will be set to "All".
        y : str, default='trx_util'
            A column name in `data` to use as the `y` variable.
        color : str, default=None
            A column name in `data` to use as the `color` and `fill` variables.
            If `None`, `y` will default to the second grouping variable. If 
            there are less than two grouping variables, the plot will not use 
            a color aesthetic.
        facets : list | str, default=None
            Faceting variables in `data` passed to `plotnine.facet_wrap()`. If 
            `None`, grouping variables 3+ will be used (assuming there are more
            than two grouping variables).
        mapping : aes, default=None
            Aesthetic mapping added to `plotnine.ggplot()`. NOTE: If `mapping` 
            is supplied, the `x`, `y`, and `color` arguments will be ignored.
        scales : str, default='fixed'
            The `scales` argument passed to `plotnine.facet_wrap()`.
        geoms : {'lines', 'bars', 'points}
            Type of geometry. If "lines" is passed, the plot will display lines
            and points. If "bars", the plot will display bars. If "points",
            the plot will display points only.
        y_labels : callable, default=lambda l: [f"{v * 100:.1f}%" for v in l]
            Label function passed to `plotnine.scale_y_continuous()`.
        y_log10 : bool, default=False
            If `True`, the y-axes are plotted on a log-10 scale.
        conf_int_bars: bool, default=False
            If `True`, confidence interval error bars are included in the plot.
            This option is only available for utilization rates and any 
            `pct_of` columns.


        Notes
        ----------
        If no aesthetic map is supplied, the plot will use the first grouping
        variable in the `groups` property on the x axis and `trx_util` on
        the y axis. In addition, the second grouping variable in `groups` will 
        be used for color and fill.

        If no faceting variables are supplied, the plot will use grouping
        variables 3 and up as facets. These variables are passed into
        `plotnine.facet_wrap()`.


        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status="Surrender")
        expo.add_transactions(withdrawals)

        trx_res = (expo.group_by('pol_yr').
                   trx_stats(percent_of='premium'))

        trx_res.plot()
        ```        
        """

        if facets is None:
            facets = self.groups[2:]
        facets = ['trx_type'] + np.atleast_1d(facets).tolist()

        return _plot_experience(self, x, y, color, mapping, scales,
                                geoms, y_labels, facets, y_log10,
                                conf_int_bars)

    def plot_utilization_rates(self, **kwargs):
        """
        Plot transaction frequency and severity. 

        Frequency is represented by utilization rates (`trx_util`). Severity is
        represented by transaction amounts as a percentage of one or more other
        columns in the data (`{*}_w_trx`). All severity series begin with the 
        prefix "pct_of_" and end with the suffix "_w_trx". The suffix refers to 
        the fact that the denominator only includes records with non-zero 
        transactions. Severity series are based on column names passed to the 
        `percent_of` argument in `trx_stats()`. If no "percentage of" columns
        exist, this function will only plot utilization rates.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to `plot()`

        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        account_vals = xp.load_account_vals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status="Surrender")
        expo.add_transactions(withdrawals)
        expo.data = expo.data.join(account_vals, how='left',
                                   on=["pol_num", "pol_date_yr"])        

        trx_res = (expo.group_by('pol_yr').
                   trx_stats(percent_of='av_anniv', combine_trx=True))

        trx_res.plot_utilization_rates()
        ```                    
        """
        piv_cols = ["trx_util"]
        piv_cols.extend(np.intersect1d(
            [f"pct_of_{x}_w_trx" for x in self.percent_of],
            self.data.columns))

        if _len2(self.groups) == 1:
            kwargs.update({'color': '_no_color'})

        if 'facets' not in kwargs:
            facets = self.groups[2:]
        else:
            facets = []
        facets = ['trx_type'] + np.atleast_1d(facets).tolist() + ['Series']

        piv_data = _pivot_plot_special(self, piv_cols)

        return _plot_experience(self, y="Rate",
                                facets=facets,
                                alt_data=piv_data,
                                scales="free_y",
                                group_insert=999,
                                **kwargs)

    def table(self,
              fontsize: int = 100,
              decimals: int = 1,
              colorful: bool = True,
              color_util: str = "GnBu",
              color_pct_of: str = "RdBu",
              show_conf_int: bool = False,
              decimals_amt: int = 0,
              suffix_amt: bool = False,
              **rename_cols: str):
        """
        Tabular transaction study summary

        Convert transaction study results to a presentation-friendly format.

        Parameters
        ----------
        fontsize : int, default=100
            Font size percentage multiplier

        decimals : int, default=1
            Number of decimals to display for percentages

        colorful : bool, default=True
            If `True`, color will be added to the the observed utilization rate
            and "percentage of" columns.

        color_util : str, default='GnBu'
            ColorBrewer palette used for the observed utilization rate.

        color_pct_of : str, default='RdBu'
            ColorBrewer palette used for "percentage of" columns.

        show_conf_int: bool, default=False 
            If `True` any confidence intervals will be displayed.

        decimals_amt: bool, default=0
            Number of decimals to display for amount columns (transaction 
            counts, total transactions, and average transactions)

        suffix_amt: bool, default=False
            This argument has the same meaning as the `compact` argument in 
            great_tables.gt.GT.fmt_number() for amount columns. If `False`,
            no scaling or suffixing are applied to amount columns. If `True`, 
            all amount columns are automatically scaled and suffixed by "K" 
            (thousands), "M" (millions), "B" (billions), or "T" (trillions).

        rename_cols : str, default=None
            Key-value pairs where keys are column names and values are labels
            that will appear on the output table. This parameter is useful for
            renaming grouping variables that will appear under their original
            variable names if left unchanged.

        Notes
        ----------
        Further customizations can be added using great_tables.gt.GT methods. 
        See the `great_tables` package documentation for more information.

        Returns
        ----------
        great_tables.gt.GT
            A formatted HTML table

        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status="Surrender")
        expo.add_transactions(withdrawals)

        trx_res = (expo.group_by('pol_yr').
                   trx_stats(percent_of='premium'))

        trx_res.table()
        ```
        """

        # set up properties
        data = self.data.clone()
        percent_of = self.percent_of
        trx_types = self.trx_types
        start_date = self.start_date
        end_date = self.end_date
        conf_int = self.xp_params['conf_int']

        # remove unnecessary columns
        if len(percent_of) > 0:
            data = data.drop(percent_of + [x + "_w_trx" for x in percent_of])
            if conf_int:
                data = data.drop('trx_amt_sq')

        ci_cols = col_matches(data, '_(?:upp|low)er$')
        if show_conf_int and not conf_int:
            _conf_int_warning()
        elif conf_int and not show_conf_int:
            data = data.drop(ci_cols)
            ci_cols = []
        conf_int = show_conf_int and conf_int

        # set up index and groups
        data = (data.
                drop('exposure').
                sort(['trx_type'] + self.groups))
        if len(self.groups) > 0:
            groupname_col = 'trx_type'
            rowname_col = self.groups[0]
            data = data.drop('index')
        else:
            groupname_col = None
            rowname_col = None

        # TODO - once implemented, add `sub_missing()`
        tab = (GT(data,
                  groupname_col=groupname_col,
                  rowname_col=rowname_col).
               fmt_number(['trx_n', 'trx_amt', 'trx_flag',
                           'avg_trx', 'avg_all'],
                          decimals=decimals_amt,
                          compact=suffix_amt).
               fmt_number('trx_freq', decimals=1).
               fmt_percent(col_starts_with(data, "trx_util") +
                           col_starts_with(data, "pct_of_"),
                           decimals=decimals).
               # sub_missing().
               tab_options(table_font_size=pct(fontsize),
                           # row_striping_include_table_body=True,
                           column_labels_font_weight='bold').
               tab_spanner(md("**Counts**"), ["trx_n", "trx_flag"]).
               tab_spanner(md("**Averages**"), ["avg_trx", "avg_all"]).
               cols_label(trx_n="Total",
                          trx_flag="Periods",
                          trx_amt="Amount",
                          avg_trx=md("*w/ trx*"),
                          avg_all=md("*all*"),
                          trx_freq="Frequency",
                          trx_util="Utilization",
                          **rename_cols).
               tab_header(title="Transaction Study Results",
                          subtitle=f"Transaction type{'s' if len(trx_types) > 1 else ''}: {', '.join(trx_types)}").
               tab_source_note(f"Study range: {start_date} to {end_date}")
               )

        # TODO add this once great_tables supports cols_merge_range
        # merge confidence intervals into a single range column
        if conf_int:
            tab = (tab.
                   tab_spanner(md("**Utilization**"),
                               ['trx_util', 'trx_util_lower', 'trx_util_upper']).
                   cols_label(trx_util=md("*Rate*"),
                              trx_util_lower=md("*CI*"),
                              trx_util_upper=""))
            # TODO merge pct_of columns
            # for i in percent_of:
            #     pass

        for i in percent_of:
            tab = _span_percent_of(tab, i, conf_int)

        if colorful:
            if data['trx_util'].n_unique() > 1:
                tab = tab.data_color(['trx_util'], palette=color_util)

            if len(percent_of) > 0:
                pct_of_cols = [f"pct_of_{x}_w_trx" for x in percent_of] + \
                    [f"pct_of_{x}_all" for x in percent_of]
                pct_of_vals = data[pct_of_cols].to_numpy()
                pct_of_vals = pct_of_vals[~np.isnan(pct_of_vals)]
                domain_pct = pct_of_vals.min(), pct_of_vals.max()
                if domain_pct[0] != domain_pct[1]:
                    tab = tab.data_color(pct_of_cols, palette=color_pct_of,
                                         reverse=True, domain=domain_pct)

        return tab


def _span_percent_of(tab: GT, pct_of, conf_int):

    pct_names = [f"pct_of_{pct_of}{x}" for x in ["_w_trx", "_all"]]
    if conf_int:
        pct_names = pct_names + [x + y for x, y in
                                 product(pct_names, ['_lower', '_upper'])]

    rename_dict = {
        pct_names[0]: md("*w/ trx*"),
        pct_names[1]: md("*all*")
    }

    if conf_int:
        rename_dict.update({
            pct_names[2]: md("*w/ trx CI*"),
            pct_names[3]: "",
            pct_names[4]: md("*all CI*"),
            pct_names[5]: "",
        })

    tab = (tab.
           tab_spanner(md(f"**% of {pct_of}**"), pct_names).
           cols_label(**rename_dict))

    return tab
