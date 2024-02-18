import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm, binom
from warnings import warn
from functools import singledispatchmethod
from actxps.expose import ExposedDF
from actxps.tools import (
    _plot_experience,
    _pivot_plot_special,
    _verify_exposed_df,
    _conf_int_warning,
    _verify_col_names,
    _date_str,
    _qnorm
)
from actxps.col_select import (
    col_contains,
    col_starts_with,
    col_ends_with
)
from actxps.expose_split import _check_split_expose_basis
from matplotlib.colors import Colormap
from plotnine import (
    aes,
    geom_hline
)
from great_tables import (
    GT,
    pct,
    md
)


class ExpStats():
    """
    Experience study summary class

    Create a summary of termination experience for a given target status
    (an `ExpStats` object).

    Typically, the `ExpStats` class constructor should not be called directly.
    The preferred method for creating an `ExpStats` object is to call the
    `exp_stats()` method on an `ExposedDF` object.

    Parameters
    ----------

    expo : ExposedDF
        An exposed data frame class
    target_status : str | list | np.ndarray, default=None
        A single string, list, or array of target status values
    expected : str | list | np.ndarray, default=None
        Single string, list, or array of column names in the `data` property of
        `expo` with expected values
    wt : str, default=None
        Name of the column in the `data` property of `expo` containing
        weights to use in the calculation of claims, exposures, and
        partial credibility.
    conf_int: bool, default=False
        If `True`, the output will include confidence intervals around the
        observed termination rates and any actual-to-expected ratios.
    credibility : bool, default=False
        Whether the output should include partial credibility weights and
        credibility-weighted decrement rates.
    conf_level : float, default=0.95
        Confidence level under the Limited Fluctuation credibility method 
        and confidence intervals
    cred_r : float, default=0.05
        Error tolerance under the Limited Fluctuation credibility method
    col_exposure : str, default='exposure'
        Name of the column in `data` containing exposures.

    Attributes
    ----------
    data : pd.DataFrame
        A data frame containing experience study summary results that includes
        columns for any grouping variables, claims, exposures, and observed
        decrement rates (`q_obs`). If any values are passed to `expected`,
        additional columns will be added for expected decrements and
        actual-to-expected ratios. If `credibility` is set to `True`, additional
        columns are added for partial credibility and credibility-weighted
        decrement rates (assuming values are passed to `expected`).

    target_status, groups, start_date, end_date, expected, wt, xp_params
        Metadata about the experience study inferred from the `ExposedDF`
        object (`expo`) or passed directly to `ExpStats`.


    Notes
    ----------
    If `expo` is grouped (see the `ExposedDF.groupby()` method),
    the returned `ExpStats` object's data will contain one row per group.

    If nothing is passed to `target_status`, the `target_status` property
    of `expo` will be used. If that property is `None`, all status values except
    the first level will be assumed. This will produce a warning message.

    **Expected values**

    The `expected` argument is optional. If provided, this argument must
    be a string, list, or array with values corresponding to columns in
    `expo.data` containing expected experience. More than one expected basis
    can be provided.

    **Confidence intervals**

    If `conf_int` is set to `True`, the output will contain lower and upper
    confidence interval limits for the observed termination rate and any
    actual-to-expected ratios. The confidence level is dictated
    by `conf_level`. If no weighting variable is passed to `wt`, confidence
    intervals will be constructed assuming a binomial distribution of claims.
    Otherwise, confidence intervals will be calculated assuming that the
    aggregate claims distribution is normal with a mean equal to observed claims
    and a variance equal to:

    `Var(S) = E(N) * Var(X) + E(X)**2 * Var(N)`,

    Where `S` is the aggregate claim random variable, `X` is the weighting
    variable assumed to follow a normal distribution, and `N` is a binomial
    random variable for the number of claims.

    If `credibility` is `True` and expected values are passed to `expected`,
    the output will also contain confidence intervals for any
    credibility-weighted termination rates.

    **Credibility**

    If `credibility` is set to `True`, the output will contain a
    `credibility` column equal to the partial credibility estimate under
    the Limited Fluctuation credibility method (also known as Classical
    Credibility) assuming a binomial distribution of claims.

    **Alternative class constructor**

    `ExpStats.from_DataFrame()` can be used to coerce a data frame containing 
    pre-aggregated experience into an `ExpStats` object. This is most useful
    for working with industry study data where individual exposure records are
    not available.

    See Also
    ----------
    Herzog, Thomas (1999). Introduction to Credibility Theory
    """

    @singledispatchmethod
    def __init__(self,
                 expo: ExposedDF,
                 target_status: str | list | np.ndarray = None,
                 expected: str | list | np.ndarray = None,
                 wt: str = None,
                 conf_int: bool = False,
                 credibility: bool = False,
                 conf_level: float = 0.95,
                 cred_r: float = 0.05,
                 col_exposure: str = 'exposure'):

        _verify_exposed_df(expo)
        self.data = None
        # set up target statuses. First, attempt to use the statuses that
        # were passed. If none, then use the target_status
        # property from the ExposedDF. If that property is None, assume
        # all statuses but the first are the target.
        if target_status is None:
            target_status = expo.target_status
        if target_status == [None]:
            target_status = list(expo.data.status.cat.categories[1:])
            warn(f"No target status was provided. {', '.join(target_status)} "
                 "was assumed.")

        data = expo.data.assign(
            n_claims=expo.data.status.isin(target_status)
        )

        # set up weights
        if wt is not None:

            assert isinstance(wt, str), "`wt` must have type `str`"
            data = data.rename(columns={wt: 'weight'})
            data['claims'] = data.n_claims * data.weight
            data['exposure'] = data.exposure * data.weight
            data['weight_sq'] = data.weight ** 2
            data['weight_n'] = 1

        else:
            data['claims'] = data.n_claims

        _check_split_expose_basis(expo, col_exposure)
        data = data.rename(columns={col_exposure: 'exposure'})

        xp_params = {'credibility': credibility,
                     'conf_level': conf_level,
                     'cred_r': cred_r,
                     'conf_int': conf_int}

        # set up properties and summarize data
        self._finalize(data, expo.groups, target_status,
                       expo.end_date, expo.start_date,
                       expected, wt, xp_params)

        return None

    def _finalize(self,
                  data: pd.DataFrame,
                  groups,
                  target_status,
                  end_date,
                  start_date,
                  expected,
                  wt,
                  xp_params,
                  agg: bool = True):
        """
        Internal method for finalizing experience study summary objects
        """

        # set up properties
        self.groups = groups
        self.target_status = target_status
        self.end_date = _date_str(end_date)
        self.start_date = _date_str(start_date)
        if isinstance(expected, str | int):
            expected = [expected]
        self.expected = expected
        self.wt = wt
        self.xp_params = xp_params

        # finish exp stats
        if agg:
            if groups is not None:
                res = (data.groupby(groups, observed=True).
                       apply(self._calc, include_groups=False).
                       reset_index().
                       drop(columns=[f'level_{len(groups)}']))
            else:
                res = self._calc(data)

            self.data = res
        else:
            self.data = data

        return None

    def _calc(self, data: pd.DataFrame):
        """
        Support function for summarizing data for one group
        """
        # dictionary of summarized values
        fields = {'n_claims': sum(data.n_claims),
                  'claims': sum(data.claims),
                  'exposure': sum(data.exposure)}

        expected = self.expected
        wt = self.wt
        credibility = self.xp_params['credibility']
        conf_level = self.xp_params['conf_level']
        cred_r = self.xp_params['cred_r']
        conf_int = self.xp_params['conf_int']

        if expected is not None:
            ex_mean = {k: np.average(data[k], weights=data.exposure)
                       for k in expected}
            fields.update(ex_mean)

        # additional columns for weighted studies
        if wt is not None:
            wt_forms = {
                'weight': sum(data.weight),
                'weight_sq': sum(data.weight_sq),
                'weight_n': sum(data.weight_n)}
            fields.update(wt_forms)

            wt_forms2 = {
                'ex_wt': lambda x: x.weight / x.weight_n,
                'ex2_wt': lambda x: x.weight_sq / x.weight_n}
        else:
            wt_forms2 = {}

        # credibility formulas - varying by weights
        if credibility:
            y = (_qnorm((1 + conf_level) / 2) / cred_r) ** 2

            if wt is None:
                cred = {
                    'credibility': lambda x:
                        np.minimum(1,
                                   (x.n_claims / (y * (1 - x.q_obs)))
                                   ** 0.5)
                }
            else:
                cred = {
                    'credibility': lambda x:
                        np.minimum(1,
                                   (x.n_claims / (
                                       y * ((x.ex2_wt - x.ex_wt ** 2) *
                                            x.weight_n / (x.weight_n - 1) /
                                            x.ex_wt ** 2 + 1 - x.q_obs)))
                                   ** 0.5)
                }

        else:
            cred = {}

        # confidence interval formulas
        if conf_int:

            p = [(1 - conf_level) / 2, 1 - (1 - conf_level) / 2]

            if wt is None:
                ci = {
                    'q_obs_lower': lambda x:
                        binom.ppf(p[0], np.round(x.exposure),
                                  x.q_obs) / x.exposure,
                    'q_obs_upper': lambda x:
                        binom.ppf(p[1], np.round(x.exposure),
                                  x.q_obs) / x.exposure
                }
            else:
                ci = {
                    # For binomial N
                    # Var(S) = n * p * (Var(X) + E(X)**2 * (1 - p))
                    'sd_agg': lambda x:
                        (x.n_claims * ((x.ex2_wt - x.ex_wt ** 2) +
                                       x.ex_wt ** 2 * (1 - x.q_obs))) ** 0.5,
                    'q_obs_lower': lambda x:
                        _qnorm(p[0], x.claims, x.sd_agg) / x.exposure,
                    'q_obs_upper': lambda x:
                        _qnorm(p[1], x.claims, x.sd_agg) / x.exposure
                }

        else:
            ci = {}

        # dictionary of columns that depend on summarized values
        fields2 = {
            'q_obs': lambda x: x.claims / x.exposure
        }
        fields2.update(wt_forms2)
        fields2.update(cred)
        fields2.update(ci)

        # convert dataults to a data frame
        data = pd.DataFrame(fields, index=range(1)).assign(**fields2)

        # add A/E's and adjusted q's
        if expected is not None:
            for k in expected:
                data['ae_' + k] = data.q_obs / data[k]

        if credibility and (expected is not None):
            for k in expected:
                data['adj_' + k] = (data.credibility * data.q_obs +
                                    (1 - data.credibility) * data[k])

        if conf_int and (expected is not None):
            for k in expected:
                data[f'ae_{k}_lower'] = data.q_obs_lower / data[k]
                data[f'ae_{k}_upper'] = data.q_obs_upper / data[k]

                if credibility:
                    data[f'adj_{k}_lower'] = \
                        (data.credibility * data.q_obs_lower +
                         (1 - data.credibility) * data[k])
                    data[f'adj_{k}_upper'] = \
                        (data.credibility * data.q_obs_upper +
                         (1 - data.credibility) * data[k])

        # rearrange and drop columns
        if wt is not None:
            data = data.drop(columns=['ex_wt', 'ex2_wt'])

        cols = ['n_claims', 'claims', 'exposure', 'q_obs']
        if conf_int:
            cols.extend(['q_obs_lower', 'q_obs_upper'])

        if expected is not None:
            cols.extend([k for k in expected] +
                        ['ae_' + k for k in expected])
            if conf_int:
                cols.extend([f'ae_{k}_lower' for k in expected] +
                            [f'ae_{k}_upper' for k in expected])

        if credibility:
            cols.extend(['credibility'])
            if expected is not None:
                cols.extend(['adj_' + k for k in expected])
                if conf_int:
                    cols.extend([f'adj_{k}_lower' for k in expected] +
                                [f'adj_{k}_upper' for k in expected])

        if wt is not None:
            cols.extend(['weight', 'weight_sq', 'weight_n'])

        data = data[cols]

        return data

    @classmethod
    def from_DataFrame(cls,
                       data: pd.DataFrame,
                       target_status: str | list | np.ndarray = None,
                       expected: str | list | np.ndarray = None,
                       wt: str = None,
                       conf_int: bool = False,
                       credibility: bool = False,
                       conf_level: float = 0.95,
                       cred_r: float = 0.05,
                       col_claims: str = 'claims',
                       col_exposure: str = 'exposure',
                       col_n_claims: str = 'n_claims',
                       col_weight_sq: str = 'weight_sq',
                       col_weight_n: str = 'weight_n',
                       start_date: datetime | int | str = datetime(1900, 1, 1),
                       end_date: datetime | int | str = None):
        """
        Convert a data frame containing aggregate termination experience study
        results to the `ExpStats` class.

        `from_DataFrame()` is most useful for working with aggregate summaries 
        of experience that were not created by actxps where individual policy
        information is not available. After converting the data to the 
        `ExpStats` class, `summary()` can be used to summarize data by any 
        grouping variables, and `plot()` and `table()` are available for 
        reporting.

        Parameters
        ----------

        data : pd.DataFrame
            A DataFrame containing aggregate experience study results. See the 
            Notes section for required columns that must be present.
        target_status : str | list | np.ndarray, default=None
            Target status values
        expected : str | list | np.ndarray, default=None
            Column names in x with expected values.
        wt : str, default=None
            Name of the column in `data` containing weights to use in the
            calculation of claims, exposures, partial credibility, and 
            confidence intervals.
        conf_int : bool, default=False
            If `True`, future calls to `summary()` will include confidence 
            intervals around the observed termination rates and any 
            actual-to-expected ratios.
        credibility : bool, default=False
            If `True`, future calls to `summary()` will include partial 
            credibility weights and credibility-weighted termination rates.
        conf_level : float, default=0.95
            Confidence level used for the Limited Fluctuation credibility method
            and confidence intervals.
        cred_r : float, default=0.05
            Error tolerance under the Limited Fluctuation credibility method.            
        col_claims : str, default='claims'
            Name of the column in `data` containing claims.
        col_exposure : str, default='exposure'
            Name of the column in `data` containing exposures.
        col_n_claims : str, default='n_claims'
            Only used used when `wt` is passed. Name of the column in `data` 
            containing the number of claims.
        col_weight_sq : default='weight_sq
            Only used used when `wt` is passed. Name of the column in `data` 
            containing the sum of squared weights.
        col_weight_n : str, default='weight_n'
            Only used used when `wt` is passed. Name of the column in `data` 
            containing exposure record counts.
        start_date : datetime | int | str, default='1900-01-01'
            Experience study start date
        end_date : datetime | int | str: default=None
            Experience study end date

        Returns
        -------
        ExpStats
            An `ExpStats` object

        Notes
        ----------        
        If nothing is passed to `wt`, the data frame `data` must include columns
        containing:

        - Exposures (`exposure`)
        - Claim counts (`claims`)

        If `wt` is passed, the data must include columns containing:

        - Weighted exposures (`exposure`)
        - Weighted claims (`claims`)
        - Claim counts (`n_claims`)
        - The raw sum of weights **NOT** multiplied by exposures
        - Exposure record counts (`.weight_n`)
        - The raw sum of squared weights (`.weight_sq`)

        The names in parentheses above are expected column names. If the data
        frame passed to `from_DataFrame()` uses different column names, these 
        can be specified using the `col_*` arguments.

        When a column name is passed to `wt`, the columns `.weight`, 
        `.weight_n`, and `.weight_sq` are used to calculate credibility and
        confidence intervals. If credibility and confidence intervals aren't
        required, then it is not necessary to pass anything to `wt`. The 
        resulting `ExpStats` class and any downstream summaries will still be
        weighted as long as the exposures and claims are pre-weighted.

        `target_status`, `start_date`, and `end_date` are optional arguments 
        that are only used for printing the resulting `ExpStats` object.


        Examples
        ----------
        ``` {python}
        import actxps as xp

        # convert pre-aggregated experience into an ExpStats object
        agg_sim_dat = xp.load_agg_sim_dat()
        dat = xp.ExpStats.from_DataFrame(
            agg_sim_dat,
            col_exposure="exposure_n",
            col_claims="claims_n",
            target_status="Surrender",
            start_date=2005,
            end_date=2019,
            conf_int=True)

        # summary by policy year
        dat.summary('pol_yr')

        # repeat the prior exercise on a weighted basis
        dat_wt = xp.ExpStats.from_DataFrame(
            agg_sim_dat, wt="av",
            col_exposure="exposure_amt",
            col_claims="claims_amt",
            col_n_claims="claims_n",
            col_weight_sq="av_sq",
            col_weight_n="n",
            target_status="Surrender",
            start_date=2005, end_date=2019,
            conf_int=True)
        dat_wt

        # summary by policy year
        dat_wt.summary('pol_yr')
        ```

        See Also
        ----------
        `ExposedDF.exp_stats()` for information on how `ExpStats` objects are
        typically created from individual exposure records.
        """

        target_status = np.atleast_1d(target_status)

        assert isinstance(data, pd.DataFrame), \
            '`data` must be a Pandas DataFrame'

        # column name alignment
        rename_dict = {col_claims: 'claims',
                       col_exposure: 'exposure'}

        req_names = {"exposure", "claims"}
        if wt is not None:
            req_names.update(["n_claims", "weight", "weight_sq", "weight_n"])
            rename_dict.update({col_n_claims: 'n_claims',
                                wt: 'weight',
                                col_weight_sq: 'weight_sq',
                                col_weight_n: 'weight_n'})

        data = data.rename(columns=rename_dict)

        # # check required columns
        _verify_col_names(data.columns, req_names)

        if wt is None:
            data['n_claims'] = data['claims']

        return ExpStats(data, groups=None,
                        target_status=target_status,
                        end_date=end_date, start_date=start_date,
                        expected=expected, wt=wt,
                        xp_params={'conf_int': conf_int,
                                   'credibility': credibility,
                                   'conf_level': conf_level,
                                   'cred_r': cred_r},
                        agg=False)

    @ __init__.register(pd.DataFrame)
    def _special_init(self, data: pd.DataFrame, **kwargs):
        """
        Special constructor for the ExpStats class. This constructor is used
        by the `from_DataFrame()` class method to create new ExpStats objects 
        from pre-aggregated data frames.
        """
        self._finalize(data, **kwargs)

    def summary(self, *by):
        """
        Re-summarize termination experience data

        Re-summarize the data while retaining any grouping variables passed to
        the `*by` argument.

        Parameters
        ----------
        *by : tuple, optional
            Quoted column names in `data` that will be used as grouping 
            variables in the re-summarized object. Passing nothing is acceptable
            and will produce a 1-row experience summary.

        Returns
        ----------
        ExpStats
            A new `ExpStats` object with rows for all the unique groups in `*by`

        Examples
        ----------
        ```{python}
        import actxps as xp

        exp_res = (xp.ExposedDF(xp.load_census_dat(),
                                "2019-12-31", 
                                target_status="Surrender").
                   groupby('pol_yr', 'inc_guar').
                   exp_stats())

        exp_res.summary('inc_guar')
        ```
        """
        return ExpStats(by, self)

    @ __init__.register(tuple)
    def _special_init(self, by: tuple, old_self):
        """
        Special constructor for the ExpStats class. This constructor is used
        by the `summary()` class method to create new summarized instances and 
        by the `from_DataFrame()` class method to create ExpStats objects from
        pre-aggregated data.
        """

        by = list(by)

        if len(by) == 0:
            by = None
        else:
            assert all(pd.Series(by).isin(old_self.data.columns)), \
                "All grouping variables passed to `*by` must be in the " + \
                "`data` property."

        self._finalize(old_self.data, by,
                       old_self.target_status, old_self.end_date,
                       old_self.start_date, old_self.expected,
                       old_self.wt, old_self.xp_params)

    def __repr__(self):
        repr = "Experience study results\n\n"

        if self.groups is not None:
            repr += f"Groups: {', '.join([str(i) for i in self.groups])}\n"

        repr += f"Target status: {', '.join([str(i) for i in self.target_status])}\n"
        repr += f"Study range: {self.start_date} to {self.end_date}\n"

        if self.expected is not None:
            repr += f"Expected values: {', '.join([str(i) for i in self.expected])}\n"

        if self.wt is not None:
            repr += f"Weighted by: {self.wt}\n"

        if self.data is not None:
            repr = (repr +
                    f"\n\nA DataFrame: {self.data.shape[0]:,} x {self.data.shape[1]:,}" +
                    f'\n{self.data.head(10)}')

        return repr

    def plot(self,
             x: str = None,
             y: str = "q_obs",
             color: str = None,
             facets: list | str = None,
             mapping: aes = None,
             scales: str = "fixed",
             geoms: str = "lines",
             y_labels: callable = lambda l: [f"{v * 100:.1f}%" for v in l],
             y_log10: bool = False,
             conf_int_bars: bool = False):
        """
        Plot experience study results

        Parameters
        ----------
        x : str, default=None
            A column name in `data` to use as the `x` variable. If `None`,
            `x` will default to the first grouping variable. If there are no
            grouping variables, `x` will be set to "All".
        y : str, default='q_obs'
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
        geoms : {'lines', 'bars', 'points'}
            Type of geometry. If "lines" is passed, the plot will display lines
            and points. If "bars", the plot will display bars. If "points",
            the plot will display points only.
        y_labels : callable, default=lambda l: [f"{v * 100:.1f}%" for v in l]
            Label function passed to `plotnine.scale_y_continuous()`.
        y_log10 : bool, default=False
            If `True`, the y-axes are plotted on a log-10 scale.
        conf_int_bars: bool, default=False
            If `True`, confidence interval error bars are included in the plot.
            This option is only available for termination rates and 
            actual-to-expected ratios.

        Notes
        ----------
        If no aesthetic map is supplied, the plot will use the first
        grouping variable in the `groups` property on the x axis and `q_obs` on
        the y axis. In addition, the second grouping variable in `groups` will 
        be used for color and fill.

        If no faceting variables are supplied, the plot will use grouping
        variables 3 and up as facets. These variables are passed into
        `plotnine.facet_wrap()`.

        Examples
        ----------
        ```{python}
        import actxps as xp

        exp_res = (xp.ExposedDF(xp.load_census_dat(),
                                "2019-12-31", 
                                target_status="Surrender").
                   groupby('pol_yr').
                   exp_stats())

        exp_res.plot()
        ```        
        """

        return _plot_experience(self, x, y, color, mapping, scales,
                                geoms, y_labels, facets, y_log10,
                                conf_int_bars)

    def plot_termination_rates(self,
                               include_cred_adj: bool = False,
                               **kwargs):
        """
        Plot observed termination rates and any expected termination rates 
        found in the `expected` property.

        Parameters
        ----------
        include_cred_adj : bool, default=False
            If `True`, credibility-weighted termination rates will be plotted 
            as well.
        **kwargs
            Additional arguments passed to `plot()`

        Examples
        ----------
        ```{python}
        import actxps as xp
        import numpy as np

        expo = xp.ExposedDF(xp.load_census_dat(),
                            "2019-12-31", 
                            target_status="Surrender")

        expected_table = np.concatenate((np.linspace(0.005, 0.03, 10), 
                                         np.array([0.2, 0.15]), 
                                         np.repeat(0.05, 3)))
        expo.data['expected_1'] = expected_table[expo.data.pol_yr - 1]
        expo.data['expected_2'] = np.where(expo.data.inc_guar, 0.015, 0.03)

        exp_res = (expo.
                   groupby('pol_yr').
                   exp_stats(expected=['expected_1', 'expected_2']))

        exp_res.plot_termination_rates()
        ```                
        """
        if include_cred_adj:
            self._cred_adj_warning()

        piv_cols = ["q_obs"]
        if self.expected is not None:
            piv_cols.extend(self.expected)
            if include_cred_adj:
                piv_cols.extend([f"adj_{x}" for x in self.expected])

        piv_data = _pivot_plot_special(self, piv_cols)

        return _plot_experience(self, y="Rate", alt_data=piv_data, **kwargs)

    def plot_actual_to_expected(self,
                                add_hline: bool = True,
                                **kwargs):
        """
        Plot actual-to-expected termination rates for any expected termination 
        rates found in the `expected` property.

        Parameters
        ----------
        add_hline : bool, default=True
            If `True`, a blue dashed horizontal line will be drawn at 100%.
        **kwargs
            Additional arguments passed to `plot()`

        Examples
        ----------
        ```{python}
        import actxps as xp
        import numpy as np

        expo = xp.ExposedDF(xp.load_census_dat(),
                            "2019-12-31", 
                            target_status="Surrender")

        expected_table = np.concatenate((np.linspace(0.005, 0.03, 10), 
                                         np.array([0.2, 0.15]), 
                                         np.repeat(0.05, 3)))
        expo.data['expected_1'] = expected_table[expo.data.pol_yr - 1]
        expo.data['expected_2'] = np.where(expo.data.inc_guar, 0.015, 0.03)

        exp_res = (expo.
                   groupby('pol_yr').
                   exp_stats(expected=['expected_1', 'expected_2']))

        exp_res.plot_actual_to_expected()
        ```                            
        """
        assert self.expected is not None, \
            "This object does not have any actual-to-expected results " + \
            "available. Hint: to add expected values, use the " + \
            "`expected` argument in `exp_stats()`"

        piv_cols = np.intersect1d([f"ae_{x}" for x in self.expected],
                                  self.data.columns)

        piv_data = _pivot_plot_special(self, piv_cols, values_to="A/E ratio")

        p = _plot_experience(self, y="A/E ratio", alt_data=piv_data, **kwargs)
        if add_hline:
            return p + geom_hline(yintercept=1,
                                  linetype='dashed', color="#112599")
        else:
            return p

    def table(self,
              fontsize: int = 100,
              decimals: int = 1,
              colorful: bool = True,
              color_q_obs: str = "GnBu",
              color_ae_: str = "RdBu",
              show_conf_int: bool = False,
              show_cred_adj: bool = False,
              decimals_amt: int = 0,
              suffix_amt: bool = False,
              **rename_cols: str):
        """
        Tabular experience study summary

        Convert experience study results to a presentation-friendly format.

        Parameters
        ----------
        fontsize : int, default=100
            Font size percentage multiplier

        decimals : int, default=1
            Number of decimals to display for percentages

        colorful : bool, default=True
            If `True`, color will be added to the the observed decrement rate
            and actual-to-expected columns.

        color_q_obs : str or colormap, default='GnBu'
            ColorBrewer palette used for the observed decrement rate.

        color_ae_ : str or colormap, default='RdBu'
            ColorBrewer palette used for actual-to-expected rates.

        show_conf_int: bool, default=False 
            If `True` any confidence intervals will be displayed.

        show_cred_adj: bool, default=False
            If `True` any credibility-weighted termination rates will be 
            displayed.

        decimals_amt: bool, default=0
            Number of decimals to display for amount columns (number of claims, 
            claim amounts, and exposures.

        suffix_amt: bool: default=False
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
        import numpy as np

        expo = xp.ExposedDF(xp.load_census_dat(),
                            "2019-12-31", 
                            target_status="Surrender")

        expected_table = np.concatenate((np.linspace(0.005, 0.03, 10), 
                                         np.array([0.2, 0.15]), 
                                         np.repeat(0.05, 3)))
        expo.data['expected_1'] = \
            expected_table[expo.data.pol_yr - 1]
        expo.data['expected_2'] = \
            np.where(expo.data.inc_guar, 0.015, 0.03)

        exp_res = (expo.
                   groupby('pol_yr').
                   exp_stats(expected=['expected_1', 'expected_2'],
                             credibility=True))

        exp_res.table()
        ```                            
        """

        # set up properties
        data = self.data.copy()
        expected = self.expected
        if expected is None:
            has_expected = False
            ex_cols = []
        else:
            has_expected = True
            ex_cols = expected
        target_status = self.target_status
        wt = self.wt
        cred = self.xp_params['credibility']
        start_date = self.start_date
        end_date = self.end_date
        conf_int = self.xp_params['conf_int']

        ci_cols = col_contains(data, '_(?:upp|low)er$')
        if show_conf_int and not conf_int:
            _conf_int_warning()
        elif conf_int and not show_conf_int:
            data.drop(columns=ci_cols, inplace=True)
            ci_cols = []

        conf_int = show_conf_int and conf_int

        if has_expected:
            adj_cols = col_contains(
                data,
                f'adj_(?:{"|".join([str(x) for x in expected])})')
        else:
            adj_cols = []

        if show_cred_adj and (not cred or not has_expected):
            self._cred_adj_warning()
        elif cred and not show_cred_adj and has_expected:
            data.drop(columns=adj_cols, inplace=True)
            adj_cols = []

        show_cred_adj = show_cred_adj and cred

        wgt_cols = col_starts_with(data, 'weight')

        tab = (GT(data.drop(columns=wgt_cols)).
               fmt_number(['n_claims', 'claims', 'exposure'],
                          decimals=decimals_amt, compact=suffix_amt).
               fmt_percent(['q_obs'] +
                           col_ends_with(data, '_lower') +
                           col_ends_with(data, '_upper') +
                           col_starts_with(data, 'ae_') +
                           adj_cols +
                           col_contains(data, '^credibility$') +
                           ex_cols,
                           decimals=decimals).
               tab_options(table_font_size=pct(fontsize),
                           # row_striping_include_table_body=True,
                           column_labels_font_weight='bold').
               cols_label(q_obs=md("*q<sup>obs</sup>*"),
                          claims="Claims",
                          exposure="Exposures",
                          **rename_cols).
               tab_header(title='Experience Study Results',
                          subtitle=f"Target status{'es' if len(target_status) > 1 else ''}: {', '.join(target_status)}").
               tab_source_note(f"Study range: {start_date} to {end_date}")
               )

        if wt is not None:
            tab = (tab.
                   tab_source_note(md(f"Results weighted by `{wt}`")).
                   cols_label(n_claims=f"# Claims"))
        else:
            tab = tab.cols_hide('n_claims')

        # TODO add this once great_tables supports cols_merge_range
        # merge confidence intervals into a single range column
        if conf_int:
            tab = (tab.
                   cols_label(q_obs_lower=md("*q<sup>obs</sup> CI*"),
                              q_obs_upper=""))
        #            cols_merge_range('q_obs_lower', 'q_obs_upper').
        #            cols_label(q_obs_lower=md("*q<sup>obs</sup> CI*")))
        #     if has_expected:
        #         for i in expected:
        #             tab = (tab.
        #                    cols_merge_range(f"ae_{i}_lower",
        #                                 f"ae_{i}_upper"))
        #             if show_cred_adj:
        #                 tab = (tab.
        #                        cols_merge_range(f"adj_{i}_lower",
        #                                         f"adj_{i}_upper"))

        if has_expected:
            for i in expected:
                tab = _span_expected(tab, i, conf_int, show_cred_adj)

        if cred:
            tab = tab.cols_label(credibility=md("*Z<sup>cred</sup>*"))

        if colorful:
            if data['q_obs'].nunique() > 1:
                tab = tab.data_color(['q_obs'], palette=color_q_obs)

            if has_expected:
                ae_cols = ["ae_" + x for x in expected]
                ae_vals = data[ae_cols].values
                ae_vals = ae_vals[~np.isnan(ae_vals)]
                domain_ae = ae_vals.min(), ae_vals.max()
                if domain_ae[0] != domain_ae[1]:
                    tab = tab.data_color(ae_cols, palette=color_ae_, 
                                         reverse=True, domain=domain_ae)

        return tab

    def _cred_adj_warning(self):
        """
        This internal function provides a common warning that is used by 
        multiple functions.
        """
        if not self.xp_params['credibility'] or self.expected is None:
            warn("This object has no credibility-weighted termination " +
                 "rates. Hint: pass `credibility=True` and one or more " +
                 "column names to `expected` when calling `exp_stats()` " +
                 "to calculate credibility-weighted termination rates.")


def _span_expected(tab: GT, ex, conf_int, show_cred_adj):
    """
    Internal helper function for adding spanners to table outputs
    """
    ae = "ae_" + ex
    cols = [ex, ae]
    if conf_int:
        # TODO update after cols_merge is implemented & use append instead of extend
        ae_ci = [ae + "_lower", ae + "_upper"]
        cols.extend(ae_ci)
    if show_cred_adj:
        adj = "adj_" + ex
        cols.append(adj)
        if conf_int:
            # TODO update after cols_merge is implemented & use append instead of extend
            adj_ci = ["adj_" + ex + "_lower", "adj_" + ex + "_upper"]
            cols.extend(adj_ci)

    tab = (tab.
           tab_spanner(md(f"`{ex}`"), cols).
           cols_label(**{ex: md("*q<sup>exp</sup>*"),
                         ae: md("*A/E*")}))

    if show_cred_adj:
        tab = tab.cols_label(**{adj: md("*q<sup>adj</sup>*")})

    if conf_int:
        rename_dict = {ae_ci[0]: md("*A/E CI*"),
                       ae_ci[1]: ""}
        if show_cred_adj:
            rename_dict.update({adj_ci[0]: md("*q<sup>adj</sup> CI*"),
                                adj_ci[1]: ""})
        tab = tab.cols_label(**rename_dict)

    return tab
