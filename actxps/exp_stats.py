import pandas as pd
import numpy as np
from scipy.stats import norm
from warnings import warn
from functools import singledispatchmethod
from actxps.expose import ExposedDF
from actxps.tools import _plot_experience
from matplotlib.colors import Colormap
from plotnine import aes


class ExpStats():
    """
    # Experience study summary class

    Create a summary of termination experience for a given target status
    (an `ExpStats` object).

    Typically, the `ExpStats` class constructor should not be called directly.
    The preferred method for creating an `ExpStats` object is to call the
    `exp_stats()` method on an `ExposedDF` object.

    ## Parameters

    `expo`: ExposedDF
        An exposed data frame class
    `target_status`: str | list | np.ndarray, default = None
        Optional. A single string, list, or array of target status values
    `expected`: str | list | np.ndarray, default = None
        Optional. A single string, list, or array of column names in the
        `data` property of `expo` with expected values
    `wt`: str, default = None
        Optional. Name of the column in the `data` property of `expo` containing
        weights to use in the calculation of claims, exposures, and
        partial credibility.
    `credibility`: bool, default = False
        Whether the output should include partial credibility weights and
        credibility-weighted decrement rates.
    `cred_p`: float, default = 0.95
        Confidence level under the Limited Fluctuation credibility method
    `cred_r`: float, default = 0.05
        Error tolerance under the Limited Fluctuation credibility method

    ## Details

    If `expo` is grouped (see the `ExposedDF.groupby()` method),
    the returned `ExpStats` object's data will contain one row per group.

    If nothing is passed to `target_status`, the `target_status` property
    of `expo` will be used. If that property is `None`,
    all status values except the first level will be assumed. This will
    produce a warning message.

    ### Expected values

    The `expected` argument is optional. If provided, this argument must
    be a string, list, or array with values corresponding to columns in
    `expo.data` containing expected experience. More than one expected basis
    can be provided.

    ### Credibility

    If `credibility` is set to `True`, the output will contain a
    `credibility` column equal to the partial credibility estimate under
    the Limited Fluctuation credibility method (also known as Classical
    Credibility) assuming a binomial distribution of claims.

    ## Methods

    `summary()`
        Calling `summary()` will re-summarize the data while retaining any
        grouping variables passed to the `*by` argument. This will return a new
        `ExpStats` object.

    `plot()`
        Produce an experience summary plot.

    `table()`
        Produce an experience summary table.

    ## Properties

    `data`: pd.DataFrame
        A data frame containing experience study summary results that includes
        columns for any grouping variables, claims, exposures, and observed
        decrement rates (`q_obs`). If any values are passed to `expected`,
        additional columns will be added for expected decrements and
        actual-to-expected ratios. If `credibility` is set to `True`, additional
        columns are added for partial credibility and credibility-weighted
        decrement rates (assuming values are passed to `expected`).

    `target_status`, `groups`, `start_date`, `end_date`, `expected`, `wt`,
    `cred_params`
        Metadata about the experience study inferred from the `ExposedDF`
        object (`expo`) or passed directly to `ExpStats`.

    ### References

    Herzog, Thomas (1999). Introduction to Credibility Theory
    """

    @singledispatchmethod
    def __init__(self,
                 expo: ExposedDF,
                 target_status: str | list | np.ndarray = None,
                 expected: str | list | np.ndarray = None,
                 wt: str = None,
                 credibility: bool = False,
                 cred_p: float = 0.95,
                 cred_r: float = 0.05):

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

        cred_params = {'credibility': credibility,
                       'cred_p': cred_p,
                       'cred_r': cred_r}

        # set up properties and summarize data
        self._finalize(data, expo.groups, target_status,
                       expo.end_date, expo.start_date,
                       expected, wt, cred_params)

        return None

    def _finalize(self,
                  data: pd.DataFrame,
                  groups,
                  target_status,
                  end_date,
                  start_date,
                  expected,
                  wt,
                  cred_params):
        """
        Internal method for finalizing experience study summary objects
        """

        # set up properties
        self.groups = groups
        self.target_status = target_status
        self.end_date = end_date
        self.start_date = start_date
        if isinstance(expected, str | int):
            expected = [expected]
        self.expected = expected
        self.wt = wt
        self.cred_params = cred_params

        # finish exp stats
        if groups is not None:
            res = (data.groupby(groups).
                   apply(self._calc).
                   reset_index().
                   drop(columns=[f'level_{len(groups)}']))

        else:
            res = self._calc(data)

        self.data = res

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
        credibility = self.cred_params['credibility']
        cred_p = self.cred_params['cred_p']
        cred_r = self.cred_params['cred_r']

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
            y = (norm.ppf((1 + cred_p) / 2) / cred_r) ** 2

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

        # dictionary of columns that depend on summarized values
        fields2 = {
            'q_obs': lambda x: x.claims / x.exposure
        }
        fields2.update(wt_forms2)
        fields2.update(cred)

        # convert dataults to a data frame
        data = pd.DataFrame(fields, index=range(1)).assign(**fields2)

        # add A/E's and adjusted q's
        if expected is not None:
            for k in expected:
                data['ae_' + k] = data.q_obs / data[k]

        if credibility & (expected is not None):
            for k in expected:
                data['adj_' + k] = (data.credibility * data.q_obs +
                                    (1 - data.credibility) * data[k])

        # rearrange and drop columns
        if wt is not None:
            data = data.drop(columns=['ex_wt', 'ex2_wt'])

        cols = ['n_claims', 'claims', 'exposure', 'q_obs']
        if expected is not None:
            cols.extend([k for k in expected] +
                        ['ae_' + k for k in expected])

        if credibility:
            cols.extend(['credibility'])
            if expected:
                cols.extend(['adj_' + k for k in expected])

        if wt is not None:
            cols.extend(['weight', 'weight_sq', 'weight_n'])

        data = data[cols]

        return data

    def summary(self, *by):
        """
        # Re-summarize termination experience data

        Re-summarize the data while retaining any grouping variables passed to
        the `*by` argument.

        ## Parameters

        *`by`:
            Column names in `data` that will be used as grouping variables in
            the re-summarized object. Passing nothing is acceptable and will
            produce a 1-row experience summary.

        ## Returns

        A new `ExpStats` object.
        """

        by = list(by)

        if len(by) == 0:
            by = None
        else:
            assert all(pd.Series(by).isin(self.data.columns)), \
                "All grouping variables passed to `*by` must be in the`.data` property."

        self.groups = by

        return ExpStats('from_summary', self)

    @ __init__.register(str)
    def _special_init(self,
                      style: str,
                      old_self):
        """
        Special constructor for the ExpStats class. This constructor is used
        by the `summary()` class method to create new summarized instances.
        """

        assert style == "from_summary"
        self.data = None
        self._finalize(old_self.data, old_self.groups, old_self.target_status,
                       old_self.end_date, old_self.start_date,
                       old_self.expected, old_self.wt, old_self.cred_params)

    def __repr__(self):
        repr = "Experience study results\n\n"

        if self.groups is not None:
            repr += f"Groups: {', '.join([str(i) for i in self.groups])}\n"

        repr += f"Target status: {', '.join([str(i) for i in self.target_status])}\n"
        repr += f"Study range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"

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
             y_labels: callable = lambda l: [f"{v * 100:.1f}%" for v in l]):
        """
        # Plot experience study results

        ## Parameters

        `x`: str
            A column name in `data` to use as the `x` variable. If `None`,
            `x` will default to the first grouping variable. If there are no
            grouping variables, `x` will be set to "All".
        `y`: str
            A column name in `data` to use as the `y` variable. If `None`, 
            `y` will default to the observed termination rate ("q_obs").
        `color`: str
            A column name in `data` to use as the `color` and `fill` variables.
            If `None`, `y` will default to the second grouping variable. If 
            there are less than two grouping variables, the plot will not use 
            a color aesthetic.
        `facets`: list or str
            Faceting variables in `data` passed to `plotnine.facet_wrap()`. If 
            `None`, grouping variables 3+ will be used (assuming there are more
            than two grouping variables).
        `mapping`: aes
            Aesthetic mapping added to `plotnine.ggplot()`. NOTE: If `mapping` 
            is supplied, the `x`, `y`, and `color` arguments will be ignored.
        `scales`: str
            The `scales` argument passed to `plotnine.facet_wrap()`.
        `geoms`: str, must be "lines" (default) or "bars"
            Type of geometry. If "lines" is passed, the plot will display lines
            and points. If "bars", the plot will display bars.
        `y_labels`: callable 
            Label function passed to `plotnine.scale_y_continuous()`.

        ## Details 

        If no aesthetic map is supplied, the plot will use the first
        grouping variable in the `groups` property on the x axis and `q_obs` on
        the y axis. In addition, the second grouping variable in `groups` will 
        be used for color and fill.

        If no faceting variables are supplied, the plot will use grouping
        variables 3 and up as facets. These variables are passed into
        `plotnine.facet_wrap()`.
        """

        return _plot_experience(self, x, y, color, mapping, scales,
                                geoms, y_labels, facets)

    def table(self,
              fontsize: int = 100,
              decimals: int = 1,
              colorful: bool = True,
              color_q_obs: str | Colormap = "GnBu",
              color_ae_: str | Colormap = "RdBu_r",
              rename_cols: dict = None):
        """
        # Tabular experience study summary

        Convert experience study results to a presentation-friendly format.

        ## Parameters

        `fontsize`: int, default = 100
            Font size percentage multiplier

        `decimals`: int, default = 1
            Number of decimals to display for percentages

        `colorful`: bool, default = `True`
            If `TRUE`, color will be added to the the observed decrement rate
            and actual-to-expected columns.

        `color_q_obs`: str or colormap, default = 'GnBu'
            Matplotlib colormap used for the observed decrement rate.

        `color_ae_`: str or colormap, default = 'RdBu_r'
            Matplotlib colormap used for actual-to-expected rates.

        `rename_cols`: dict
            An optional list consisting of key-value pairs. This can be used to
            relabel columns on the output table. This parameter is useful 
            for renaming grouping variables that will appear under their 
            original variable names if left unchanged. 
            See `pandas.DataFrame.drop()` for more information.

        ## Details

        Further customizations can be added using Pandas Styler functions. See 
        `pandas.DataFrame.style` for more information.

        ## Returns

        A formatted HTML table of the Pandas styler class.
        """

        # set up properties
        data = self.data.copy()
        expected = self.expected
        if expected is None:
            expected = [None]
        target_status = self.target_status
        wt = self.wt
        cred = self.cred_params['credibility']
        start_date = self.start_date.strftime('%Y-%m-%d')
        end_date = self.end_date.strftime('%Y-%m-%d')

        # display column names
        q_obs = '<em>q<sup>obs</sup></em>'
        claims = 'Claims'
        exposure = 'Exposures'
        credibility = '<em>Z<sup>cred</sup></em>'

        # rename and drop unnecessary columns
        if rename_cols is None:
            rename_cols = {}

        rename_cols.update({'q_obs': q_obs,
                            'claims': claims,
                            'exposure': exposure,
                            'credibility': credibility})

        data = (data.
                drop(columns=data.columns[data.columns.str.startswith('weight')]).
                rename(columns=rename_cols)
                )

        if expected == [None]:
            l1 = ['' for x in data.columns]
            l2 = data.columns
        else:
            l1 = data.columns.str.extract(
                f"^.*({'|'.join(expected)})$").fillna('')[0]
            l2 = data.columns.str.replace(
                f"{'|'.join(expected)}$", "", regex=True)
        l2 = np.where(l2 == '', '<em>q<sup>exp</sup></em>', l2)
        l2 = np.where(l2 == 'ae_', '<em>A/E</em>', l2)
        l2 = np.where(l2 == 'adj_', '<em>q<sup>adj</sup></em>', l2)

        # set up spanners by creating a multi-index and relocating columns
        data.columns = pd.MultiIndex.from_arrays([l1, l2])

        if expected != [None]:
            data = data[[''] + expected]
        if cred:
            z = data.pop(('', credibility))
            data[('', credibility)] = z

        # identify percentage and A/E columns for formatting
        pct_cols = [(x, y) for x, y in zip(l1, l2) if y.startswith('<em>')]
        ae_cols = [(x, y) for x, y in zip(l1, l2) if y.startswith('<em>A/E')]

        if wt is not None:
            data = data.rename(columns={'n_claims': '# Claims'})
        else:
            data = data.drop(columns=('', 'n_claims'))

        # apply all styling except colors
        tab = (
            data.
            style.
            format('{:,.0f}', subset=[("", claims), ("", exposure)]).
            format('{:.' + str(decimals) + '%}', subset=pct_cols).
            set_table_styles([{'selector': 'th',
                               'props': [('font-weight', 'bold'),
                                         ('font-size', str(fontsize) + '%')]},
                              {'selector': 'tr',
                             'props': [('font-size', str(fontsize) + '%')]},
                              {'selector': 'caption',
                               'props': f'font-size: {fontsize}%;'},
                              {'selector': 'th.col_heading',
                               'props': 'text-align: center;'},
                              {'selector': 'th.col_heading.level0',
                               'props': 'font-size: 1.1em;'}]).
            set_caption('<h1>Experience Study Results</h1>' +
                        f"Target status{'es' if len(target_status) > 1 else ''}: " +
                        f"{', '.join(target_status)}<br>" +
                        f"Study range: {start_date} to "
                        f"{end_date}" +
                        (f"<br>Results weighted by {wt}" if wt is not None else "")).
            hide(axis='index')
        )

        # apply colors
        if colorful:
            tab = (
                tab.
                background_gradient(subset=[("", q_obs)], cmap=color_q_obs).
                background_gradient(subset=ae_cols, cmap=color_ae_)
            )

        return tab
