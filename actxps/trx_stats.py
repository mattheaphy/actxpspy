import pandas as pd
import numpy as np
from functools import singledispatchmethod
from actxps.expose import ExposedDF
from actxps.tools import _plot_experience
from plotnine import aes
from matplotlib.colors import Colormap


class TrxStats():
    """
    # Experience study summary class

    Create a summary of transaction counts, amounts, and utilization rates
    (a `TrxStats` object).

    Typically, the `TrxStats` class constructor should not be called directly.
    The preferred method for creating a `TrxStats` object is to call the
    `trx_stats()` method on an `ExposedDF` object.    

    ## Parameters

    `expo`: ExposedDF
        An exposed data frame class
    `trx_types`: list or str, default = None
        A list of transaction types to include in the output. If `None` is
        provided, all available transaction types in the `trx_types` 
        property of `expo` will be used.
    `percent_of`: list or str, default = None
        A optional list containing column names in the `data` property of `expo`
        to use as denominators in the calculation of utilization rates or
        actual-to-expected ratios.
    `combine_trx`: bool, default = False
        If `False` (default), the results will contain output rows for each 
        transaction type. If `True`, the results will contains aggregated
        results across all transaction types.
    `col_exposure`: str, default = 'exposure'
        Name of the column in the `data` property of `expo` containing exposures
    `full_exposures_only`: bool, default = True
        If `True` (default), partially exposed records will be ignored 
        in the results.

    ## Details

    If the `ExposedDF` object is grouped (see the `groupby()` method), the
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

    ### "Percentage of" calculations

    The `percent_of` argument is optional. If provided, this argument must
    be list with values corresponding to columns in the `data` property of 
    `expo` containing values to use as denominators in the calculation of 
    utilization rates or actual-to-expected ratios. Example usage:

    - In a study of partial withdrawal transactions, if `percent_of` refers
    to account values, observed withdrawal rates can be determined.
    - In a study of recurring claims, if `percent_of` refers to a column
    containing a maximum benefit amount, utilization rates can be 
    determined.

    ### Default removal of partial exposures

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

    ## Methods

    `summary()`
        Calling `summary()` will re-summarize the data while retaining any
        grouping variables passed to the `*by` argument. This will return a new
        `TrxStats` object.

    `plot()`
        Produce an transaction summary plot.

    `table()`
        Produce an transaction summary table.

    ## Properties

    `data`: pd.DataFrame
        A data framethat includes columns for any grouping variables and
        transaction types, plus the following:

        - `trx_n`: the number of unique transactions.
        - `trx_amt`: total transaction amount
        - `trx_flag`: the number of observation periods with non-zero 
        transaction amounts.
        - `exposure`: total exposures
        - `avg_trx`: mean transaction amount (`trx_amt / trx_flag`)
        - `avg_all`: mean transaction amount over all records 
        (`trx_amt / exposure`)
        - `trx_freq`: transaction frequency when a transaction occurs 
        (`trx_n / trx_flag`)
        - `trx_utilization`: transaction utilization per observation period 
        (`trx_flag / exposure`)

        If `percent_of` is provided, the results will also include:

        - The sum of any columns passed to `percent_of` with non-zero
        transactions. These columns include the suffix `_w_trx`.
        - The sum of any columns passed to `percent_of`
        - `pct_of_{*}_w_trx`: total transactions as a percentage of column
        `{*}_w_trx`
        - `pct_of_{*}_all`: total transactions as a percentage of column `{*}`
    """
    @singledispatchmethod
    def __init__(self,
                 expo: ExposedDF,
                 trx_types: list | str = None,
                 percent_of: list | str = None,
                 combine_trx: bool = False,
                 col_exposure: str = 'exposure',
                 full_exposures_only: bool = True):

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

        start_date = expo.start_date
        end_date = expo.end_date
        data = expo.data.copy()
        groups = expo.groups
        if groups is None:
            groups = []

        data = data.rename(columns={col_exposure: 'exposure'})

        # remove partial exposures
        if full_exposures_only:
            data = data.loc[np.isclose(data.exposure, 1)]

        trx_cols = data.columns[data.columns.str.match('trx_(n|amt)_')]
        trx_cols = trx_cols[trx_cols.str.contains('|'.join(trx_types))]

        if combine_trx:
            trx_n_cols = trx_cols[trx_cols.str.contains("_n_")]
            trx_amt_cols = trx_cols[trx_cols.str.contains("_amt_")]
            data['trx_n_All'] = 0
            data['trx_amt_All'] = 0
            for n, amt in zip(trx_n_cols, trx_amt_cols):
                data['trx_n_All'] += data[n]
                data['trx_amt_All'] += data[amt]
            trx_cols = ["trx_n_All", "trx_amt_All"]

        if percent_of is None:
            percent_of = []
        else:
            # coerce to list in case percent_of is str
            percent_of = np.atleast_1d(percent_of).tolist()

        # subset columns
        id_vars = ['pol_num', 'exposure'] + groups + percent_of
        data = data[id_vars + list(trx_cols)]
        # pivot longer
        data = data.reset_index().melt(id_vars=id_vars + ['index'])
        # split transaction types from kinds
        data.variable = data.variable.str.replace('^trx_', '', regex=True)
        data[['kind', 'trx_type']] = \
            data.variable.str.split('_', expand=True, n=2)
        # pivot wider
        data = (data.
                pivot(index=['index'] + id_vars + ['trx_type'],
                      values='value', columns='kind').
                reset_index().
                drop(columns='index').
                rename(columns={'n': 'trx_n',
                                'amt': 'trx_amt'}))
        data.columns.name = None
        # fill in missing values
        data.trx_n = data.trx_n.fillna(0)
        data.trx_amt = data.trx_amt.fillna(0)
        data['trx_flag'] = data.trx_n.abs() > 0

        for x in percent_of:
            data[x + '_w_trx'] = data[x] * data.trx_flag

        self._finalize(data, trx_types, percent_of,
                       groups, start_date, end_date)

    def _finalize(self,
                  data: pd.DataFrame,
                  trx_types,
                  percent_of,
                  groups,
                  start_date,
                  end_date):
        """
        Internal method for finalizing transaction study summary objects
        """

        # set up properties
        self.groups = groups
        self.trx_types = trx_types
        self.start_date = start_date
        self.percent_of = percent_of
        self.end_date = end_date

        # finish trx stats
        res = (data.groupby(groups + ['trx_type']).
               apply(self._calc).
               reset_index().
               drop(columns=[f'level_{len(groups) + 1}']))

        self.data = res

        return None

    def _calc(self, data: pd.DataFrame):
        """
        Support function for summarizing data for one group
        """
        # safe divide by zero that returns infinite without warning
        def div(x, y):
            if y == 0:
                if x >= 0:
                    return np.Inf
                else:
                    return -np.Inf
            else:
                return x / y

        # dictionary of summarized values
        fields = {'trx_n': sum(data.trx_n),
                  'trx_flag': sum(data.trx_flag),
                  'trx_amt': sum(data.trx_amt),
                  'exposure': sum(data.exposure)}

        fields['avg_trx'] = div(fields['trx_amt'], fields['trx_flag'])
        fields['avg_all'] = div(fields['trx_amt'], fields['exposure'])
        fields['trx_freq'] = div(fields['trx_n'], fields['trx_flag'])
        fields['trx_util'] = div(fields['trx_flag'], fields['exposure'])

        for x in self.percent_of:
            xw = x + "_w_trx"
            fields[x] = sum(data[x])
            fields[xw] = sum(data[xw])
            fields[f"pct_of_{x}_all"] = div(fields['trx_amt'], fields[x])
            fields[f"pct_of_{xw}"] = div(fields['trx_amt'], fields[xw])

        # convert dataults to a data frame
        data = pd.DataFrame(fields, index=range(1))

        return data

    def summary(self, *by):
        """
        # Re-summarize transaction experience data

        Re-summarize the data while retaining any grouping variables passed to
        the `*by` argument.

        ## Parameters

        *`by`:
            Column names in `data` that will be used as grouping variables in
            the re-summarized object. Passing nothing is acceptable and will
            produce a 1-row experience summary.

        ## Examples

        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status = "Surrender")
        expo.add_transactions(withdrawals)

        trx_res = expo.groupby('inc_guar', 'pol_yr').\
            trx_stats(percent_of = "premium")
        trx_res.summary()
        trx_res.summary('inc_guar')

        ## Returns

        A new `TrxStats` object.
        """

        by = list(by)

        if len(by) > 0:
            assert all(pd.Series(by).isin(self.data.columns)), \
                "All grouping variables passed to `*by` must be in the`.data` property."

        self.groups = by

        return TrxStats('from_summary', self)

    @ __init__.register(str)
    def _special_init(self,
                      style: str,
                      old_self):
        """
        Special constructor for the TrxStats class. This constructor is used
        by the `summary()` class method to create new summarized instances.
        """
        assert style == "from_summary"
        self.data = None
        self._finalize(old_self.data, old_self.trx_types, old_self.percent_of,
                       old_self.groups, old_self.start_date, old_self.end_date)

    def __repr__(self):
        repr = "Transaction study results\n\n"

        if len(self.groups) > 0:
            repr += f"Groups: {', '.join([str(i) for i in self.groups])}\n"

        repr += f"Study range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"

        repr += f"Transaction types: {', '.join([str(i) for i in self.trx_types])}\n"

        if len(self.percent_of) > 0:
            repr += f"Transactions as % of: {', '.join([str(i) for i in self.percent_of])}\n"

        if self.data is not None:
            repr = (repr +
                    f"\n\nA DataFrame: {self.data.shape[0]:,} x {self.data.shape[1]:,}" +
                    f'\n{self.data.head(10)}')

        return repr

    def plot(self,
             x: str = None,
             y: str = "trx_util",
             color: str = None,
             facets: list | str = None,
             mapping: aes = None,
             scales: str = "fixed",
             geoms: str = "lines",
             y_labels: callable = lambda l: [f"{v * 100:.1f}%" for v in l]):
        """
        # Plot transaction study results

        ## Parameters

        `x`: str
            A column name in `data` to use as the `x` variable. If `None`,
            `x` will default to the first grouping variable. If there are no
            grouping variables, `x` will be set to "All".
        `y`: str
            A column name in `data` to use as the `y` variable. If `None`, 
            `y` will default to the observed utilization rate ("q_obs").
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

        If no aesthetic map is supplied, the plot will use the first grouping
        variable in the `groups` property on the x axis and `trx_util` on
        the y axis. In addition, the second grouping variable in `groups` will 
        be used for color and fill.

        If no faceting variables are supplied, the plot will use grouping
        variables 3 and up as facets. These variables are passed into
        `plotnine.facet_wrap()`.
        """

        if facets is None:
            facets = self.groups[2:]
        facets = ['trx_type'] + np.atleast_1d(facets).tolist()

        return _plot_experience(self, x, y, color, mapping, scales,
                                geoms, y_labels, facets)

    def table(self,
              fontsize: int = 100,
              decimals: int = 1,
              colorful: bool = True,
              color_util: str | Colormap = "GnBu",
              color_pct_of: str | Colormap = "RdBu_r",
              rename_cols: dict = None):
        """
        # Tabular transaction study summary

        Convert transaction study results to a presentation-friendly format.

        ## Parameters

        `fontsize`: int, default = 100
            Font size percentage multiplier

        `decimals`: int, default = 1
            Number of decimals to display for percentages

        `colorful`: bool, default = `True`
            If `True`, color will be added to the the observed utilization rate
            and "percentage of" columns.

        `color_util`: str or colormap, default = 'GnBu'
            Matplotlib colormap used for the observed utilization rate.

        `color_pct_of`: str or colormap, default = 'RdBu_r'
            Matplotlib colormap used for "percentage of" columns.

        `rename_cols`: dict
            An optional dictionaryof key-value pairs where keys are column names
            and values are labels that will appear on the output table. This
            parameter is useful for renaming grouping variables that will 
            appear under their original variable names if left unchanged.

        ## Details

        Further customizations can be added using Pandas Styler functions. See 
        `pandas.DataFrame.style` for more information.

        ## Returns

        A formatted HTML table of the Pandas styler class.
        """

        # set up properties
        data = (self.data.copy().
                rename(columns={'trx_type': 'Type'}).
                set_index(['Type'] + self.groups).
                sort_index())
        percent_of = self.percent_of
        trx_types = self.trx_types
        start_date = self.start_date.strftime('%Y-%m-%d')
        end_date = self.end_date.strftime('%Y-%m-%d')

        # display column names
        trx_n = "Total"
        trx_flag = "Periods"
        trx_amt = "Amount"
        avg_trx = "<em>w/ trx</em>"
        avg_all = "<em>all</em>"
        trx_freq = "Frequency"
        trx_util = "Utilization"

        # rename and drop unnecessary columns
        if rename_cols is None:
            rename_cols = {}

        drop_pct_of = percent_of + [x + '_w_trx' for x in percent_of]

        rename_cols.update({'trx_n': trx_n,
                            'trx_flag': trx_flag,
                            'trx_amt': trx_amt,
                            'avg_trx': avg_trx,
                            'avg_all': avg_all,
                            'trx_freq': trx_freq,
                            'trx_util': trx_util})

        data = (data.
                drop(columns=drop_pct_of + ['exposure']).
                rename(columns=rename_cols)
                )

        if len(percent_of) == 0:
            l1 = ['' for x in data.columns]
            l2 = data.columns
        else:
            l1 = data.columns.str.extract(
                f"^pct_of_({'|'.join(percent_of)})").fillna('')[0]
            l2 = data.columns.str.replace(
                f"^pct_of_({'|'.join(percent_of)})", "", regex=True)
        l2 = np.where(l2 == '_all', '<em>all</em>', l2)
        l2 = np.where(l2 == '_w_trx', '<em>w/ trx</em>', l2)

        for i, x in enumerate(l2):
            if l1[i] != '':
                l1[i] = "% of " + l1[i]
            if x in [trx_n, trx_flag] and l1[i] == "":
                l1[i] = "Counts"
            if x in [avg_trx, avg_all] and l1[i] == "":
                l1[i] = "Averages"

        # set up spanners by creating a multi-index and relocating columns
        data.columns = pd.MultiIndex.from_arrays([l1, l2])

        # TODO - remove after confirming columns are alread in the right order
        # if expected != [None]:
        #     data = data[[''] + expected]
        # if cred:
        #     z = data.pop(('', credibility))
        #     data[('', credibility)] = z

        # identify percentage and A/E columns for formatting
        pct_of_cols = [(x, y) for x, y in zip(l1, l2)
                       if x.startswith("% of ")]
        pct_cols = [('', trx_util)] + pct_of_cols

        # apply all styling except colors
        tab = (
            data.
            style.
            # TODO can we do a groupname column?
            format('{:,.0f}', subset=[("Counts", trx_n),
                                      ("Counts", trx_flag),
                                      ("", trx_amt),
                                      ("Averages", avg_trx),
                                      ("Averages", avg_all)]).
            format('{:,.1f}', subset=[("", trx_freq)]).
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
                               'props': 'font-size: 1.1em;'},
                              {'selector': 'caption',
                               'props': 'caption-side: top;'
                               }]).
            set_caption('<h1>Transaction Study Results</h1>' +
                        f"Transaction type{'s' if len(trx_types) > 1 else ''}: " +
                        f"{', '.join(trx_types)}<br>" +
                        f"Study range: {start_date} to "
                        f"{end_date}")
        )

        tab.columns.names = [None, None]

        # apply colors
        if colorful:
            tab = (
                tab.
                background_gradient(subset=[("", trx_util)], cmap=color_util).
                background_gradient(subset=pct_of_cols, cmap=color_pct_of)
            )

        return tab
