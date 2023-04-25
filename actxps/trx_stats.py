import pandas as pd
import numpy as np
from functools import singledispatchmethod
from actxps.expose import ExposedDF


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
            fields[f"pct_of_{xw}_all"] = div(fields['trx_amt'], fields[xw])

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

    def plot(self):
        pass

    def table(self):
        pass
