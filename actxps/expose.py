import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
from datetime import datetime
from actxps.tools import arg_match
from actxps.dates import frac_interval, add_interval
from warnings import warn
from functools import singledispatchmethod
from itertools import product


class ExposedDF():
    """
    # Exposed data frame class

    Convert a data frame of census-level records to exposure-level
    records.

    ## Parameters

    `data`: pd.DataFrame
        A data frame with census-level records
    `end_date`: datetime
        Experience study end date
    `start_date`: datetime, default = '1900-01-01'
        Experience study start date
    `target_status`: str | list | np.ndarray, default = `None`
        Target status values
    `cal_expo`: bool, default = `False`
        Set to `True` for calendar year exposures. Otherwise policy year
        exposures are assumed.
    `expo_length`: str
        Exposure period length. Must be 'year', 'quarter', 'month', or 'week'
    `col_pol_num`: str, default = 'pol_num'
        Name of the column in `data` containing the policy number
    `col_status`: str, default = 'status'
        name of the column in `data` containing the policy status
    `col_issue_date` str, default = 'issue_date'
        name of the column in `data` containing the issue date
    `col_term_date` str, default = 'term_date'
        name of the column in `data` containing the termination date
    `default_status`: str, default = `None`
        Optional default active status code

    ## Details

    Census-level data refers to a data set wherein there is one row
    per unique policy. Exposure-level data expands census-level data such that
    there is one record per policy per observation period. Observation periods
    could be any meaningful period of time such as a policy year, policy month,
    calendar year, calendar quarter, calendar month, etc.

    `target_status` is used in the calculation of exposures. The annual
    exposure method is applied, which allocates a full period of exposure for
    any statuses in `target_status`. For all other statuses, new entrants
    and exits are partially exposed based on the time elapsed in the observation
    period. This method is consistent with the Balducci Hypothesis, which assumes
    that the probability of termination is proportionate to the time elapsed
    in the observation period. If the annual exposure method isn't desired,
    `target_status` can be ignored. In this case, partial exposures are
    always applied regardless of status.

    `default_status` is used to indicate the default active status that
    should be used when exposure records are created. If `None`, then the
    first status level will be assumed to be the default active status.

    ### References

    Atkinson and McGarry (2016). Experience Study Calculations

    https://www.soa.org/49378a/globalassets/assets/files/research/experience-study-calculations.pdf

    ## Methods

    ### ExposedDF class construction helpers

    The class methods `expose_py()`, `expose_pq()`, `expose_pm()`,
    `expose_pw()`, `expose_cy()`, `expose_cq()`, `expose_cm()`, and
    `expose_cw()` are convenience functions for specific exposure calculations.
    The two characters after the underscore describe the exposure type and
    exposure period, respectively.

    For exposures types:

    - `p` refers to policy years
    - `c` refers to calendar years

    For exposure periods:

    - `y` = years
    - `q` = quarters
    - `m` = months
    - `w` = weeks

    Each constructor has the same inputs as the `__init__` method except that
    `expo_length` and `cal_expo` arguments are prepopulated.

    The class method `from_DataFrame` can be used to convert a data frame that
    already has exposure-level records into an `ExposedDF` object.

    ### `groupby()` and `ungroup()`

    Add or remove grouping variables for summary methods like `exp_stats()`
    and 'trx_stats()`.

    ### `exp_stats()`

    Summarize experience study results and return an `ExpStats` object.

    ### `add_transactions()`

    Attach a data frame of transactions to the `data` property. Once 
    transactions are added, the summary method `trx_stats()` can be used.

    ### `trx_stats()`

    Summarize transactions results and return a `TrxStats` object.

    ## Properties

    `data`: pd.DataFrame
        A data frame with exposure level records. The results include all
        existing columns in the original input data plus new columns for
        exposures and observation periods. Observation periods include counters
        for policy exposures, start dates, and end dates. Both start dates and
        end dates are inclusive bounds.

        For policy year exposures, two observation period columns are returned.
        Columns beginning with (`pol_`) are integer policy periods. Columns
        beginning with (`pol_date_`) are calendar dates representing
        anniversary dates, monthiversary dates, etc.

    `end_date`, `start_date`, `target_status`, `cal_expo`, `expo_length`:
        Values passed on class instantiation. See Parameters for definitions.

    `exposure_type`: str
        A description of the exposure type that combines the `cal_expo` and
        `expo_length` properties

    `date_cols`: tuple
        Names of the start and end date columns in `data` for each exposure
        period

    `trx_types`: list
        List of transaction types that have been attached to `data` using 
        the `add_transactions()` method.
    """

    # helper dictionary for abbreviations
    abbr_period = {
        "year": "yr",
        "quarter": "qtr",
        "month": "mth",
        "week": "wk"
    }

    @singledispatchmethod
    def __init__(self,
                 data: pd.DataFrame,
                 end_date: datetime,
                 start_date: datetime = datetime(1900, 1, 1),
                 target_status: str | list | np.ndarray = None,
                 cal_expo: bool = False,
                 expo_length: str = 'year',
                 col_pol_num: str = "pol_num",
                 col_status: str = "status",
                 col_issue_date: str = "issue_date",
                 col_term_date: str = "term_date",
                 default_status: str = None):

        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)
        target_status = np.atleast_1d(target_status)

        # column rename helper function
        def rename_col(prefix: str,
                       suffix: str = ""):
            res = ExposedDF.abbr_period[expo_length]
            return prefix + "_" + res + suffix

        # set up exposure period lengths
        arg_match('expo_length', expo_length,
                  ["year", "quarter", "month", "week"])

        def per_frac(start, end): return frac_interval(start, end, expo_length)
        def per_add(dates, x): return add_interval(dates, x, expo_length)

        if cal_expo:
            match expo_length:
                case 'year':
                    floor_date = pd.offsets.YearBegin()

                case 'quarter':
                    floor_date = pd.offsets.QuarterBegin(startingMonth=1)

                case 'month':
                    floor_date = pd.offsets.MonthBegin()

                case 'week':
                    floor_date = pd.offsets.Week(weekday=6)

        # column renames and name conflicts
        data = data.rename(columns={
            col_pol_num: 'pol_num',
            col_status: 'status',
            col_issue_date: 'issue_date',
            col_term_date: 'term_date'
        })

        # check for potential name conflicts
        abbrev = ExposedDF.abbr_period[expo_length]
        x = np.array([
            "exposure",
            ("cal_" if cal_expo else "pol_") + abbrev,
            'pol_date_' + abbrev if not cal_expo else None,
            ('cal_' if cal_expo else 'pol_date_') + abbrev + '_end'
        ])

        x = x[np.isin(x, data.columns)]
        data = data.drop(columns=x)

        if len(x) > 0:
            warn("`data` contains the following conflicting columns that "
                 f"will be overridden: {', '.join(x)}. If you don't want "
                 "this to happen, rename these columns before creating an "
                 "`ExposedDF` object.")

        # set up default status
        status_levels = data.status.unique()
        if default_status is None:
            default_status = pd.Categorical(
                [status_levels[0]],
                categories=status_levels)
        else:
            status_levels = np.union1d(status_levels, default_status)
            default_status = pd.Categorical(
                [default_status],
                categories=status_levels
            )

        # pre-exposure updates
        # drop policies issued after the study end and
        #   policies that terminated before the study start
        data = data.loc[(data.issue_date < end_date) &
                        (data.term_date.isna() | (data.term_date > start_date))]
        data.term_date = pd.to_datetime(
            np.where(data.term_date > end_date,
                     pd.NaT, data.term_date))
        data.status = np.where(data.term_date.isna(),
                               default_status, data.status)
        data['last_date'] = data.term_date.fillna(end_date)

        if cal_expo:

            start_dates = pd.Series(np.repeat(start_date, len(data)),
                                    index=data.index)
            data['first_date'] = np.maximum(data.issue_date, start_dates)
            data['cal_b'] = data.first_date + Day() - floor_date
            data['tot_per'] = per_frac((data.cal_b - Day()), data.last_date)

        else:
            data['tot_per'] = per_frac((data.issue_date - Day()),
                                       data.last_date)

        data['rep_n'] = np.ceil(data.tot_per)

        # apply exposures
        ndx = data.index
        data = data.loc[np.repeat(ndx, data.rep_n)].reset_index(drop=True)
        data['time'] = data.groupby('pol_num').cumcount() + 1
        data['last_per'] = data.time == data.rep_n
        data.status = np.where(data.last_per, data.status, default_status)
        data.term_date = pd.to_datetime(
            np.where(data.last_per, data.term_date, pd.NaT))

        if cal_expo:
            data['first_per'] = data.time == 1
            # necessary to convert to a series to avoid an error when Day() \
            # is subtracted
            data['cal_e'] = pd.Series(per_add(data.cal_b, data.time)) - Day(1)
            data['cal_b'] = per_add(data.cal_b, data.time - 1)
            data['cal_days'] = (data.cal_e - data.cal_b).dt.days + 1

            def cal_frac(x):
                """
                Faster function per_frac for computing the distance
                between two calendar dates. Only works for partial periods
                less than 1 full period.
                """
                numer = (x - data.cal_b).dt.days + 1
                return numer / data.cal_days

            # partial exposure calculations
            expo_cond = [
                data.status.isin in target_status,
                data.first_per & data.last_per,
                data.first_per,
                data.last_per
            ]

            expo_choice = [
                1,
                cal_frac(data.last_date) - cal_frac(data.first_date - Day(1)),
                1 - cal_frac(data.first_date - Day(1)),
                cal_frac(data.last_date)
            ]

            data['exposure'] = np.select(expo_cond, expo_choice, 1)

            data = (data.
                    drop(columns={'rep_n', 'first_date', 'last_date', 'cal_days',
                                  'first_per', 'last_per', 'time', 'tot_per'}).
                    rename(columns={
                        'cal_b': rename_col('cal'),
                        'cal_e': rename_col('cal', '_end')
                    })
                    )

        else:
            data['cal_b'] = per_add(data.issue_date, data.time - 1)
            # necessary to convert to a series to avoid an error when Day() \
            # is subtracted
            data['cal_e'] = pd.Series(per_add(data.issue_date, data.time)) - \
                Day(1)

            # partial exposure calculations
            data['exposure'] = np.where(
                data.last_per & ~data.status.isin(target_status),
                data.tot_per % 1, 1)
            # exposure = 0 is possible if exactly 1 period has elapsed.
            # replace these with 1's
            data['exposure'] = np.where(data.exposure == 0, 1, data.exposure)

            data = (data.
                    drop(columns={'last_per', 'last_date', 'tot_per', 'rep_n'}).
                    loc[(data.cal_b >= start_date) & (data.cal_b <= end_date)].
                    rename(columns={
                        'time': rename_col('pol'),
                        'cal_b': rename_col('pol_date'),
                        'cal_e': rename_col('pol_date', '_end')
                    })
                    )

        # convert status to categorical
        data.status = data.status.astype('category')
        data.status = data.status.cat.set_categories(status_levels)

        # set up other properties
        self._finalize(data, end_date, start_date, target_status,
                       cal_expo, expo_length)

        return None

    def _finalize(self,
                  data, end_date, start_date, target_status,
                  cal_expo, expo_length, trx_types=None):
        """
        This internal function finalizes class construction for `ExposedDF`
        objects.
        """
        self.data = data
        self.groups = None
        self.end_date = end_date
        self.start_date = start_date
        self.target_status = target_status
        self.cal_expo = cal_expo
        self.expo_length = expo_length
        self.exposure_type = ('calendar' if (cal_expo) else 'policy') + \
            '_' + expo_length
        self.date_cols = ExposedDF._make_date_col_names(cal_expo, expo_length)
        if trx_types is None:
            self.trx_types = []
        else:
            self.trx_types = trx_types

    @classmethod
    def expose_py(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='year', **kwargs)

    @classmethod
    def expose_pq(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='quarter', **kwargs)

    @classmethod
    def expose_pm(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='month', **kwargs)

    @classmethod
    def expose_pw(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='week', **kwargs)

    @classmethod
    def expose_cy(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='year', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cq(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='quarter', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cm(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='month', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cw(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        return cls(data, end_date, expo_length='week', cal_expo=True,
                   **kwargs)

    @classmethod
    def from_DataFrame(cls,
                       data: pd.DataFrame,
                       end_date: datetime,
                       start_date: datetime = datetime(1900, 1, 1),
                       target_status: str = None,
                       cal_expo: bool = False,
                       expo_length: str = 'year',
                       trx_types: list | str = None,
                       col_pol_num: str = "pol_num",
                       col_status: str = "status",
                       col_exposure: str = "exposure",
                       col_pol_per: str = None,
                       cols_dates: str = None,
                       col_trx_n_: str = "trx_n_",
                       col_trx_amt_: str = "trx_amt_"):
        """
        # Coerce a data frame to an `ExposedDF` object

        ## Parameters

        `data`: pd.DataFrame
            A data frame with exposure-level records
        `end_date`: datetime
            Experience study end date
        `start_date`: datetime, default = '1900-01-01'
            Experience study start date
        `target_status`: str | list | np.ndarray, default = `None`
            Target status values
        `cal_expo`: bool, default = `False`
            Set to `True` for calendar year exposures. Otherwise policy year
            exposures are assumed.
        `expo_length`: str, default = 'year'
            Exposure period length. Must be 'year', 'quarter', 'month', or 
            'week'
        `trx_types`: list or str
            Optional list containing unique transaction types that have been 
            attached to `data`. For each value in `trx_types`, `from_DataFrame` 
            requires that columns exist in `data` named `trx_n_{*}` and 
            `trx_amt_{*}` containing transaction counts and amounts,
            respectively. The prefixes "trx_n_" and "trx_amt_" can be overridden
            using the `col_trx_n_` and `col_trx_amt_` arguments.
        `col_pol_num`: str, default = 'pol_num'
            Name of the column in `data` containing the policy number
        `col_status`: str, default = 'status'
            name of the column in `data` containing the policy status
        `col_exposure`: str, default = 'exposure'
            Name of the column in `data` containing exposures.
        `col_pol_per`: str, default = None
            Name of the column in `data` containing policy exposure periods.
            Only necessary if `cal_expo` is `False`. The assumed default is
            either "pol_yr", "pol_qtr", "pol_mth", or "pol_wk" depending on
            the value of `expo_length`.
        `col_dates`: str, default = None
            Names of the columns in `data` containing exposure start and end 
            dates. Both date ranges are assumed to be exclusive. The assumed
            default is of the form *A*_*B*. *A* is "cal" if `cal_expo` is `True`
            or "pol" otherwise. *B* is either "yr", "qtr", "mth",  or "wk"
            depending on the value of `expo_length`.
        `col_trx_n_`: str, default = "trx_n_"
            Prefix to use for columns containing transaction counts.
        `col_trx_amt_`: str, default = "trx_amt_"
            Prefix to use for columns containing transaction amounts.

        ## Details

        The input data frame must have columns for policy numbers, statuses, 
        exposures, policy periods (for policy exposures only), and exposure 
        start / end dates. Optionally, if `data` has transaction counts and 
        amounts by type, these can be specified without calling 
        `add_transactions()`.

        ## Returns:

        An `ExposedDF` object.
        """

        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)
        target_status = np.atleast_1d(target_status)

        arg_match('expo_length', expo_length,
                  ["year", "quarter", "month", "week"])

        assert isinstance(data, pd.DataFrame), \
            '`data` must be a Pandas DataFrame'

        # column name alignment
        data = data.rename(columns={
            col_pol_num: 'pol_num',
            col_status: 'status',
            col_exposure: 'exposure'
        })

        # column name alignment - policy exposure periods
        if not cal_expo:
            exp_col_pol_per = 'pol_' + ExposedDF.abbr_period[expo_length]
            if col_pol_per is not None:
                data = data.rename(columns={
                    col_pol_per: exp_col_pol_per
                })
        else:
            exp_col_pol_per = None

        # column name alignment - period start and end dates
        exp_cols_dates = ExposedDF._make_date_col_names(cal_expo, expo_length)

        if cols_dates is not None:
            assert len(cols_dates) == 2, \
                "`cols_dates` must be a length 2 character vector"

            data = data.rename(columns={
                cols_dates[0]: exp_cols_dates[0],
                cols_dates[1]: exp_cols_dates[1]
            })

        # minimum required columns - pol_num, status, exposure,
        #  policy period (policy expo only)
        unmatched = {"pol_num", "status", "exposure", exp_col_pol_per}

        # check transaction types
        if trx_types != None:

            def trx_renamer(x):
                return x.replace(col_trx_n_, 'trx_n_').\
                    replace(col_trx_amt_, 'trx_amt_')

            data.columns = [trx_renamer(x) for x in data.columns]

            trx_types = np.unique(trx_types).tolist()
            exp_cols_trx = [x + y for x, y in product(["trx_n_", "trx_amt_"],
                                                      trx_types)]
            unmatched.update(exp_cols_trx)

        # check required columns
        unmatched.update(exp_cols_dates)
        unmatched = unmatched.difference(data.columns)

        assert len(unmatched) == 0, \
            ("The following columns are missing from `data`: "
                f"{', '.join(unmatched)}.\nHint: create these columns or use "
                "the `col_*` arguments to specify existing columns that "
                "should be mapped to these elements.")

        return cls('already_exposed',
                   data, end_date, start_date, target_status, cal_expo,
                   expo_length, trx_types)

    @__init__.register(str)
    def _special_init(self,
                      style,
                      data: pd.DataFrame,
                      end_date: datetime,
                      start_date: datetime = datetime(1900, 1, 1),
                      target_status: str = None,
                      cal_expo: bool = False,
                      expo_length: str = 'year',
                      trx_types: list = None):
        """
        Special constructor for the ExposedDF class. This constructor is used
        by the `from_DataFrame()` class method to create new classes from
        DataFrames that already contain exposure records.
        """

        assert style == "already_exposed", \
            "`style` must be 'already_exposed'"

        self._finalize(data, end_date, start_date, target_status,
                       cal_expo, expo_length, trx_types)

    @staticmethod
    def _make_date_col_names(cal_expo: bool, expo_length: str):
        abbrev = ExposedDF.abbr_period[expo_length]
        x = ("cal_" if cal_expo else "pol_date_") + abbrev
        return x, x + "_end"

    def __repr__(self) -> str:
        repr = ("Exposure data\n\n" +
                f"Exposure type: {self.exposure_type}\n" +
                f"Target status: {', '.join([str(i) for i in self.target_status])}\n" +
                f"Study range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")

        if self.trx_types != []:
            repr += f"Transaction types: {', '.join(self.trx_types)}\n"

        repr += (f"\nA DataFrame: {self.data.shape[0]:,} x {self.data.shape[1]:,}"
                 f'\n{self.data.head(10)}')

        return repr

    def groupby(self, *by):
        """
        Set grouping variables for summary methods like `exp_stats()` and
        `trx_stats()`.

        ## Parameters

        *`by`:
            Column names in `data` that will be used as grouping variables

        ## Details

        This function will not directly apply the `DataFrame.groupby()` method
        to the `data` property. Instead, it will set the `groups` property of
        the `ExposedDF` object. The `groups` property is subsequently used to
        group data within summary methods like `exp_stats()` and `trx_stats()`.

        ## Returns

        self

        """

        by = list(by)

        if len(by) == 0:
            by = None
        else:
            assert all(pd.Series(by).isin(self.data.columns)), \
                "All grouping variables passed to `*by` must be in the `data` property."

        self.groups = by
        return self

    def ungroup(self):
        """
        Remove all grouping variables for summary methods like `exp_stats()`
        and `trx_stats()`.

        ## Returns

        self

        """
        self.groups = None
        return self

    def exp_stats(self,
                  target_status: str | list | np.ndarray = None,
                  expected: str | list | np.ndarray = None,
                  wt: str = None,
                  credibility: bool = False,
                  cred_p: float = 0.95,
                  cred_r: float = 0.05):
        """
        # Summarize experience study records

        Create a summary of termination experience for a given target status
        (an `ExpStats` object).

        ## Parameters

        `target_status`: str | list | np.ndarray, default = None
            Optional. A single string, list, or array of target status values
        `expected`: str | list | np.ndarray, default = None
            Optional. A single string, list, or array of column names in the
            `data` property with expected values
        `wt`: str, default = None
            Optional. Name of the column in the `data` property containing
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

        If the `ExposedDF` object is grouped (see the `groupby()` method), the
        returned `ExpStats` object's data will contain one row per group.

        If nothing is passed to `target_status`, the `target_status` property
        of the `ExposedDF` object will be used. If that property is `None`,
        all status values except the first level will be assumed. This will
        produce a warning message.

        ### Expected values

        The `expected` argument is optional. If provided, this argument must
        be a string, list, or array with values corresponding to columns in
        the `data` property containing expected experience. More than one
        expected basis can be provided.

        ### Credibility

        If `credibility` is set to `True`, the output will contain a
        `credibility` column equal to the partial credibility estimate under
        the Limited Fluctuation credibility method (also known as Classical
        Credibility) assuming a binomial distribution of claims.

        ## Returns

        An `ExpStats` object with a `data` property that includes columns for
        any grouping variables, claims, exposures, and observed decrement rates
        (`q_obs`). If any values are passed to `expected`, additional columns
        will be added for expected decrements and actual-to-expected ratios. If
        `credibility` is set to `True`, additional columns are added for partial
        credibility and credibility-weighted decrement rates (assuming values
        are passed to `expected`).

        ### References

        Herzog, Thomas (1999). Introduction to Credibility Theory
        """
        from actxps.exp_stats import ExpStats
        return ExpStats(self, target_status, expected, wt,
                        credibility, cred_p, cred_r)

    def add_transactions(self,
                         trx_data: pd.DataFrame,
                         col_pol_num: str = "pol_num",
                         col_trx_date: str = "trx_date",
                         col_trx_type: str = "trx_type",
                         col_trx_amt: str = "trx_amt"):
        """
        Add transactions to an experience study

        ## Parameters

        `trx_data`: pd.DataFrame
            A data frame containing transactions details. This data frame must
            have columns for policy numbers, transaction dates, transaction
            types, and transaction amounts.
        `col_pol_num`: str, default = 'pol_num'
            Name of the column in `trx_data` containing the policy number
        `col_trx_date`: str, default = 'trx_date'
            Name of the column in `trx_data` containing the transaction date
        `col_trx_type`:str, default = 'trx_type'
            Name of the column in `trx_data` containing the transaction type
        `col_trx_amt`: str, default = 'trx_amt'
            Name of the column in `trx_data` containing the transaction amount

        ## Details

        This function attaches transactions to an `ExposedDF` object.
        Transactions are grouped and summarized such that the number of rows in
        the data does not change. Two columns are added to the output
        for each transaction type. These columns have names of the pattern
        `trx_n_{*}` (transaction counts) and `trx_amt_{*}`
        (transaction_amounts).

        Transactions are associated with the data object by matching
        transactions dates with exposure dates ranges found in the `ExposedDF`.

        ## Examples

        ```
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status = "Surrender")
        expo.add_transactions(withdrawals)
        ```

        ## Returns

        self

        Two new columns are added to the `data` property containing transaction
        counts and amounts for each transaction type found in `trx_data`. The
        `trx_types` property will be updated to include the new transaction
        types found in `trx_data.`
        """

        assert isinstance(trx_data, pd.DataFrame), \
            "`data` must be a DataFrame"
        date_cols = list(self.date_cols)

        # select a minimum subset of columns
        date_lookup = self.data.copy()[['pol_num'] + date_cols]

        # column renames
        trx_data = trx_data.rename(columns={
            col_pol_num: 'pol_num',
            col_trx_date: 'trx_date',
            col_trx_type: 'trx_type',
            col_trx_amt: 'trx_amt'
        })

        # check for conflicting transaction types
        new_trx_types = pd.unique(trx_data.trx_type)
        existing_trx_types = self.trx_types
        conflict_trx_types = np.intersect1d(new_trx_types,
                                            existing_trx_types)
        if len(conflict_trx_types) > 0:
            raise ValueError("`trx_data` contains transaction types that " +
                             "have already been attached to `data`: " +
                             ', '.join(conflict_trx_types) +
                             ". \nUpdate `trx_data` with unique transaction " +
                             "types.")

        # add dates to transaction data
        trx_data = (trx_data.
                    merge(date_lookup, how='inner', on='pol_num').
                    query(f"(trx_date >= {date_cols[0]}) & " +
                          f"(trx_date <= {date_cols[1]})"))

        # pivot / summarize to match the grain of exposure data
        trx_data['trx_n'] = 1

        trx_data = (trx_data.
                    pivot_table(values=['trx_n', 'trx_amt'],
                                index=['pol_num', date_cols[0]],
                                columns='trx_type',
                                aggfunc='sum',
                                observed=True,
                                fill_value=0).
                    reset_index())

        # flatten column index
        cols = trx_data.columns.to_flat_index()
        cols = ['_'.join(x) if x[1] != '' else x[0] for x in cols]
        trx_data.columns = cols

        # add new transaction types
        self.trx_types = self.trx_types + list(new_trx_types)

        # update exposed_df structure to document transaction types
        self.data = (self.data.
                     merge(trx_data,
                           on=['pol_num', date_cols[0]],
                           how='left'))

        # replace missing values
        trx_cols = [x for x in self.data.columns if x.startswith('trx_')]
        self.data.loc[:, trx_cols] = \
            self.data.loc[:, trx_cols].apply(lambda x: x.fillna(0))

        return self

    def trx_stats(self,
                  trx_types: list | str = None,
                  percent_of: list | str = None,
                  combine_trx: bool = False,
                  col_exposure: str = 'exposure',
                  full_exposures_only: bool = True):
        """
        # Summarize transactions and utilization rates

        Create a summary of transaction counts, amounts, and utilization rates
        (a `TrxStats` object).

        ## Parameters

        `trx_types`: list or str, default = None
            A list of transaction types to include in the output. If `None` is
            provided, all available transaction types in the `trx_types` 
            property will be used.
        `percent_of`: list or str, default = None
            A optional list containing column names in the `data` property to
            use as denominators in the calculation of utilization rates or
            actual-to-expected ratios.
        `combine_trx`: bool, default = False
            If `False` (default), the results will contain output rows for each 
            transaction type. If `True`, the results will contains aggregated
            results across all transaction types.
        `col_exposure`: str, default = 'exposure'
            Name of the column in the `data` property containing exposures
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
        for each transaction type. To ensure `data` is in the appropriate 
        format, use the class method `ExposedDF.from_DataFrame()` to convert 
        an existing data frame with transactions or use `add_transactions()` 
        to attach transactions to an existing `ExposedDF` object.

        ### "Percentage of" calculations

        The `percent_of` argument is optional. If provided, this argument must
        be list with values corresponding to columns in the `data` property
        containing values to use as denominators in the calculation of 
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

        ## Examples

        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status = "Surrender")
        expo.add_transactions(withdrawals)

        expo.groupby('inc_guar').trx_stats(percent_of = "premium")
        expo.groupby('inc_guar').trx_stats(percent_of = "premium",
                                           combine_trx = True)

        ## Returns

        A `TrxStats` object with a `data` property that includes columns for
        any grouping variables and transaction types, plus the following:

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
        from actxps.trx_stats import TrxStats
        return TrxStats(self, trx_types, percent_of, combine_trx,
                        col_exposure, full_exposures_only)
