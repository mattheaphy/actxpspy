import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
from datetime import datetime
from actxps.tools import (
    arg_match,
    _verify_col_names
)
from actxps.dates import frac_interval, add_interval
from warnings import warn
from functools import singledispatchmethod
from itertools import product


class ExposedDF():
    """
    Exposed data frame class

    Convert a data frame of census-level records into an object with 
    exposure-level records.


    Parameters
    ----------
    data : pd.DataFrame
        A data frame with census-level records
    end_date : datetime
        Experience study end date
    start_date : datetime, default='1900-01-01'
        Experience study start date
    target_status : str | list | np.ndarray, default=`None`
        Target status values
    cal_expo : bool, default=`False`
        Set to `True` for calendar year exposures. Otherwise policy year
        exposures are assumed.
    expo_length : {'year', 'quarter', 'month', 'week'}
        Exposure period length
    col_pol_num : str, default='pol_num'
        Name of the column in `data` containing the policy number
    col_status : str, default='status'
        name of the column in `data` containing the policy status
    col_issue_date : str, default='issue_date'
        name of the column in `data` containing the issue date
    col_term_date str, default='term_date'
        name of the column in `data` containing the termination date
    default_status : str, default=`None`
        Default active status code. If `None`, the most common status is
        assumed.


    Attributes
    ----------
    data : pd.DataFrame
        A data frame with exposure level records. The results include all
        existing columns in the original input data plus new columns for
        exposures and observation periods. Observation periods include counters
        for policy exposures, start dates, and end dates. Both start dates and
        end dates are inclusive bounds.

        For policy year exposures, two observation period columns are returned.
        Columns beginning with (`pol_`) are integer policy periods. Columns
        beginning with (`pol_date_`) are calendar dates representing
        anniversary dates, monthiversary dates, etc.
    end_date, start_date, target_status, cal_expo, expo_length, default_status :
        Values passed on class instantiation. See Parameters for definitions.
    exposure_type : str
        A description of the exposure type that combines the `cal_expo` and
        `expo_length` properties
    date_cols : tuple
        Names of the start and end date columns in `data` for each exposure
        period
    trx_types: list
        List of transaction types that have been attached to `data` using 
        the `add_transactions()` method.


    Notes
    ----------
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
    most common status will be assumed.

    **Alternative class constructors**

    - `expose_py()`, `expose_pq()`, `expose_pm()`, `expose_pw()`, `expose_cy()`, 
        `expose_cq()`, `expose_cm()`, `expose_cw()`

        Convenience constructor functions for specific exposure calculations.
        The two characters after the underscore describe the exposure type and
        exposure period, respectively.
        For exposures types
        `p` refers to policy years
        `c` refers to calendar years
        For exposure periods
        `y` = years
        `q` = quarters
        `m` = months
        `w` = weeks
        Each constructor has the same inputs as the `__init__` method except 
        that `expo_length` and `cal_expo` arguments are prepopulated.

    - `from_DataFrame()`
        Convert a data frame that already has exposure-level records into an 
        `ExposedDF` object.


    References
    ----------
    Atkinson and McGarry (2016). Experience Study Calculations

    https://www.soa.org/49378a/globalassets/assets/files/research/experience-study-calculations.pdf

    Examples
    ----------
    ```{python}
    import actxps as xp

    xp.ExposedDF(xp.load_toy_census(), "2020-12-31", 
                 target_status='Surrender')
    ```
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
                [_most_common(data.status)],
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
                Faster function than per_frac for computing the distance
                between two calendar dates. Only works for partial periods
                less than 1 full period.
                """
                numer = (x - data.cal_b).dt.days + 1
                return numer / data.cal_days

            # partial exposure calculations
            expo_cond = [
                data.status.isin(target_status),
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
                       cal_expo, expo_length, default_status=default_status[0])

        return None

    def _finalize(self,
                  data, end_date, start_date, target_status,
                  cal_expo, expo_length, trx_types=None,
                  default_status=None, split=False):
        """
        This internal function finalizes class construction for `ExposedDF`
        objects.
        """
        self.data = data
        self.groups = None
        self.end_date = end_date
        self.start_date = start_date
        self.target_status = target_status
        self.default_status = default_status
        self.cal_expo = cal_expo
        self.expo_length = expo_length
        if split:
            self.exposure_type = 'split'
        elif cal_expo:
            self.exposure_type = 'calendar'
        else:
            self.exposure_type = 'policy'
        self.exposure_type = self.exposure_type + '_' + expo_length
        self.date_cols = ExposedDF._make_date_col_names(cal_expo, expo_length)
        if trx_types is None:
            self.trx_types = []
        else:
            self.trx_types = trx_types

    @classmethod
    def expose_py(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with policy year exposures
        """
        return cls(data, end_date, expo_length='year', **kwargs)

    @classmethod
    def expose_pq(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with policy quarter exposures
        """
        return cls(data, end_date, expo_length='quarter', **kwargs)

    @classmethod
    def expose_pm(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with policy month exposures
        """
        return cls(data, end_date, expo_length='month', **kwargs)

    @classmethod
    def expose_pw(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with policy week exposures
        """
        return cls(data, end_date, expo_length='week', **kwargs)

    @classmethod
    def expose_cy(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with calendar year exposures
        """
        return cls(data, end_date, expo_length='year', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cq(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with calendar quarter exposures
        """
        return cls(data, end_date, expo_length='quarter', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cm(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with calendar month exposures
        """
        return cls(data, end_date, expo_length='month', cal_expo=True,
                   **kwargs)

    @classmethod
    def expose_cw(cls, data: pd.DataFrame, end_date: datetime, **kwargs):
        """
        Create an `ExposedDF` with calendar week exposures
        """
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
                       col_trx_amt_: str = "trx_amt_",
                       default_status: str = None):
        """
        Coerce a data frame to an `ExposedDF` object

        The input data frame must have columns for policy numbers, statuses, 
        exposures, policy periods (for policy exposures only), and exposure 
        start / end dates. Optionally, if `data` has transaction counts and 
        amounts by type, these can be specified without calling 
        `add_transactions()`.


        Parameters
        ----------
        data : pd.DataFrame
            A data frame with exposure-level records
        end_date : datetime
            Experience study end date
        start_date : datetime, default='1900-01-01'
            Experience study start date
        target_status : str | list | np.ndarray, default=`None`
            Target status values
        cal_expo : bool, default=`False`
            Set to `True` for calendar year exposures. Otherwise policy year
            exposures are assumed.
        expo_length : str, default='year'
            Exposure period length. Must be 'year', 'quarter', 'month', or 
            'week'
        trx_types : list | str, optional
            List containing unique transaction types that have been 
            attached to `data`. For each value in `trx_types`, `from_DataFrame` 
            requires that columns exist in `data` named `trx_n_{*}` and 
            `trx_amt_{*}` containing transaction counts and amounts,
            respectively. The prefixes "trx_n_" and "trx_amt_" can be overridden
            using the `col_trx_n_` and `col_trx_amt_` arguments.
        col_pol_num : str, default='pol_num'
            Name of the column in `data` containing the policy number
        col_status : str, default='status'
            name of the column in `data` containing the policy status
        col_exposure : str, default='exposure'
            Name of the column in `data` containing exposures.
        col_pol_per : str, default=None
            Name of the column in `data` containing policy exposure periods.
            Only necessary if `cal_expo` is `False`. The assumed default is
            either "pol_yr", "pol_qtr", "pol_mth", or "pol_wk" depending on
            the value of `expo_length`.
        cols_dates : str, default=None
            Names of the columns in `data` containing exposure start and end 
            dates. Both date ranges are assumed to be exclusive. The assumed
            default is of the form *A*_*B*. *A* is "cal" if `cal_expo` is `True`
            or "pol" otherwise. *B* is either "yr", "qtr", "mth",  or "wk"
            depending on the value of `expo_length`.
        col_trx_n_ : str, default="trx_n_"
            Prefix to use for columns containing transaction counts.
        col_trx_amt_ : str, default="trx_amt_"
            Prefix to use for columns containing transaction amounts.
        default_status : str, default=`None`
            Default active status code


        Returns
        ----------
        ExposedDF
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
        req_names = {"pol_num", "status", "exposure", exp_col_pol_per}

        # check transaction types
        if trx_types != None:

            def trx_renamer(x):
                return x.replace(col_trx_n_, 'trx_n_').\
                    replace(col_trx_amt_, 'trx_amt_')

            data.columns = [trx_renamer(x) for x in data.columns]

            trx_types = np.unique(trx_types).tolist()
            exp_cols_trx = [x + y for x, y in product(["trx_n_", "trx_amt_"],
                                                      trx_types)]
            req_names.update(exp_cols_trx)

        # check required columns
        _verify_col_names(data.columns, req_names)

        if default_status is None:
            default_status = _most_common(data.status)

        return cls('already_exposed',
                   data, end_date, start_date, target_status, cal_expo,
                   expo_length, trx_types, default_status)

    @__init__.register(str)
    def _special_init(self,
                      style,
                      data: pd.DataFrame,
                      end_date: datetime,
                      start_date: datetime = datetime(1900, 1, 1),
                      target_status: str = None,
                      cal_expo: bool = False,
                      expo_length: str = 'year',
                      trx_types: list = None,
                      default_status: str = None):
        """
        Special constructor for the ExposedDF class. This constructor is used
        by the `from_DataFrame()` class method to create new classes from
        DataFrames that already contain exposure records.
        """

        assert style == "already_exposed", \
            "`style` must be 'already_exposed'"

        self._finalize(data, end_date, start_date, target_status,
                       cal_expo, expo_length, trx_types, default_status)

    @staticmethod
    def _make_date_col_names(cal_expo: bool, expo_length: str):
        abbrev = ExposedDF.abbr_period[expo_length]
        x = ("cal_" if cal_expo else "pol_date_") + abbrev
        return x, x + "_end"

    def expose_split(self):
        """
        Split calendar exposures by policy year

        Split calendar period exposures that cross a policy anniversary
        into a pre-anniversary record and a post-anniversary record.

        Returns
        -------
        SplitExposedDF
            A subclass of ExposedDF with calendar period exposures split by 
            policy year.

        Notes
        ----------
        The `ExposedDF` must have calendar year, quarter, month, or week 
        exposure records. Calendar year exposures are created by passing 
        `cal_expo = True` to `ExposedDF` (or alternatively, with the class 
        methods `ExposedDF.expose_cy()`, `ExposedDF.expose_cq()`, 
        `ExposedDF.expose_cm()`, and `ExposedDF.expose_cw()`).

        After splitting, the resulting data will contain both calendar exposures
        and policy year exposures. These columns will be named 'exposure_cal' 
        and 'exposure_pol', respectively. Calendar exposures will be in the 
        original units passed to `SplitExposedDF()`. Policy exposures will 
        always be expressed in years. Downstream functions like `exp_stats()` 
        and `exp_shiny()` will require clarification as to which exposure basis 
        should be used to summarize results.    

        After splitting, the column 'pol_yr' will contain policy years.

        Examples
        ----------
        ```{python}
        import actxps as xp
        toy_census = xp.load_toy_census()
        expo = xp.ExposedDF.expose_cy(toy_census, "2022-12-31")
        expo.expose_split()
        ```        

        See Also
        --------
        `SplitExposedDF()` for full information on `SplitExposedDF` class.
        """
        from actxps.expose_split import SplitExposedDF
        return SplitExposedDF(self)

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

        Parameters
        ----------
        *by:
            Column names in `data` that will be used as grouping variables

        Notes
        ----------
        This function will not directly apply the `DataFrame.groupby()` method
        to the `data` property. Instead, it will set the `groups` property of
        the `ExposedDF` object. The `groups` property is subsequently used to
        group data within summary methods like `exp_stats()` and `trx_stats()`.
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
        """
        self.groups = None
        return self

    def exp_stats(self,
                  target_status: str | list | np.ndarray = None,
                  expected: str | list | np.ndarray = None,
                  wt: str = None,
                  conf_int: bool = False,
                  credibility: bool = False,
                  conf_level: float = 0.95,
                  cred_r: float = 0.05,
                  col_exposure: str = 'exposure'):
        """
        Summarize experience study records

        Create a summary of termination experience for a given target status
        (an `ExpStats` object).

        Parameters
        ----------
        target_status : str | list | np.ndarray, default=None
            A single string, list, or array of target status values
        expected: str | list | np.ndarray, default=None
            A single string, list, or array of column names in the
            `data` property with expected values
        wt: str, default=None
            Name of the column in the `data` property containing
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
        cred_r : float, default=0.05
            Error tolerance under the Limited Fluctuation credibility method
        col_exposure : str, default='exposure'
            Name of the column in `data` containing exposures. Only necessary 
            for `SplitExposedDF` objects.

        Notes
        ----------
        If the `ExposedDF` object is grouped (see the `groupby()` method), the
        returned `ExpStats` object's data will contain one row per group.

        If nothing is passed to `target_status`, the `target_status` property
        of the `ExposedDF` object will be used. If that property is `None`,
        all status values except the first level will be assumed. This will
        produce a warning message.

        **Expected values**

        The `expected` argument is optional. If provided, this argument must
        be a string, list, or array with values corresponding to columns in
        the `data` property containing expected experience. More than one
        expected basis can be provided.

        **Confidence intervals**

        If `conf_int` is set to `True`, the output will contain lower and upper
        confidence interval limits for the observed termination rate and any
        actual-to-expected ratios. The confidence level is dictated
        by `conf_level`. If no weighting variable is passed to `wt`, confidence
        intervals will be constructed assuming a binomial distribution of 
        claims. Otherwise, confidence intervals will be calculated assuming that
        the aggregate claims distribution is normal with a mean equal to 
        observed claims and a variance equal to:

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

        Returns
        ----------
        `ExpStats`
            An `ExpStats` object with a `data` property that includes columns 
            for any grouping variables, claims, exposures, and observed 
            decrement rates (`q_obs`). If any values are passed to `expected`, 
            additional columns will be added for expected decrements and 
            actual-to-expected ratios. If `credibility` is set to `True`, 
            additional columns are added for partial credibility and
            credibility-weighted decrement rates (assuming values are passed to
            `expected`). If `conf_int` is set to `True`, additional columns are
            added for lower and upper confidence interval limits around the 
            observed termination rates and any actual-to-expected ratios. 
            Additionally, if `credibility` is `True` and expected values are 
            passed to `expected`, the output will contain confidence intervals
            around credibility-weighted termination rates. Confidence interval
            columns include the name of the original output column suffixed by
            either `_lower` or `_upper`. If a value is passed to `wt`, 
            additional columns are created containing the the sum of weights 
            (`.weight`), the sum of squared weights (`.weight_qs`), and the 
            number of records (`.weight_n`).

        References
        ----------
        Herzog, Thomas (1999). Introduction to Credibility Theory

        Examples
        ----------
        ```{python}
        import actxps as xp

        (xp.ExposedDF(xp.load_census_dat(),
                      "2019-12-31", 
                      target_status="Surrender").
                   groupby('pol_yr', 'inc_guar').
                   exp_stats(conf_int=True))
        ```        
        """
        from actxps.exp_stats import ExpStats
        return ExpStats(self, target_status, expected, wt, conf_int,
                        credibility, conf_level, cred_r, col_exposure)

    def add_transactions(self,
                         trx_data: pd.DataFrame,
                         col_pol_num: str = "pol_num",
                         col_trx_date: str = "trx_date",
                         col_trx_type: str = "trx_type",
                         col_trx_amt: str = "trx_amt"):
        """
        Add transactions to an experience study

        Parameters
        ----------
        trx_data : pd.DataFrame
            A data frame containing transactions details. This data frame must
            have columns for policy numbers, transaction dates, transaction
            types, and transaction amounts.
        col_pol_num : str, default='pol_num'
            Name of the column in `trx_data` containing the policy number
        col_trx_date : str, default='trx_date'
            Name of the column in `trx_data` containing the transaction date
        col_trx_type :str, default='trx_type'
            Name of the column in `trx_data` containing the transaction type
        col_trx_amt : str, default='trx_amt'
            Name of the column in `trx_data` containing the transaction amount

        Notes
        ----------
        This function attaches transactions to an `ExposedDF` object.
        Transactions are grouped and summarized such that the number of rows in
        the data does not change. Two columns are added to the output
        for each transaction type. These columns have names of the pattern
        `trx_n_{*}` (transaction counts) and `trx_amt_{*}`
        (transaction_amounts). The `trx_types` property is updated to include 
        the new transaction types found in `trx_data.`

        Transactions are associated with the data object by matching
        transactions dates with exposure dates ranges found in the `ExposedDF`.

        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status = "Surrender")
        expo.add_transactions(withdrawals)
        ```
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
                  full_exposures_only: bool = True,
                  conf_int: bool = False,
                  conf_level: float = 0.95,
                  col_exposure: str = 'exposure'):
        """
        Summarize transactions and utilization rates

        Create a summary of transaction counts, amounts, and utilization rates
        (a `TrxStats` object).

        Parameters
        ----------
        trx_types : list or str, default=None
            A list of transaction types to include in the output. If `None` is
            provided, all available transaction types in the `trx_types` 
            property will be used.
        percent_of : list or str, default=None
            A list containing column names in the `data` property to
            use as denominators in the calculation of utilization rates or
            actual-to-expected ratios.
        combine_trx : bool, default=False
            If `False` (default), the results will contain output rows for each 
            transaction type. If `True`, the results will contains aggregated
            results across all transaction types.
        full_exposures_only : bool, default=True
            If `True` (default), partially exposed records will be ignored 
            in the results.
        conf_int : bool, default=False 
            If `True`, the output will include confidence intervals around the
            observed utilization rate and any `percent_of` output columns.
        conf_level : float, default=0.95 
            Confidence level for confidence intervals
        col_exposure : str, default='exposure'
            Name of the column in the `data` property containing exposures. 
            Only necessary for `SplitExposedDF` objects.            


        Notes
        ----------
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

        **"Percentage of" calculations**

        The `percent_of` argument is optional. If provided, this argument must
        be list with values corresponding to columns in the `data` property
        containing values to use as denominators in the calculation of 
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
        non-zero transactions (`pct_of_{*}_w_trx`) are constructed using a 
        normal distribution
        - Intervals for transactions as a percentage of another column
        regardless of transaction utilization (`pct_of_{*}_all`) are calculated
        assuming that the aggregate distribution is normal with a mean equal to
        observed transactions and a variance equal to:

            `Var(S) = E(N) * Var(X) + E(X)**2 * Var(N)`,

        Where `S` is the aggregate transactions random variable, `X` is an 
        individual transaction amount assumed to follow a normal distribution, 
        and `N` is a binomial random variable for transaction utilization.

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


        Returns
        ----------
        `TrxStats`
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

            If `conf_int` is set to `True`, additional columns are added for 
            lower and upper confidence interval limits around the observed 
            utilization rate and any `percent_of` output columns. Confidence 
            interval columns include the name of the original output column
            suffixed by either `_lower` or `_upper`. If values are passed to 
            `percent_of`, an additional column is created containing the the sum
            of squared transaction amounts (`trx_amt_sq`).

        Examples
        ----------
        ```{python}
        import actxps as xp
        census = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        expo = xp.ExposedDF.expose_py(census, "2019-12-31",
                                      target_status="Surrender")
        expo.add_transactions(withdrawals)

        expo.groupby('inc_guar').trx_stats(percent_of="premium",
                                           combine_trx=True,
                                           conf_int=True)
        ```            
        """
        from actxps.trx_stats import TrxStats
        return TrxStats(self, trx_types, percent_of, combine_trx,
                        full_exposures_only,
                        conf_int, conf_level, col_exposure)

    def exp_shiny(self,
                  predictors=None,
                  expected=None,
                  distinct_max=25):
        """
        Interactively explore experience data

        Launch a shiny application to interactively explore drivers of
        experience.

        Parameters
        ----------
        predictors : str | list | np.ndarray, default=`None`
            A character vector of independent variables in the `data` property 
            to include in the shiny app.
        expected : str | list | np.ndarray, default=`None`
            A character vector of expected values in the `data` property to
            include in the shiny app.
        distinct_max : int
            Maximum number of distinct values allowed for `predictors`
            to be included as "Color" and "Facets" grouping variables. This 
            input prevents the drawing of overly complex plots. Default 
            value = 25.

        Notes
        ----------
        If transactions have been attached to the `ExposedDF` object, the app
        will contain features for both termination and transaction studies.
        Otherwise, the app will only support termination studies.

        If nothing is passed to `predictors`, all columns names in `dat` will be
        used (excluding the policy number, status, termination date, exposure,
        transaction counts, and transaction amounts columns).

        The `expected` argument is optional. As a default, any column names
        containing the word "expected" are used.

        **Layout**

        *Filters*

        The sidebar contains filtering widgets for all variables passed
        to the `predictors` argument.

        *Study options*

        Grouping variables

        This box includes widgets to select grouping variables for summarizing
        experience. The "x" widget is also used as the x variable in the plot
        output. Similarly, the "Color" and "Facets" widgets are used for color
        and facets in the plot. Multiple faceting variables are allowed. For 
        the table output, "x", "Color", and "Facets" have no particular meaning 
        beyond the order in which of grouping variables are displayed.

        Study type

        This box also includes a toggle to switch between termination studies 
        and transaction studies (if available).

        - Termination studies:
        The expected values checkboxes are used to activate and deactivate
        expected values passed to the `expected` argument. This impacts the
        table output directly and the available "y" variables for the plot. If
        there are no expected values available, this widget will not appear.
        The "Weight by" widget is used to specify which column, if any, 
        contains weights for summarizing experience.

        - Transaction studies:
        The transaction types checkboxes are used to activate and deactivate
        transaction types that appear in the plot and table outputs. The
        available transaction types are taken from the `trx_types` property of 
        the `ExposedDF` object. In the plot output, transaction type will 
        always appear as a faceting variable. The "Transactions as % of"
        selector will expand the list of available "y" variables for the plot 
        and impact the table output directly. Lastly, a checkbox exists that 
        allows for all transaction types to be aggregated into a single group.

        **Output**

        *Plot Tab*

        This tab includes a plot and various options for customization:

        - y: y variable
        - Geometry: plotting geometry
        - Add Smoothing?: activate to plot loess curves
        - Free y Scales: activate to enable separate y scales in each plot.

        *Table*

        This tab includes a data table.

        *Export Data*

        This tab includes a download button that will save a copy of the 
        summarized experience data.

        **Filter Information**

        This box contains information on the original number of exposure 
        records, the number of records after filters are applied, and the 
        percentage of records retained.

        Examples 
        ----------
        ```{python}
        import actxps as xp
        import numpy as np

        census_dat = xp.load_census_dat()
        withdrawals = xp.load_withdrawals()
        account_vals = xp.load_account_vals()

        expo = xp.ExposedDF(census_dat, "2019-12-31",
                            target_status = "Surrender")
        expected_table = np.concatenate((np.linspace(0.005, 0.03, 10),
                                        [.2, .15], np.repeat(0.05, 3)))
        expo.data['expected_1'] = expected_table[expo.data.pol_yr - 1]
        expo.data['expected_2'] = np.where(expo.data.inc_guar, 0.015, 0.03)
        expo.add_transactions(withdrawals)
        expo.data = expo.data.merge(account_vals, how='left',
                                    on=["pol_num", "pol_date_yr"])

        app = expo.exp_shiny(expected=['expected_1', 'expected_2'])
        ```
        """
        from actxps.exp_shiny import _exp_shiny
        return _exp_shiny(self, predictors, expected, distinct_max)


def _most_common(x: pd.Series):
    """
    Determine the most common status

    Parameters
    ----------
    x : pd.Series
        A Series of policy statuses

    Returns
    ----------
    str
        Label of the most common policy status
    """
    return x.value_counts(sort=True).index[0]
