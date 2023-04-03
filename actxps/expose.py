import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
from datetime import datetime
from actxps.tools import arg_match
from actxps.dates import frac_interval, add_interval
from warnings import warn
from functools import singledispatchmethod
from scipy.stats import norm


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

    ## Parameters

    `data`: pd.DataFrame
        A data frame with census-level records
    `end_date`: datetime
        Experience study end date
    `**kwargs`:
        Additional parameters passed to `ExposedDF.__init__()`

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
                  cal_expo, expo_length):
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
                       col_pol_num: str = "pol_num",
                       col_status: str = "status",
                       col_exposure: str = "exposure",
                       col_pol_per: str = None,
                       cols_dates: str = None):

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

        # check required columns
        # pol_num, status, exposure, 2 date cols,
        # policy period (policy expo only)
        unmatched = {"pol_num", "status", "exposure", exp_col_pol_per}
        unmatched.update(exp_cols_dates)
        unmatched = unmatched.difference(data.columns)

        assert len(unmatched) == 0, \
            ("The following columns are missing from `data`: "
                f"{', '.join(unmatched)}.\nHint: create these columns or use "
                "the `col_*` arguments to specify existing columns that "
                "should be mapped to these elements.")

        return cls('already_exposed',
                   data, end_date, start_date, target_status, cal_expo,
                   expo_length)

    @__init__.register(str)
    def _special_init(self,
                      style,
                      data: pd.DataFrame,
                      end_date: datetime,
                      start_date: datetime = datetime(1900, 1, 1),
                      target_status: str = None,
                      cal_expo: bool = False,
                      expo_length: str = 'year'):
        """
        Special constructor for the ExposedDF class. This constructor is used
        by the `.from_DataFrame` class method to create new classes from 
        DataFrames that already contain exposure records.
        """

        assert style == "already_exposed"

        self._finalize(data, end_date, start_date, target_status,
                       cal_expo, expo_length)

    @staticmethod
    def _make_date_col_names(cal_expo: bool, expo_length: str):
        abbrev = ExposedDF.abbr_period[expo_length]
        x = ("cal_" if cal_expo else "pol_date_") + abbrev
        return x, x + "_end"

    def __repr__(self) -> str:
        repr = ("Exposure data\n\n" +
                f"Exposure type: {self.exposure_type}\n" +
                f"Target status: {', '.join([str(i) for i in self.target_status])}\n" +
                f"Study range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}" +
                f"\n\nA DataFrame: {self.data.shape[0]:,} x {self.data.shape[1]:,}"
                f'\n{self.data.head(10)}')

        return repr

    def groupby(self, *by):
        """
        Set grouping variables for summary methods like `.exp_stats()`.

        ## Parameters

        *`by`: 
            Column names in `data` that will be used as grouping variables

        ## Details

        This function will not directly apply the `DataFrame.groupby()` method 
        to the `data` property. Instead, it will set the `groups` property of
        the `ExposedDF` object. The `groups` property is subsequently used to
        group data within summary methods like `exp_stats()`.

        ## Returns

        self

        """

        by = list(by)

        assert all(pd.Series(by).isin(self.data.columns)), \
            "All grouping variables passed to `*by` must be in the`.data` property."

        self.groups = by
        return self
    
    def ungroup(self):
        """
        Remove all grouping variables for summary methods like `.exp_stats()`.

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

        # set up target statuses. First, attempt to use the statuses that
        # were passed to this method. If none, then use the target_status
        # property. If that property is None
        if target_status is None:
            target_status = self.target_status
        if target_status == [None]:
            target_status = list(self.data.status.cat.categories[1:])
            warn(f"No target status was provided. {', '.join(target_status)} "
                 "was assumed.")

        res = self.data.assign(
            n_claims=self.data.status.isin(target_status)
        )

        # set up weights
        if wt is not None:

            assert isinstance(wt, str), "`wt` must have type `str`"
            res = res.rename(columns={wt: 'weight'})
            res['claims'] = res.n_claims * res.weight
            res['exposure'] = res.exposure * res.weight
            res['weight_sq'] = res.weight ** 2
            res['weight_n'] = 1

        else:
            res['claims'] = res.n_claims

        # finish exp stats
        def _finish_exp_stats(res, expected, credibility,
                              cred_p, cred_r, wt):
            # dictionary of summarized values
            fields = {'n_claims': sum(res.n_claims),
                      'claims': sum(res.claims),
                      'exposure': sum(res.exposure)}

            if expected is not None:

                if isinstance(expected, str | int):
                    expected = [expected]

                ex_mean = {k: np.average(res[k], weights=res.exposure)
                           for k in expected}
                fields.update(ex_mean)

            # additional columns for weighted studies
            if wt is not None:
                wt_forms = {
                    'weight': sum(res.weight),
                    'weight_sq': sum(res.weight_sq),
                    'weight_n': sum(res.weight_n)}
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
                        'credibility':
                            np.minimum(1, lambda x:
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

            # convert results to a data frame
            res = pd.DataFrame(fields, index=range(1)).assign(**fields2)

            # add A/E's and adjusted q's
            if expected is not None:
                for k in expected:
                    res['ae_' + k] = res.q_obs / res[k]

            if credibility & (expected is not None):
                for k in expected:
                    res['adj_' + k] = (res.credibility * res.q_obs +
                                       (1 - res.credibility) * res[k])

            # rearrange and drop columns
            if wt is not None:
                res = res.drop(columns=['ex_wt', 'ex2_wt'])

            cols = (['n_claims', 'claims', 'exposure', 'q_obs'] +
                    [k for k in expected] +
                    ['ae_' + k for k in expected] +
                    (['credibility'] if credibility else None) +
                    (['adj_' + k for k in expected] if credibility else None) +
                    (['weight', 'weight_sq', 'weight_n']
                    if wt is not None else None)
                    )

            res = res[cols]

            return res

        if self.groups is not None:
            res = (res.groupby(self.groups).
                   apply(_finish_exp_stats,
                         expected=expected,
                         credibility=credibility,
                         cred_p=cred_p,
                         cred_r=cred_r, wt=wt).
                   reset_index().
                   drop(columns=[f'level_{len(self.groups)}']))

        else:
            res = _finish_exp_stats(res,
                                    expected=expected,
                                    credibility=credibility,
                                    cred_p=cred_p,
                                    cred_r=cred_r, wt=wt)
        return res
