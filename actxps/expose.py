import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
from datetime import datetime
from actxps.tools import arg_match, document
from actxps.dates import frac_interval, add_interval
from warnings import warn
from pandas.api.types import is_categorical_dtype


class ExposedDF():
    """
    # Exposed data frame class

    ## Parameters

    ## Details

    ## Methods

    ## Properties

    """

    # helper dictionary for abbreviations
    abbr_period = {
        "year": "yr",
        "quarter": "qtr",
        "month": "mth",
        "week": "wk"
    }

    def __init__(self,
                 data: pd.DataFrame,
                 end_date: datetime,
                 start_date: datetime = datetime(1900, 1, 1),
                 target_status: str = None,
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
            ('cal_' if cal_expo else 'pol_date') + abbrev + '_end'
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

            # partial exposure calculations
            expo_cond = [
                data.status.isin in target_status,
                data.first_per & data.last_per,
                data.first_per,
                data.last_per
            ]

            expo_choice = [
                1,
                per_frac(data.first_date - Day(1), data.last_date),
                1 - per_frac(data.cal_b, data.first_date),
                per_frac(data.cal_b - Day(1), data.last_date),
            ]

            data['exposure'] = np.select(expo_cond, expo_choice, 1)

            data = (data.
                    drop(columns={'rep_n', 'first_date', 'last_date',
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
        self.data = data
        self.end_date = end_date
        self.start_date = start_date
        self.target_status = target_status
        self.cal_expo = cal_expo
        self.expo_length = expo_length
        self.trx_types = None

        return self

    @classmethod
    def from_DataFrame():
        # TODO
        pass

    def exp_stats(self):
        # TODO
        pass
