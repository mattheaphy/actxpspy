from actxps.expose import *
from actxps.datasets import *
from actxps.expose_split import SplitExposedDF
import pandas as pd
from pandas.tseries.offsets import Day
import pytest

toy_census = load_toy_census()
census_dat = load_census_dat()
study_py = ExposedDF.expose_py(census_dat, "2019-12-31",
                               target_status="Surrender")
study_cy = ExposedDF.expose_cy(census_dat, "2019-12-31",
                               target_status="Surrender")


class TestExposeInit():

    def test_bad_expo_length(self):
        with pytest.raises(ValueError, match='must be one of'):
            ExposedDF(toy_census, '2022-12-31', expo_length='quantum')


# Policy year exposure checks
class TestPolExpo():

    def test_min_expo(self):
        assert all(study_py.data.exposure >= 0)

    def test_max_expo(self):
        assert all(study_py.data.exposure <= 1)

    def test_no_na(self):
        assert study_py.data.exposure.isna().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_py.data.loc[study_py.data.status == "Surrender",
                                     "exposure"] == 1)


# Calendar year exposure checks
class TestCalExpo():

    def test_min_expo(self):
        assert all(study_cy.data.exposure >= 0)

    def test_max_expo(self):
        assert all(study_cy.data.exposure <= 1)

    def test_no_na(self):
        assert study_cy.data.exposure.isna().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_cy.data.loc[study_cy.data.status == "Surrender",
                                     "exposure"] == 1)


check_period_end_pol = (ExposedDF.expose_pw(toy_census, "2020-12-31",
                                            target_status="Surrender").
                        data[['pol_num', 'pol_date_wk', 'pol_date_wk_end']])
check_period_end_pol['x'] = (check_period_end_pol.
                             groupby('pol_num').
                             pol_date_wk.
                             shift(-1))
check_period_end_pol = check_period_end_pol.dropna()
check_period_end_pol = check_period_end_pol.\
    loc[check_period_end_pol.x !=
        check_period_end_pol.pol_date_wk_end + Day(1)].shape[0]

check_period_end_cal = (ExposedDF.expose_cm(toy_census, "2020-12-31",
                                            target_status="Surrender").
                        data[['pol_num', 'cal_mth', 'cal_mth_end']])
check_period_end_cal['x'] = (check_period_end_cal.
                             groupby('pol_num').
                             cal_mth.
                             shift(-1))
check_period_end_cal = check_period_end_cal.dropna()
check_period_end_cal = check_period_end_cal.\
    loc[check_period_end_cal.x !=
        check_period_end_cal.cal_mth_end + Day(1)].shape[0]


# Period start and end dates roll
class TestRollDates():

    def test_beg_lt_end1(self):
        assert all(study_py.data.pol_date_yr < study_py.data.pol_date_yr_end)

    def test_beg_lt_end1(self):
        assert all(study_cy.data.cal_yr < study_cy.data.cal_yr_end)

    def test_roll_1(self):
        assert check_period_end_pol == 0

    def test_roll_1(self):
        assert check_period_end_cal == 0


leap_day = pd.DataFrame({'pol_num': 1,
                         'status': 'Active',
                         'issue_date': pd.to_datetime('2020-02-29'),
                         'term_date': pd.NaT},
                        index=[0])

leap_expose = ExposedDF.expose_pm(leap_day, end_date="2021-02-28")

march_1 = pd.DataFrame({'pol_num': 1,
                        'status': 'Active',
                        'issue_date': pd.to_datetime('2019-03-01'),
                        'term_date': pd.NaT},
                       index=[0])

march_1_expose = ExposedDF.expose_pm(march_1, end_date="2020-02-29")


# Test leap day stability
class TestLeapStability():

    def test_leap(self):
        assert len(leap_expose.data) == 12

    def test_march_1(self):
        assert len(march_1_expose.data) == 12


with_start_date = ExposedDF.expose_py(
    census_dat, "2019-12-31",
    start_date="2018-12-31",
    target_status="Surrender"
)


# Start and end dates work
class TestStartEnd():

    def test_min_date(self):
        assert min(with_start_date.data.pol_date_yr) == \
            pd.to_datetime("2018-12-31")

    def test_man_date(self):
        assert max(with_start_date.data.pol_date_yr) == \
            pd.to_datetime("2019-12-31")


exposed_strings = ExposedDF(toy_census, "2020-12-31", "2016-04-01")
exposed_dates = ExposedDF(toy_census, pd.to_datetime("2020-12-31"),
                          pd.to_datetime("2016-04-01"))


# Expose date arguments can be passed strings
class TestDateArgString:

    def test_same(self):
        assert exposed_strings.data.equals(exposed_dates.data)


renamer = {"pol_num": "a",
           "status": "b",
           "issue_date": "c",
           "term_date": "d"}
toy_census2 = toy_census.rename(columns=renamer)


# Renaming and name conflict warnings work
class TestRenames():

    def test_name_error(self):
        with pytest.raises(AttributeError, match='object has no attribute'):
            ExposedDF(toy_census2, '2020-12-31')

    def test_rename_works(self):
        ExposedDF(toy_census2, "2020-12-31",
                  col_pol_num="a",
                  col_status="b",
                  col_issue_date="c",
                  col_term_date="d")

    def test_warn_conflict_expo(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.assign(exposure=1), "2020-12-31")

    def test_warn_conflict_pol_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.assign(pol_yr=1), "2020-12-31")

    def test_warn_conflict_pol_date_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.assign(pol_date_yr=1), "2020-12-31")

    def test_warn_conflict_pol_date_yr_end(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.assign(pol_date_yr_end=1), "2020-12-31")

    def test_warn_conflict_cal_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF.expose_cy(toy_census.assign(cal_yr=1), "2020-12-31")

    def test_warn_conflict_cal_yr_end(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF.expose_cy(toy_census.assign(cal_yr_end=1), "2020-12-31")


expo = ExposedDF(toy_census, "2020-12-31", target_status="Surrender")
expo2 = expo.data.copy()
expo3 = expo2.rename(columns={'pol_num': 'pnum'})
expo4 = expo3.rename(columns={'status': 'stat',
                              'exposure': 'expo',
                              'pol_yr': 'py',
                              'pol_date_yr': 'start',
                              'pol_date_yr_end': 'end'})


# .from_DataFrame method
class TestFromDataFrame():

    def test_wrong_format(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            x = pd.DataFrame({'a': range(3)})
            ExposedDF.from_DataFrame(x, '2019-12-31')

    def test_bad_expo_length(self):
        with pytest.raises(ValueError, match='must be one of'):
            ExposedDF.from_DataFrame(expo2,
                                     end_date="2022-12-31",
                                     expo_length="yr")

    def test_from_DataFrame_works(self):
        assert isinstance(ExposedDF.from_DataFrame(expo2, '2022-12-31'),
                          ExposedDF)

    def test_bad_colnames(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            ExposedDF.from_DataFrame(expo3, '2019-12-31')

    def test_rename_works(self):
        assert isinstance(
            ExposedDF.from_DataFrame(expo3, '2019-12-31', col_pol_num='pnum'),
            ExposedDF)

    def test_rename_works_2(self):
        assert isinstance(
            ExposedDF.from_DataFrame(expo4, '2019-12-31', col_pol_num='pnum',
                                     col_status='stat', col_exposure='expo',
                                     cols_dates=['start', 'end'],
                                     col_pol_per='py'),
            ExposedDF)

    def test_only_dataframe(self):
        with pytest.raises(AssertionError, match='must be a Pandas DataFrame'):
            ExposedDF.from_DataFrame(1, '2020-12-31')


expo6 = expo2.copy()
expo6['trx_n_A'] = 1
expo6['trx_amt_A'] = 2
expo6['trx_n_B'] = 3
expo6['trx_amt_B'] = 4
expo7 = expo6.copy().rename(columns={
    'trx_n_A': 'n_A',
    'trx_n_B': 'n_B',
    'trx_amt_A': 'amt_A',
    'trx_amt_B': 'amt_B'
})


# .from_DataFrame method with transactions
class TestFromDataFrameTrx():

    def test_from_df_w_trx_works(self):
        assert \
            isinstance(ExposedDF.from_DataFrame(expo6, "2022-12-31",
                                                trx_types=['A', 'B']),
                       ExposedDF)

    def test_from_df_w_bad_trx(self):
        with pytest.raises(AssertionError, match='The following columns are missing'):
            ExposedDF.from_DataFrame(expo6,
                                     "2022-12-31", trx_types=['A', 'C'])

    def test_from_df_w_trx_no_rename(self):
        with pytest.raises(AssertionError, match='The following columns are missing'):
            ExposedDF.from_DataFrame(expo7,
                                     "2022-12-31", trx_types=['A', 'B'])

    def test_from_df_w_trx_and_rename(self):
        assert \
            isinstance(ExposedDF.from_DataFrame(expo7,
                                                "2022-12-31", trx_types=['A', 'B'],
                                                col_trx_amt_='amt_', col_trx_n_='n_'),
                       ExposedDF)


# from_DataFrame default_status works
class TestDefaultStatus():

    def test_default(self):
        assert ExposedDF.from_DataFrame(expo2, "2022-12-31").default_status == \
            'Active'

    def test_default_override(self):
        assert ExposedDF.from_DataFrame(
            expo2, "2022-12-31", default_status='Inforce').default_status == \
            'Inforce'

    def test_default2(self):
        assert expo.default_status == 'Active'


# split exposure tests

# expose_split() fails when passed non-calendar ExposedDF's
class TestSplitTypeErrors():

    def test_int(self):
        with pytest.raises(AssertionError,
                           match="An `ExposedDF` object is required"):
            SplitExposedDF(1)

    def test_pol_expo(self):
        with pytest.raises(AssertionError,
                           match="has calendar exposures"):
            ExposedDF.expose_py(toy_census, "2022-12-31").expose_split()

    def test_cal_expo(self):
        assert isinstance(
            ExposedDF.expose_cy(toy_census, "2022-12-31").expose_split(),
            SplitExposedDF)


withdrawals = load_withdrawals()
study_split = study_cy.expose_split().add_transactions(withdrawals)
study_cy = study_cy.add_transactions(withdrawals)


# expose_split() is consistent with expose_cy()
class TestSplitEquivalence():

    def test_expo(self):
        assert abs(sum(study_cy.data.exposure) -
                   sum(study_split.data.exposure_cal)) < 1E-8

    def test_term_count(self):
        assert sum(study_cy.data.status != "Active") == \
            sum(study_split.data.status != "Active")

    def test_trx_amt_1(self):
        assert sum(study_cy.data.trx_amt_Base) == \
            sum(study_split.data.trx_amt_Base)

    def test_trx_amt_2(self):
        assert sum(study_cy.data.trx_amt_Rider) == \
            sum(study_split.data.trx_amt_Rider)


# expose_split() warns about transactions attached too early
class TestSplitErrorsWarnings():
    def test_dup_trx_warning(self):
        with pytest.warns(UserWarning,
                          match="This will lead to duplication of transactions"):
            study_cy.expose_split()

    def test_unclear_expo_exp_stats(self):
        with pytest.raises(AssertionError,
                           match="A `SplitExposedDF` was passed without"):
            study_split.exp_stats()

    def test_unclear_expo_trx_stats(self):
        with pytest.raises(AssertionError,
                           match="A `SplitExposedDF` was passed without"):
            study_split.trx_stats()


split_dat = (
    ExposedDF.expose_cy(
        toy_census, "2020-12-31", target_status="Surrender").
    expose_split().
    data[["pol_num", "cal_yr", "cal_yr_end"]])
split_dat['x'] = split_dat.groupby('pol_num')['cal_yr'].shift(-1)
split_dat = split_dat.dropna()

check_period_end_split = (split_dat[
    split_dat.x != split_dat.cal_yr_end + Day(1)].
    shape[0]
)


# Split period start and end dates roll"
class TestSplitDateRoll():

    def test_study_split_roll_1(self):
        assert all(study_split.data.cal_yr <= study_split.data.cal_yr_end)

    def test_study_split_roll_2(self):
        assert check_period_end_split == 0
