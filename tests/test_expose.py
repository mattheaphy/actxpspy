from actxps.expose import *
from actxps.datasets import *
from actxps.expose_split import SplitExposedDF
from datetime import date
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
        assert all(study_py.data['exposure'] >= 0)

    def test_max_expo(self):
        assert all(study_py.data['exposure'] <= 1)

    def test_no_na(self):
        assert study_py.data['exposure'].is_nan().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_py.data.
                   filter(pl.col('status') == "Surrender")['exposure'] == 1)

    def test_pandas(self):
        assert isinstance(ExposedDF.expose_cy(census_dat.to_pandas(),
                                              "2019-12-31",
                                              target_status="Surrender"),
                          ExposedDF)


# Calendar year exposure checks
class TestCalExpo():

    def test_min_expo(self):
        assert all(study_cy.data['exposure'] >= 0)

    def test_max_expo(self):
        assert all(study_cy.data['exposure'] <= 1)

    def test_no_na(self):
        assert study_cy.data['exposure'].is_nan().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_cy.data.
                   filter(pl.col('status') == "Surrender")["exposure"] == 1)

    def test_pandas(self):
        assert isinstance(ExposedDF.expose_py(census_dat.to_pandas(),
                                              "2019-12-31",
                                              target_status="Surrender"),
                          ExposedDF)


check_period_end_pol = (
    ExposedDF.expose_pw(toy_census, "2020-12-31",
                        target_status="Surrender").
    data[['pol_num', 'pol_date_wk', 'pol_date_wk_end']].
    with_columns(
        x=pl.col('pol_date_wk').shift(-1).over('pol_num')
    ).drop_nulls().filter(
        pl.col('x') != pl.col('pol_date_wk_end').dt.offset_by('1d')
    ).height)

check_period_end_cal = (
    ExposedDF.expose_cm(toy_census, "2020-12-31",
                        target_status="Surrender").
    data[['pol_num', 'cal_mth', 'cal_mth_end']].
    with_columns(
        x=pl.col('cal_mth').shift(-1).over('pol_num')
    ).
    drop_nulls().filter(
        pl.col('x') != pl.col('cal_mth_end').dt.offset_by('1d')
    ).height)


# Period start and end dates roll
class TestRollDates():

    def test_beg_lt_end1(self):
        assert all(study_py.data['pol_date_yr'] <
                   study_py.data['pol_date_yr_end'])

    def test_beg_lt_end1(self):
        assert all(study_cy.data['cal_yr'] < study_cy.data['cal_yr_end'])

    def test_roll_1(self):
        assert check_period_end_pol == 0

    def test_roll_1(self):
        assert check_period_end_cal == 0


leap_day = pl.DataFrame({'pol_num': 1,
                         'status': 'Active',
                         'issue_date': date(2020, 2, 29),
                         'term_date': None})

leap_expose = ExposedDF.expose_pm(leap_day, end_date="2021-02-28")

march_1 = pl.DataFrame({'pol_num': 1,
                        'status': 'Active',
                        'issue_date': date(2019, 3, 1),
                        'term_date': None})

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
        assert min(with_start_date.data['pol_date_yr']) == date(2018, 12, 31)

    def test_man_date(self):
        assert max(with_start_date.data['pol_date_yr']) == date(2019, 12, 31)


# All terminations have termination dates
class TestTermDates():

    def test_term_date_py(self):
        assert (study_py.data['status'] != "Active").sum() == \
            study_py.data['term_date'].is_not_null().sum()

    def test_term_date_cy(self):
        assert (study_cy.data['status'] != "Active").sum() == \
            study_cy.data['term_date'].is_not_null().sum()


exposed_strings = ExposedDF(toy_census, "2020-12-31", "2016-04-01")
exposed_dates = ExposedDF(toy_census, date(2020, 12, 31), date(2016, 4, 1))


# Expose date arguments can be passed strings
class TestDateArgString:

    def test_same(self):
        assert exposed_strings.data.equals(exposed_dates.data)


renamer = {"pol_num": "a",
           "status": "b",
           "issue_date": "c",
           "term_date": "d"}
toy_census2 = toy_census.rename(renamer)


# Renaming and name conflict warnings work
class TestRenames():

    def test_name_error(self):
        with pytest.raises(pl.ColumnNotFoundError, match='term_date'):
            ExposedDF(toy_census2, '2020-12-31')

    def test_rename_works(self):
        ExposedDF(toy_census2, "2020-12-31",
                  col_pol_num="a",
                  col_status="b",
                  col_issue_date="c",
                  col_term_date="d")

    def test_warn_conflict_expo(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.with_columns(exposure=1), "2020-12-31")

    def test_warn_conflict_pol_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.with_columns(pol_yr=1), "2020-12-31")

    def test_warn_conflict_pol_date_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.with_columns(pol_date_yr=1), "2020-12-31")

    def test_warn_conflict_pol_date_yr_end(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF(toy_census.with_columns(pol_date_yr_end=1), "2020-12-31")

    def test_warn_conflict_cal_yr(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF.expose_cy(
                toy_census.with_columns(cal_yr=1), "2020-12-31")

    def test_warn_conflict_cal_yr_end(self):
        with pytest.warns(UserWarning, match="`data` contains the following"):
            ExposedDF.expose_cy(toy_census.with_columns(cal_yr_end=1),
                                "2020-12-31")


# Date format checks work
class TestDateFormatChecks():

    def test_error_missing_issue_dates(self):
        toy_census3 = toy_census.clone()
        toy_census3[0, "issue_date"] = None

        with pytest.raises(
                AssertionError,
                match="Missing values are not allowed in the `issue_date`"):
            ExposedDF.expose_py(toy_census3, "2020-12-31")


# An error is thrown if the default status is a target status
class TestDefaultTargetStatusCollision():

    def test_collision(self):
        all_deaths = pl.DataFrame({
            "pol_num": range(1, 3),
            "status": ["Death"] * 2,
            "issue_date": ["2011-05-27"] * 2,
            "term_date": ["2012-03-17", "2012-09-17"]})

        with pytest.raises(
            AssertionError,
            match="`default_status` is not allowed to be the same as `target_status"
        ):
            ExposedDF(all_deaths, end_date="2022-12-31",
                      target_status=["Death", "Surrender"])


expo = ExposedDF(toy_census, "2020-12-31", target_status="Surrender")
expo2 = expo.data.clone()
expo3 = expo2.rename({'pol_num': 'pnum'})
expo4 = expo3.rename({'status': 'stat',
                      'exposure': 'expo',
                      'pol_yr': 'py',
                      'pol_date_yr': 'start',
                      'pol_date_yr_end': 'end'})


# .from_DataFrame method
class TestFromDataFrame():

    def test_wrong_format(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            x = pl.DataFrame({'a': range(3)})
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
        with pytest.raises(AssertionError, match='must be a DataFrame'):
            ExposedDF.from_DataFrame(1, '2020-12-31')

    def test_pandas(self):
        assert isinstance(
            ExposedDF.from_DataFrame(expo2.to_pandas(), '2019-12-31'),
            ExposedDF)


expo6 = expo2.clone()
expo6 = expo6.with_columns(
    trx_n_A=1,
    trx_amt_A=2,
    trx_n_B=3,
    trx_amt_B=4)
expo7 = expo6.clone().rename({
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


def py_sum_check(py, split):
    py_sums = (py.data.group_by('pol_num', 'pol_yr').
               agg(exposure=pl.col('exposure').sum()).
               with_columns(pl.col('pol_yr').cast(int)))
    split_sums = (split.data.group_by('pol_num', 'pol_yr').
                  agg(exposure_pol=pl.col('exposure_pol').sum()).
                  with_columns(pl.col('pol_yr').cast(int)))
    return (py_sums.join(split_sums, on=['pol_num', 'pol_yr'],
                         how='inner').
            filter((pl.col('exposure') - pl.col('exposure_pol')).abs() > 1E-8).
            height)


# expose_split() is consistent with expose_cy()
class TestSplitEquivalence():

    def test_expo(self):
        assert abs(sum(study_cy.data['exposure']) -
                   sum(study_split.data['exposure_cal'])) < 1E-8

    def test_term_count(self):
        assert sum(study_cy.data['status'] != "Active") == \
            sum(study_split.data['status'] != "Active")

    def test_trx_amt_1(self):
        assert sum(study_cy.data['trx_amt_Base']) == \
            sum(study_split.data['trx_amt_Base'])

    def test_trx_amt_2(self):
        assert sum(study_cy.data['trx_amt_Rider']) == \
            sum(study_split.data['trx_amt_Rider'])

    def test_min_max_expo_cy(self):
        assert study_split.data["exposure_cal"].is_between(0, 1).all()

    def test_min_max_expo_py(self):
        assert study_split.data["exposure_pol"].is_between(0, 1).all()

    def test_expo_py(self):
        assert py_sum_check(study_py, study_split) == 0


study_py2 = ExposedDF.expose_py(census_dat, "2019-02-27",
                                target_status="Surrender",
                                start_date="2010-06-15")
study_cy2 = ExposedDF.expose_cq(census_dat, "2019-02-27",
                                target_status="Surrender",
                                start_date="2010-06-15")
study_split2 = study_cy2.expose_split()


# expose_split() is consistent with expose_cy() when using atypical start and
#   end dates
class TestSplitEquivalence2():

    def test_expo(self):
        assert abs(sum(study_cy2.data['exposure']) -
                   sum(study_split2.data['exposure_cal'])) < 1E-8

    def test_term_count(self):
        assert sum(study_cy2.data['status'] != "Active") == \
            sum(study_split2.data['status'] != "Active")

    def test_min_max_expo_cy(self):
        assert study_split2.data["exposure_cal"].is_between(0, 1).all()

    def test_min_max_expo_py(self):
        assert study_split2.data["exposure_pol"].is_between(0, 1).all()

    def test_expo_py(self):
        assert py_sum_check(study_py2, study_split2) == 0


# odd census
odd_census = pl.DataFrame(
    # death in first month
    [["D1", "Death", "2022-04-15", "2022-04-25"],
     # death in first year
     ["D2", "Death", "2022-04-15", "2022-09-25"],
        # death after 18 months
        ["D3", "Death", "2022-04-15", "2023-09-25"],
        # surrender in first month
        ["S1", "Surrender", "2022-11-10", "2022-11-20"],
        # surrender in first year
        ["S2", "Surrender", "2022-11-10", "2023-3-20"],
        # surrender after 18 months
        ["S3", "Surrender", "2022-11-10", "2024-3-20"],
        # active
        ["A", "Active", "2022-6-20", None]],
    schema=['pol_num', 'status', 'issue_date', 'term_date'],
    orient="row"
)

odd_study = ExposedDF.expose_cm(odd_census, "2024-05-19",
                                target_status="Surrender",
                                default_status="Active",
                                start_date="2022-04-10")
odd_py = ExposedDF.expose_py(odd_census, "2024-05-19",
                             target_status="Surrender",
                             default_status="Active",
                             start_date="2022-04-10")
odd_split = odd_study.expose_split()


# expose_split() checks with odd dates
class TestSplitOdddates():

    def test_expo(self):
        assert abs(sum(odd_study.data['exposure']) -
                   sum(odd_split.data['exposure_cal'])) < 1E-8

    def test_term_count(self):
        assert sum(odd_study.data['status'] != "Active") == \
            sum(odd_split.data['status'] != "Active")

    def test_min_max_expo_cy(self):
        assert odd_split.data["exposure_cal"].is_between(0, 1).all()

    def test_min_max_expo_py(self):
        assert odd_split.data["exposure_pol"].is_between(0, 1).all()

    def test_expo_py(self):
        assert abs(odd_py.data["exposure"].sum() -
                   odd_split.data["exposure_pol"].sum()) < 1E-8


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


check_period_end_split = (
    ExposedDF.expose_cy(
        toy_census, "2020-12-31", target_status="Surrender").
    expose_split().
    data[["pol_num", "cal_yr", "cal_yr_end"]].
    with_columns(
        x=pl.col('cal_yr').shift(-1).over('pol_num')
    ).drop_nulls().
    filter(pl.col('x') != pl.col('cal_yr_end').dt.offset_by('1d')).
    height)


# Split period start and end dates roll"
class TestSplitDateRoll():

    def test_study_split_roll_1(self):
        assert all(study_split.data['cal_yr'] <=
                   study_split.data['cal_yr_end'])

    def test_study_split_roll_2(self):
        assert check_period_end_split == 0
