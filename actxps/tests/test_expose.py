from actxps.expose import *
from actxps.datasets import *
import pandas as pd
import numpy as np
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


class TestPolExpo():

    def test_min_expo(self):
        assert all(study_py.data.exposure >= 0)

    def test_max_expo(self):
        assert all(study_py.data.exposure <= 1)

    def test_no_na(self):
        assert study_py.data.exposure.isna().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_py.data.loc[study_py.data.status == "Surrender"] == 1)


class TestCalExpo():

    def test_min_expo(self):
        assert all(study_cy.data.exposure >= 0)

    def test_max_expo(self):
        assert all(study_cy.data.exposure <= 1)

    def test_no_na(self):
        assert study_cy.data.exposure.isna().sum() == 0

    def test_full_expo_targ(self):
        assert all(study_cy.data.loc[study_cy.data.status == "Surrender"] == 1)


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


class TestDateArgString:

    def test_same(self):
        assert exposed_strings.data.equals(exposed_dates.data)


renamer = {"pol_num": "a",
           "status": "b",
           "issue_date": "c",
           "term_date": "d"}
toy_census2 = toy_census.rename(columns=renamer)


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
