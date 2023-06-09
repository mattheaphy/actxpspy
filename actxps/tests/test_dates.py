from actxps.dates import *
import numpy as np
import pandas as pd


class TestPolDur():

    def test_pol_yr_1(self):
        assert pol_yr("2024-03-14", "2023-03-15") == 1

    def test_pol_yr_2(self):
        assert pol_yr("2024-03-15", "2023-03-15") == 2

    def test_pol_qtr_1(self):
        assert pol_qtr("2023-06-14", "2023-03-15") == 1

    def test_pol_qtr_2(self):
        assert pol_qtr("2023-06-15", "2023-03-15") == 2

    def test_pol_mth_1(self):
        assert pol_mth("2023-04-14", "2023-03-15") == 1

    def test_pol_mth_2(self):
        assert pol_mth("2023-04-15", "2023-03-15") == 2

    def test_pol_wk_1(self):
        assert pol_wk("2023-03-21", "2023-03-15") == 1

    def test_pol_wk_2(self):
        assert pol_wk("2023-03-22", "2023-03-15") == 2

    def test_vec_1(self):
        a = pol_yr(["2024-03-14", "2024-03-15", "2025-04-15"],
                   "2023-03-15")
        b = pol_yr(pd.to_datetime(["2024-03-14", "2024-03-15", "2025-04-15"]),
                   "2023-03-15")
        c = pol_yr(["2024-03-14", "2024-03-15", "2025-04-15"],
                   ["2023-03-15", "2023-03-15", "2023-03-15"])
        d = np.array([1, 2, 3])
        assert all(a == b)
        assert all(a == c)
        assert all(a == d)

    def test_vec_2(self):
        a = pol_yr("2024-03-14",
                   ["2023-03-15", "2023-03-15"])
        b = pol_yr(["2024-03-14", "2024-03-14"],
                   ["2023-03-15", "2023-03-15"])
        assert all(a == b)


class TestLeapYear():

    def test_leap_iss_1(self):
        assert pol_yr("2021-02-27", "2020-02-29") == 1

    def test_leap_iss_2(self):
        assert pol_yr("2021-02-28", "2020-02-29") == 2

    def test_leap_iss_3(self):
        assert pol_mth("2020-03-28", "2020-02-29") == 1

    def test_leap_iss_4(self):
        assert pol_mth("2020-03-29", "2020-02-29") == 2

    def test_228_iss_1(self):
        assert pol_yr("2024-02-27", "2023-02-28") == 1

    def test_228_iss_2(self):
        assert pol_yr("2024-02-29", "2023-02-28") == 2


class TestPolInterval():
    def test_year_interval(self):
        x = pol_yr("2021-02-28", "2020-02-29")
        y = pol_interval("2021-02-28", "2020-02-29", "year")
        assert x == y

    def test_month_interval(self):
        x = pol_interval("2022-03-14", "2022-01-05", "month")
        y = pol_mth("2022-03-14", "2022-01-05")
        assert x == y


class TestPolFrac():

    def test_frac_yr_1(self):
        assert frac_yr("2023-03-15", "2024-03-14") == 365 / 366

    def test_frac_yr_2(self):
        assert frac_yr("2023-03-15", "2024-03-15") == 1

    def test_frac_qtr_1(self):
        assert frac_qtr("2023-03-15", "2023-06-14") == (31+30+30) / (31+30+31)

    def test_frac_qtr_2(self):
        assert frac_qtr("2023-03-15", "2023-06-15") == 1

    def test_frac_mth_1(self):
        assert frac_mth("2023-03-15", "2023-04-14") == 30/31

    def test_frac_mth_2(self):
        assert frac_mth("2023-03-15", "2023-04-15") == 1

    def test_frac_wk_1(self):
        assert frac_wk("2023-03-15", "2023-03-21") == 6/7

    def test_frac_wk_2(self):
        assert frac_wk("2023-03-15", "2023-03-22") == 1

    def test_vec_1(self):
        a = frac_yr("2023-03-15",
                    ["2024-03-14", "2024-03-15", "2025-04-15"])
        b = frac_yr("2023-03-15",
                    pd.to_datetime(["2024-03-14", "2024-03-15", "2025-04-15"]))
        c = frac_yr(["2023-03-15", "2023-03-15", "2023-03-15"],
                    ["2024-03-14", "2024-03-15", "2025-04-15"])
        d = np.array([365/366, 1, 2 + 31/365])
        assert all(a == b)
        assert all(a == c)
        assert all(a == d)

    def test_vec_2(self):
        a = frac_yr(["2023-03-15", "2023-03-15"],
                    "2024-03-14")
        b = frac_yr(["2023-03-15", "2023-03-15"],
                    ["2024-03-14", "2024-03-14"])
        assert all(a == b)


class TestFracInterval():
    def test_frac_year_interval(self):
        x = frac_yr("2020-02-29", "2021-02-28")
        y = frac_interval("2020-02-29", "2021-02-28", "year")
        assert x == y

    def test_month_interval(self):
        x = frac_interval("2022-01-05", "2022-03-14", "month")
        y = frac_mth("2022-01-05", "2022-03-14")
        assert x == y


class TestAddDates():

    def test_add_yr_1(self):
        assert add_yr("2023-03-15", 1) == pd.to_datetime("2024-03-15")

    def test_add_yr_2(self):
        assert add_yr("2020-02-29", 1) == pd.to_datetime("2021-02-28")

    def test_add_qtr_1(self):
        assert add_qtr("2023-03-15", 1) == pd.to_datetime('2023-06-15')

    def test_add_qtr_2(self):
        assert add_qtr("2019-11-30", 1) == pd.to_datetime('2020-02-29')

    def test_add_mth_1(self):
        assert add_mth("2023-03-15", 2) == pd.to_datetime('2023-05-15')

    def test_add_mth_2(self):
        assert add_mth("2019-12-31", 2) == pd.to_datetime('2020-02-29')

    def test_add_wk_1(self):
        assert add_wk("2023-03-15", 2) == pd.to_datetime("2023-03-29")

    def test_vec_1(self):
        a = add_yr("2023-03-15", np.arange(3))
        b = add_yr(np.repeat("2023-03-15", 3),
                   np.arange(3))
        c = add_yr(["2023-03-15"] * 3,
                   [0, 1, 2])
        d = pd.to_datetime(["2023-03-15", "2024-03-15", "2025-03-15"])
        assert all(a == b)
        assert all(a == c)
        assert all(a == d)

    def test_vec_2(self):
        a = add_yr(["2023-03-15", "2023-03-15"], 1)
        b = add_yr("2023-03-15", [1, 1])
        assert all(a == b)


class TestAddInterval():
    def test_add_year_interval(self):
        x = add_yr("2020-02-29", 1)
        y = add_interval("2020-02-29", 1, "year")
        assert x == y

    def test_month_interval(self):
        x = add_interval("2022-01-05", 10, "month")
        y = add_mth("2022-01-05", 10)
        assert x == y
