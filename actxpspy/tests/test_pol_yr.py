from actxpspy.pol_yr import *
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
        y = pol_interval("2021-02-28", "2020-02-29", "Y")
        assert x == y
        
    def test_month_interval(self):
        x = pol_interval("2022-03-14", "2022-01-05", "M")
        y = pol_mth("2022-03-14", "2022-01-05")
        assert x == y
