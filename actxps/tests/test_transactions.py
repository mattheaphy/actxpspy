import actxps as xp
import numpy as np
import pytest

census_dat = xp.load_census_dat()
withdrawals = xp.load_withdrawals()
expo = xp.ExposedDF.expose_py(census_dat, "2019-12-31",
                              target_status="Surrender")
n = len(expo.data)
expo.add_transactions(withdrawals)
withdrawals4 = withdrawals.copy()
withdrawals4.columns = list('abcd')
withdrawals4.c = np.where(withdrawals4.c == 'Base', 'X', 'Y')


class TestAddTrx():

    def test_sum_amt(self):
        assert withdrawals.trx_amt.sum() == \
            expo.data[['trx_amt_Base', 'trx_amt_Rider']].sum().sum()

    def test_sum_count(self):
        assert len(withdrawals) == \
            expo.data[['trx_n_Base', 'trx_n_Rider']].sum().sum()

    def test_multi_calls(self):
        withdrawals2 = withdrawals.copy()
        withdrawals2.trx_type = np.where(withdrawals2.trx_type == 'Base',
                                         'A', 'B')
        withdrawals3 = withdrawals.copy()
        withdrawals3.trx_type = "something"
        expo.add_transactions(withdrawals2)
        expo.add_transactions(withdrawals3)

        assert n == len(expo.data)

    def test_int_error(self):
        with pytest.raises(AssertionError,  match="must be a DataFrame"):
            expo.add_transactions(1)


class TestTrxName():

    def test_name_error(self):
        with pytest.raises(AttributeError):
            expo.add_transactions(withdrawals4)

    def test_rename_works(self):
        expo.add_transactions(withdrawals4,
                              col_pol_num='a',
                              col_trx_date="b",
                              col_trx_type="c",
                              col_trx_amt="d")

    def test_name_conflict(self):
        with pytest.raises(ValueError, match='`trx_data` contains transaction'):
            expo.add_transactions(withdrawals)
