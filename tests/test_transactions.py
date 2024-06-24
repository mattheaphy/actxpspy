import actxps as xp
import polars as pl
import pytest

census_dat = xp.load_census_dat()
withdrawals = xp.load_withdrawals()
expo = xp.ExposedDF.expose_py(census_dat, "2019-12-31",
                              target_status="Surrender")
n = len(expo.data)
expo.add_transactions(withdrawals)
withdrawals4 = withdrawals.clone()
withdrawals4.columns = list('abcd')
withdrawals4 = withdrawals4.with_columns(
    c=pl.when(pl.col('c') == 'Base').then(pl.lit('X')).otherwise(pl.lit('Y'))
)


class TestAddTrx():

    def test_sum_amt(self):
        assert withdrawals['trx_amt'].sum() == \
            expo.data[['trx_amt_Base', 'trx_amt_Rider']].to_numpy().sum()

    def test_sum_count(self):
        assert len(withdrawals) == \
            expo.data[['trx_n_Base', 'trx_n_Rider']].to_numpy().sum()

    def test_multi_calls(self):
        withdrawals2 = withdrawals.clone().with_columns(
            trx_type=(pl.when(pl.col('trx_type') == 'Base').
                      then(pl.lit('A')).otherwise(pl.lit('B'))))
        withdrawals3 = withdrawals.clone().with_columns(
            trx_type=pl.lit('something')
        ).to_pandas()
        expo.add_transactions(withdrawals2)
        expo.add_transactions(withdrawals3)

        assert n == len(expo.data)

    def test_dupplicate_error(self):
        with pytest.raises(ValueError,  match="`trx_data` contains transaction"):
            expo.add_transactions(withdrawals)

    def test_int_error(self):
        with pytest.raises(AssertionError,  match="must be a DataFrame"):
            expo.add_transactions(1)


class TestTrxName():

    def test_name_error(self):
        with pytest.raises(pl.ColumnNotFoundError):
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


# Date format checks work"
class TestTrxDateFormatChecks():

    def test_error_missing_issue_dates(self):
        withdrawals5 = withdrawals.clone()
        withdrawals5[41, "trx_date"] = None

        with pytest.raises(
                AssertionError,
                match="Missing values are not allowed in the `trx_date`"):
            expo.add_transactions(withdrawals5)
