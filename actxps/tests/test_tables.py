from actxps import ExposedDF
import actxps as xp
import numpy as np
import pandas as pd

census_dat = xp.load_census_dat()

expo = ExposedDF.expose_py(census_dat, "2019-12-31",
                           target_status="Surrender")
expo.add_transactions(xp.load_withdrawals())
expo.data['q_exp'] = np.where(expo.data.inc_guar, 0.015, 0.03)
expo.groupby('pol_yr', 'inc_guar')

exp_res = expo.exp_stats()
exp_res2 = expo.exp_stats(wt="premium", credibility=True,
                          expected="q_exp")
_ = exp_res.table()

trx_res = expo.trx_stats()
trx_res2 = expo.trx_stats(percent_of='premium')

styler = pd.io.formats.style.Styler


class TestAutotable():

    def test_table_basic(self):
        assert isinstance(exp_res.table(), styler)

    def test_table_options(self):
        assert isinstance(exp_res2.table(), styler)

    def test_table_renames(self):
        assert isinstance(exp_res2.table(
            rename_cols={'pol_yr': 'Policy year',
                         'inc_guar': 'GLWB'}), styler)


class TestTrxAutotable():

    def test_table_basic(self):
        assert isinstance(trx_res.table(), styler)

    def test_table_options(self):
        assert isinstance(trx_res2.table(), styler)

    def test_table_renames(self):
        assert isinstance(trx_res2.table(
            rename_cols={'pol_yr': 'Policy year',
                         'inc_guar': 'GLWB'}), styler)
