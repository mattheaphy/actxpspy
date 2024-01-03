from actxps import ExposedDF
import actxps as xp
import numpy as np
from great_tables import GT
import pytest

census_dat = xp.load_census_dat()

expo = ExposedDF.expose_py(census_dat, "2019-12-31",
                           target_status="Surrender")
expo.add_transactions(xp.load_withdrawals())
expo.data['q_exp'] = np.where(expo.data.inc_guar, 0.015, 0.03)
expo.groupby('pol_yr', 'inc_guar')

exp_res = expo.exp_stats()
exp_res2 = expo.exp_stats(wt="premium", credibility=True,
                          expected="q_exp", conf_int=True)

trx_res = expo.trx_stats()
trx_res2 = expo.trx_stats(percent_of='premium', conf_int=True)


# Termination study table tests
class TestAutotable():

    def test_table_basic(self):
        assert isinstance(exp_res.table(), GT)

    def test_table_options(self):
        assert isinstance(exp_res2.table(show_conf_int=True), GT)

    def test_table_renames(self):
        assert isinstance(exp_res2.table(
            pol_yr='Policy year',
            inc_guar='GLWB'), GT)

    def test_table_ci_warning(self):
        with pytest.warns(match="has no confidence intervals"):
            exp_res.table(show_conf_int=True)

    def test_table_cred_adj_warning(self):
        with pytest.warns(match="has no credibility-weighted"):
            exp_res.table(show_cred_adj=True)


# Transaction study table tests
class TestTrxAutotable():

    def test_table_basic(self):
        assert isinstance(trx_res.table(), GT)

    def test_table_options(self):
        assert isinstance(trx_res2.table(show_conf_int=True), GT)

    def test_table_renames(self):
        assert isinstance(trx_res2.table(
            pol_yr='Policy year',
            inc_guar='GLWB'), GT)

    def test_table_ci_warning(self):
        with pytest.warns(match="has no confidence intervals"):
            trx_res.table(show_conf_int=True)
