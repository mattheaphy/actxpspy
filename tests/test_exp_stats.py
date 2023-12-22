from actxps.expose import ExposedDF
import actxps as xp
import numpy as np

census_dat = xp.load_census_dat()

study_py = ExposedDF.expose_py(census_dat, "2019-12-31",
                               target_status="Surrender")

expected_table = np.concatenate((np.linspace(0.005, 0.03, 10),
                                 [.2, .15], np.repeat(0.05, 3)))

np.random.seed(123)

study_py.data['expected_1'] = expected_table[study_py.data.pol_yr - 1]
study_py.data['expected_2'] = np.where(study_py.data.inc_guar, 0.015, 0.03)
study_py.data['weights'] = np.abs(
    np.random.normal(100, 50, len(study_py.data)))

exp_res = study_py.groupby('pol_yr', 'inc_guar').\
    exp_stats(expected=["expected_1", "expected_2"], credibility=True)

exp_res_weighted = study_py.groupby('pol_yr', 'inc_guar').\
    exp_stats(expected=["expected_1", "expected_2"], credibility=True,
              wt='weights')


class TestPartialCred():

    def test_cred_lte1(self):
        assert all(exp_res.data.credibility <= 1)

    def test_cred_gte0(self):
        assert all(exp_res.data.credibility >= 0)


class TestSummaryMethod():

    def test_regroup(self):
        assert all(exp_res.data == exp_res.summary('pol_yr', 'inc_guar').data)

    def test_nogroup(self):
        assert all(study_py.ungroup().exp_stats(
            expected=['expected_1', 'expected_2'],
            credibility=True).data ==
            exp_res.summary().data)
    
    def test_nogroup_weighted(self):
        assert all(study_py.ungroup().exp_stats(
            expected=['expected_1', 'expected_2'],
            credibility=True, wt = 'weights').data ==
            exp_res_weighted.summary().data)