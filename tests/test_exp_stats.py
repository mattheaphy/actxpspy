from actxps.expose import ExposedDF
import actxps as xp
import numpy as np
import pytest

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
    exp_stats(expected=["expected_1", "expected_2"], credibility=True,
              conf_int=True)

exp_res_weighted = study_py.groupby('pol_yr', 'inc_guar').\
    exp_stats(expected=["expected_1", "expected_2"], credibility=True,
              wt='weights', conf_int=True)


# Partial credibility is between 0 and 1
class TestPartialCred():

    def test_cred_lte1(self):
        assert all(exp_res.data.credibility <= 1)

    def test_cred_gte0(self):
        assert all(exp_res.data.credibility >= 0)


# Confidence intervals surround the observed surrender rate
class TestConfIntRange():

    def test_upper(self):
        assert all(exp_res.data.q_obs < exp_res.data.q_obs_upper)

    def test_lower(self):
        assert all(exp_res.data.q_obs > exp_res.data.q_obs_lower)

    def test_upper_wt(self):
        assert all(exp_res_weighted.data.q_obs <
                   exp_res_weighted.data.q_obs_upper)

    def test_upper_wt(self):
        assert all(exp_res_weighted.data.q_obs >
                   exp_res_weighted.data.q_obs_lower)


# Experience study summary method checks
class TestSummaryMethod():

    def test_regroup(self):
        assert all(exp_res.data == exp_res.summary('pol_yr', 'inc_guar').data)

    def test_nogroup(self):
        assert all(study_py.ungroup().exp_stats(
            expected=['expected_1', 'expected_2'],
            credibility=True, conf_int=True).data ==
            exp_res.summary().data)

    def test_nogroup_weighted(self):
        assert all(study_py.ungroup().exp_stats(
            expected=['expected_1', 'expected_2'],
            credibility=True, conf_int=True, wt='weights').data ==
            exp_res_weighted.summary().data)


# Confidence intervals work
class TestExpConfInt():

    def test_lower(self):
        assert all(exp_res.data.q_obs_lower < exp_res.data.q_obs)

    def test_upper(self):
        assert all(exp_res.data.q_obs_upper > exp_res.data.q_obs)

    def test_lower_wt(self):
        assert all(exp_res_weighted.data.q_obs_lower < 
                   exp_res_weighted.data.q_obs)

    def test_upper_wt(self):
        assert all(exp_res_weighted.data.q_obs_upper > 
                   exp_res_weighted.data.q_obs)
        
    def test_lower_ae(self):
        assert all(exp_res.data.ae_expected_1_lower < 
                   exp_res.data.ae_expected_1)
    
    def test_upper_ae(self):
        assert all(exp_res.data.ae_expected_2_upper > 
                   exp_res.data.ae_expected_2)
        
    def test_lower_ae_wt(self):
        assert all(exp_res_weighted.data.ae_expected_1_lower <
                   exp_res_weighted.data.ae_expected_1)
        
    def test_upper_ae_wt(self):
        assert all(exp_res_weighted.data.ae_expected_2_upper >
                   exp_res_weighted.data.ae_expected_2)

    # verify that confidence intervals are tighter using lower confidence
    def test_lower_confidence(self):
        less_confident = (study_py.groupby('pol_yr', 'inc_guar').
                          exp_stats(expected = ["expected_1", "expected_2"],
                                    credibility = True, conf_int = True, 
                                    conf_level = 0.5))
        assert all(exp_res.data.q_obs_upper - exp_res.data.q_obs_lower >
                   less_confident.data.q_obs_upper - 
                   less_confident.data.q_obs_lower)