from actxps.expose import ExposedDF
import actxps as xp
import numpy as np
import pandas as pd
import pytest
from copy import deepcopy

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

exp_res = study_py.group_by('pol_yr', 'inc_guar').\
    exp_stats(expected=["expected_1", "expected_2"], credibility=True,
              conf_int=True)

exp_res_weighted = study_py.group_by('pol_yr', 'inc_guar').\
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
        less_confident = (study_py.group_by('pol_yr', 'inc_guar').
                          exp_stats(expected=["expected_1", "expected_2"],
                                    credibility=True, conf_int=True,
                                    conf_level=0.5))
        assert all(exp_res.data.q_obs_upper - exp_res.data.q_obs_lower >
                   less_confident.data.q_obs_upper -
                   less_confident.data.q_obs_lower)


# Test that from_DataFrame works
exp_res2 = exp_res.data.copy()
exp_res3 = xp.ExpStats.from_DataFrame(exp_res2)
exp_res4 = deepcopy(exp_res2)
exp_res4.rename(columns={'exposure': 'expo'}, inplace=True)
exp_res5 = deepcopy(exp_res4)
exp_res5.rename(columns={'claims': 'clms'}, inplace=True)


class TestFromDataFrame():

    def test_missing_column_error(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            xp.ExpStats.from_DataFrame(pd.DataFrame({'a': range(3)}))

    def test_class(self):
        assert isinstance(exp_res3, xp.ExpStats)

    def test_before_rename(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            xp.ExpStats.from_DataFrame(exp_res4)

    def test_rename_1(self):
        assert isinstance(
            xp.ExpStats.from_DataFrame(exp_res4,
                                       col_exposure='expo'),
            xp.ExpStats)

    def test_rename_2(self):
        assert isinstance(
            xp.ExpStats.from_DataFrame(exp_res5,
                                       col_exposure='expo',
                                       col_claims='clms'),
            xp.ExpStats)

    def test_non_data_frame(self):
        with pytest.raises(AssertionError,
                           match='must be a Pandas DataFrame'):
            xp.ExpStats.from_DataFrame(1)


# Test that from_DataFrame works with weights
exp_res_weighted2 = exp_res_weighted.data.copy().rename(
    columns={'weight': 'premium'})
exp_res_weighted3 = xp.ExpStats.from_DataFrame(
    exp_res_weighted2, wt="premium",
    expected=["expected_1", "expected_2"],
    conf_int=True, credibility=True)
exp_res_weighted4 = deepcopy(exp_res_weighted2).rename(
    columns={'exposure': 'expo'})
exp_res_weighted5 = deepcopy(exp_res_weighted4).rename(
    columns={'claims': 'clms',
             'n_claims': 'n',
             'weight_sq': 'sq'})

class TestFromDataFrameWeighted():
    
    def test_class(self):
        assert isinstance(exp_res_weighted3, xp.ExpStats)
        
    def test_before_rename(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            xp.ExpStats.from_DataFrame(exp_res_weighted5, wt='premium')
        
    def test_rename_1(self):
        assert isinstance(
            xp.ExpStats.from_DataFrame(exp_res_weighted4,
                                       wt='premium',
                                       col_exposure='expo'),
            xp.ExpStats)        
    
    def test_rename_2(self):
        assert isinstance(
            xp.ExpStats.from_DataFrame(exp_res_weighted5,
                                       wt='premium',
                                       col_exposure='expo',
                                       col_claims='clms',
                                       col_weight_sq='sq',
                                       col_n_claims='n'),
            xp.ExpStats)        

# Test consistency of from_DataFrame summaries and summaries created by 
# exp_stats
class TestSummaryConsistency():
    
    def test_unweighted(self):
        x = exp_res.summary('inc_guar').data.drop(columns='inc_guar')
        y = exp_res3.summary('inc_guar').data.drop(columns='inc_guar')
        assert all(x - y == 0)

    def test_weighted(self):
        x = exp_res_weighted.summary('inc_guar').data.drop(columns='inc_guar')
        y = exp_res_weighted3.summary('inc_guar').data.drop(columns='inc_guar')
        assert all(x - y == 0)