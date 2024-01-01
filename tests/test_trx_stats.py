from actxps.expose import ExposedDF
import actxps as xp
import numpy as np
import pytest
from copy import deepcopy

census_dat = xp.load_census_dat()
withdrawals = xp.load_withdrawals()
account_vals = xp.load_account_vals()

no_trx = ExposedDF.expose_py(census_dat, "2019-12-31",
                             target_status="Surrender")
expo = (ExposedDF.expose_py(census_dat, "2019-12-31",
                            target_status="Surrender").
        add_transactions(withdrawals))
expo.data = expo.data.merge(account_vals, how='left',
                            on=["pol_num", "pol_date_yr"])

res = (expo.
       groupby('pol_yr', 'inc_guar').
       trx_stats(percent_of=["av_anniv", "premium"]))
# results with non-zero transactions only
dat_nz = res.data[res.data.trx_n > 0]


class TestErrorChecks():

    def test_int_error(self):
        with pytest.raises(AssertionError, match="No transactions have been attached"):
            no_trx.trx_stats()

    def test_bad_trx_types(self):
        with pytest.raises(AssertionError, match="The following transactions do not exist"):
            expo.trx_stats(trx_types=['abc', 'def'])

    def test_no_error(self):
        assert isinstance(expo.trx_stats(), xp.TrxStats)


class TestTrxSummaryMethod():

    def test_regroup(self):
        assert all(res.data == res.summary('pol_yr', 'inc_guar').data)

    def test_nogrroup(self):
        assert all(expo.ungroup().trx_stats(
            percent_of=['av_anniv', 'premium']).data ==
            res.summary())

expo2 = deepcopy(expo)
expo2.data = expo2.data.head(6).rename(columns={'exposure': 'ex'})

class TestTrxRenaming():
    
    def test_bad_name(self):
        with pytest.raises(AttributeError):
            expo2.trx_stats()
            
    def test_rename(self):
        assert isinstance(expo2.trx_stats(col_exposure='ex'), xp.TrxStats)

class TestTrxStats():
    
    def test_shape(self):
        assert (expo.
                ungroup().
                trx_stats(trx_types = "Base", 
                          percent_of = ["av_anniv", "premium"]).
                data.shape[1] == res.data.shape[1] - 2)
        
    def test_expo_gte_trx_flag(self):
        assert all(res.data.exposure >= res.data.trx_flag)
        
    def test_avg_trx_gte_avg_all(self):
        assert all(dat_nz.avg_trx >= dat_nz.avg_all)
        
    def test_trx_freq_gte_trx_util(self):
        assert all(dat_nz.trx_freq >= dat_nz.trx_util)
    
    def test_pct_trx_gte_pct_all(self):
        assert all(dat_nz.pct_of_av_anniv_w_trx >= 
                   dat_nz.pct_of_av_anniv_all)
        
    def test_part_expo_lte_full_expo(self):
        assert all(res.data.exposure <=
                   (expo.groupby('pol_yr', 'inc_guar').
                    trx_stats(full_exposures_only=False).data.exposure))
        
    def test_combine_single_trx(self):
        assert all(expo.
                   trx_stats(combine_trx=True, trx_types="Rider").
                   data.drop(columns='trx_type') ==
                   expo.
                   trx_stats(trx_types="Rider").
                   data.drop(columns='trx_type'))

    def test_combine_trx(self):
        assert all(expo.
                   ungroup().
                   trx_stats(combine_trx=True).data ==
                   no_trx.add_transactions(
                       withdrawals.assign(trx_type = 'All')).
                   trx_stats().data)