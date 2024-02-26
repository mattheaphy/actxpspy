from actxps.expose import ExposedDF
import actxps as xp
import polars as pl
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
expo.data = expo.data.join(account_vals, how='left',
                           on=["pol_num", "pol_date_yr"])

res = (expo.
       group_by('pol_yr', 'inc_guar').
       trx_stats(percent_of=["av_anniv", "premium"],
                 conf_int=True))
# results with non-zero transactions only
dat_nz = res.data.filter(pl.col('trx_n') > 0)


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
        a = res.data
        b = res.summary('pol_yr', 'inc_guar').data
        assert a.equals(b)

    def test_nogrroup(self):
        a = expo.ungroup().trx_stats(percent_of=['av_anniv', 'premium'],
                                     conf_int=True).data.select(
                                         pl.all().exclude('trx_type'))
        b = res.summary().data.select(pl.all().exclude('trx_type'))
        assert a.equals(b)


expo2 = deepcopy(expo)
expo2.data = expo2.data.head(6).rename({'exposure': 'ex'})


class TestTrxRenaming():

    def test_bad_name(self):
        with pytest.raises(pl.ColumnNotFoundError):
            expo2.trx_stats()

    def test_rename(self):
        assert isinstance(expo2.trx_stats(col_exposure='ex'), xp.TrxStats)


class TestTrxStats():

    def test_shape(self):
        assert (expo.
                ungroup().
                trx_stats(trx_types="Base",
                          percent_of=["av_anniv", "premium"],
                          conf_int=True).
                data.shape[1] == res.data.shape[1] - 2)

    def test_expo_gte_trx_flag(self):
        assert all(res.data['exposure'] >= res.data['trx_flag'])

    def test_avg_trx_gte_avg_all(self):
        assert all(dat_nz['avg_trx'] >= dat_nz['avg_all'])

    def test_trx_freq_gte_trx_util(self):
        assert all(dat_nz['trx_freq'] >= dat_nz['trx_util'])

    def test_pct_trx_gte_pct_all(self):
        assert all(dat_nz['pct_of_av_anniv_w_trx'] >=
                   dat_nz['pct_of_av_anniv_all'])

    def test_part_expo_lte_full_expo(self):
        assert all(res.data['exposure'] <=
                   (expo.group_by('pol_yr', 'inc_guar').
                    trx_stats(full_exposures_only=False).data['exposure']))

    def test_combine_single_trx(self):
        a = (expo.
             trx_stats(combine_trx=True, trx_types="Rider").
             data.drop('trx_type'))
        b = (expo.
             trx_stats(trx_types="Rider").
             data.drop('trx_type'))
        assert a.equals(b)

    def test_combine_trx(self):
        a = (expo.
             ungroup().
             trx_stats(combine_trx=True).data)
        b = (no_trx.add_transactions(
             withdrawals.with_columns(trx_type=pl.lit('All'))).
             trx_stats().data)
        assert a.equals(b)


# Confidence interval tests
res2 = deepcopy(res)
res2.data = res2.data.filter(pl.col('trx_util') > 0)
less_confident = (expo.
                  group_by('pol_yr', 'inc_guar').
                  trx_stats(percent_of=["av_anniv", "premium"],
                            conf_int=True, conf_level=0.5))
less_confident.data = less_confident.data.filter(pl.col('trx_util') > 0)


class TestTrxCI():

    def test_ci_upper(self):
        assert all(res2.data['trx_util'] < res2.data['trx_util_upper'])

    def test_ci_lower(self):
        assert all(res2.data['trx_util'] > res2.data['trx_util_lower'])

    def test_ci_upper_pct_w_trx(self):
        assert all(res2.data['pct_of_premium_w_trx'] <
                   res2.data['pct_of_premium_w_trx_upper'])

    def test_ci_lower_pct_w_trx(self):
        assert all(res2.data['pct_of_premium_w_trx'] >
                   res2.data['pct_of_premium_w_trx_lower'])

    def test_ci_upper_pct_all(self):
        assert all(res2.data['pct_of_premium_all'] <
                   res2.data['pct_of_premium_all_upper'])

    def test_ci_lower_pct_all(self):
        assert all(res2.data['pct_of_premium_all'] >
                   res2.data['pct_of_premium_all_lower'])

    # verify that confidence intervals are tighter using lower confidence
    def test_lower_confidence(self):
        assert all(res2.data['trx_util_upper'] - res2.data['trx_util_lower'] >
                   less_confident.data['trx_util_upper'] -
                   less_confident.data['trx_util_lower'])


# Test that from_DataFrame works
trx_res2 = res.data.clone()
trx_res3 = xp.TrxStats.from_DataFrame(trx_res2, col_percent_of='av_anniv',
                                      conf_int=True)
trx_res4 = trx_res2.clone().rename({'exposure': 'expo'})
trx_res5 = trx_res4.clone().rename({'trx_amt': 'tamt', 'trx_n': 'tn'})


class TestFromDataFrame():

    def test_missing_column_error(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            xp.TrxStats.from_DataFrame(pl.DataFrame({'a': range(3)}))

    def test_class(self):
        assert isinstance(trx_res3, xp.TrxStats)

    def test_before_rename(self):
        with pytest.raises(AssertionError,
                           match='The following columns are missing'):
            xp.TrxStats.from_DataFrame(trx_res4)

    def test_rename_1(self):
        assert isinstance(
            xp.TrxStats.from_DataFrame(trx_res4, col_exposure='expo'),
            xp.TrxStats)

    def test_rename_2(self):
        assert isinstance(
            xp.TrxStats.from_DataFrame(trx_res5, col_exposure='expo',
                                       col_trx_amt='tamt',
                                       col_trx_n='tn'),
            xp.TrxStats)

    def test_non_data_frame(self):
        with pytest.raises(AssertionError,
                           match='must be a DataFrame'):
            xp.TrxStats.from_DataFrame(1)

    def test_pandas(self):
        assert isinstance(
            xp.TrxStats.from_DataFrame(trx_res2.to_pandas(),
                                       col_percent_of='av_anniv',
                                       conf_int=True),
            xp.TrxStats
        )


# Test consistency of from_DataFrame summaries and summaries created by
# trx_stats
class TestSummaryConsistency():

    def test_unweighted(self):
        base = (expo.
                group_by('pol_yr', 'inc_guar').
                trx_stats(percent_of="av_anniv",
                          conf_int=True, trx_types="Base"))
        x = base.summary('inc_guar').data.drop(['inc_guar', 'trx_type'])

        base2 = xp.TrxStats.from_DataFrame(base.data,
                                           col_percent_of="av_anniv",
                                           conf_int=True)

        y = base2.summary('inc_guar').data.drop(['inc_guar', 'trx_type'])
        assert x.equals(y)
