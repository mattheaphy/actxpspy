from actxps.expose import ExposedDF
import actxps as xp
import polars as pl
from plotnine.ggplot import ggplot, aes
import pytest

census_dat = xp.load_census_dat()
withdrawals = xp.load_withdrawals()

expo = ExposedDF.expose_py(census_dat, "2019-12-31",
                           target_status="Surrender")
expo.add_transactions(withdrawals)
expo.data = expo.data.with_columns(
    q_exp=pl.when(pl.col('inc_guar')).then(0.015).otherwise(0.03)
)


def exp_stats2(obj):
    return obj.exp_stats(wt="premium", conf_int=True,
                         credibility=True, expected="q_exp")


def trx_stats2(obj):
    return obj.trx_stats(conf_int=True, percent_of='premium')


# ungrouped summaries
exp_res = exp_stats2(expo)
trx_res = trx_stats2(expo)

# 1 grouping variable
expo.group_by('pol_yr')
exp_res2 = exp_stats2(expo)
trx_res2 = trx_stats2(expo)

# 2 grouping variables
expo.group_by('pol_yr', 'inc_guar')
exp_res3 = exp_stats2(expo)
trx_res3 = trx_stats2(expo)

# 3 grouping variables
expo.group_by('pol_yr', 'inc_guar', 'product')
exp_res4 = exp_stats2(expo)
trx_res4 = trx_stats2(expo)


# Plot methods work
class TestPlotWorks():

    def test_exp_1(self):
        assert isinstance(exp_res.plot(), ggplot)

    def test_exp_2(self):
        assert isinstance(exp_res2.plot(), ggplot)

    def test_exp_3(self):
        assert isinstance(exp_res3.plot(), ggplot)

    def test_exp_4(self):
        assert isinstance(exp_res4.plot(), ggplot)

    def test_trx_1(self):
        assert isinstance(trx_res.plot(), ggplot)

    def test_trx_2(self):
        assert isinstance(trx_res2.plot(), ggplot)

    def test_trx_3(self):
        assert isinstance(trx_res3.plot(), ggplot)

    def test_trx_4(self):
        assert isinstance(trx_res4.plot(), ggplot)


# Plot methods work with mapping overrides
class TestPlotOverrides():

    def test_aes_override_1(self):
        p = exp_res4.plot(x='pol_yr',
                          y='ae_q_exp',
                          color='product',
                          facets='inc_guar',
                          scales="free_y",
                          geoms="bars",
                          y_labels=lambda l: [f"{v:,.0f}" for v in l])
        assert isinstance(p, ggplot)

    def test_aes_override_2(self):
        p = exp_res4.plot(mapping=aes(x="pol_yr",
                                      y="ae_q_exp",
                                      fill="product"),
                          facets='inc_guar',
                          scales="free_y",
                          geoms="bars",
                          y_labels=lambda l: [f"{v:,.0f}" for v in l])
        assert isinstance(p, ggplot)

    def test_aes_override_3(self):
        p = trx_res4.plot(x='pol_yr',
                          y='pct_of_premium_w_trx',
                          color='product',
                          facets='inc_guar',
                          scales="free_y",
                          geoms="bars",
                          y_labels=lambda l: [f"{v:,.2f}" for v in l])
        assert isinstance(p, ggplot)

    def test_aes_override_4(self):
        p = trx_res4.plot(mapping=aes(x='pol_yr',
                                      y='pct_of_premium_w_trx',
                                      fill='product'),
                          facets='inc_guar',
                          scales="free_y",
                          geoms="bars",
                          y_labels=lambda l: [f"{v:,.2f}" for v in l])
        assert isinstance(p, ggplot)

    def test_points_1(self):
        assert isinstance(exp_res4.plot(geoms='points'), ggplot)

    def test_points_2(self):
        assert isinstance(trx_res4.plot(geoms='points'), ggplot)


# Termination plots work
class TestTerminationPlots():

    def test_term_plots_1(self):
        assert isinstance(exp_res.plot_termination_rates(), ggplot)

    def test_term_plots_2(self):
        assert isinstance(exp_res.plot_termination_rates(include_cred_adj=True,
                                                         conf_int_bars=True),
                          ggplot)


# AE plots work
class TestAEPlots():

    def test_ae_plots_1(self):
        assert isinstance(exp_res.plot_actual_to_expected(), ggplot)

    def test_ae_plots_2(self):
        assert isinstance(exp_res.plot_actual_to_expected(conf_int_bars=True),
                          ggplot)

    def test_ae_plot_error_no_expected(self):
        with pytest.raises(AssertionError,
                           match="does not have any actual-to-expected"):
            expo.exp_stats().plot_actual_to_expected()


# Transaction utilization plots work"
class TestUtilPlots():

    def test_util_plots_1(self):
        assert isinstance(trx_res.plot_utilization_rates(), ggplot)

    def test_util_plots_2(self):
        assert isinstance(trx_res2.plot_utilization_rates(), ggplot)

    def test_util_plots_3(self):
        assert isinstance(trx_res3.plot_utilization_rates(), ggplot)

    def test_util_plots_4(self):
        assert isinstance(trx_res4.plot_utilization_rates(), ggplot)


# Log y scale works

class TestLogYScale():

    def test_exp_logy(self):
        assert isinstance(exp_res4.plot(y_log10=True), ggplot)

    def test_trx_logy(self):
        assert isinstance(trx_res4.plot(y_log10=True), ggplot)


no_ci = (expo.
         group_by('pol_yr', 'inc_guar', 'product').
         exp_stats(expected="q_exp"))
no_ci_trx = (expo.
             group_by('pol_yr', 'inc_guar', 'product').
             trx_stats(percent_of="premium"))


# Confidence interval warning messages work for termination plots
class TestPlotCIWarningExp():

    def test_no_warning(self):
        assert isinstance(exp_res4.plot(conf_int_bars=True), ggplot)

    def test_warning_1(self):
        with pytest.warns(match='has no confidence intervals'):
            no_ci.plot(conf_int_bars=True)

    def test_warning_2(self):
        with pytest.warns(match='has no confidence intervals'):
            no_ci.plot_termination_rates(conf_int_bars=True)

    def test_warning_3(self):
        with pytest.warns(match='has no confidence intervals'):
            no_ci.plot_actual_to_expected(conf_int_bars=True)

    def test_warning_4(self):
        with pytest.warns(match='Confidence intervals are not available'):
            exp_res4.plot(conf_int_bars=True, y='exposure')


# Confidence interval warning messages work for transaction plots
class TestPlotCIWarningTrx():

    def test_no_warning(self):
        assert isinstance(trx_res4.plot(conf_int_bars=True), ggplot)

    def test_warning_1(self):
        with pytest.warns(match='has no confidence intervals'):
            no_ci_trx.plot(conf_int_bars=True)

    def test_warning_2(self):
        with pytest.warns(match='has no confidence intervals'):
            no_ci_trx.plot_utilization_rates(conf_int_bars=True)

    def test_warning_3(self):
        with pytest.warns(match='Confidence intervals are not available'):
            trx_res4.plot(conf_int_bars=True, y='exposure')


no_expected = expo.group_by('pol_yr').exp_stats(credibility=True)


# .plot_termination_rates() credibility-adjusted message works
class TestPlotCredAdjWarning():

    def test_warning_1(self):
        with pytest.warns(match='has no credibility-weighted'):
            no_ci.plot_termination_rates(include_cred_adj=True)

    def test_warning_2(self):
        with pytest.warns(match='has no credibility-weighted'):
            no_expected.plot_termination_rates(include_cred_adj=True)
