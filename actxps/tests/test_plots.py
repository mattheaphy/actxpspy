from actxps.expose import ExposedDF
import actxps as xp
import numpy as np
from plotnine.ggplot import ggplot, aes

census_dat = xp.load_census_dat()

expo = ExposedDF.expose_py(census_dat, "2019-12-31",
                           target_status="Surrender")
expo.data["q_exp"] = np.where(expo.data.inc_guar, 0.015, 0.03)


def exp_stats2(obj):
    return obj.exp_stats(wt="premium",
                         credibility=True, expected="q_exp")


# ungrouped summaries
exp_res = exp_stats2(expo)

# 1 grouping variable
expo.groupby('pol_yr')
exp_res2 = exp_stats2(expo)

# 2 grouping variables
expo.groupby('pol_yr', 'inc_guar')
exp_res3 = exp_stats2(expo)

# 3 grouping variables
expo.groupby('pol_yr', 'inc_guar', 'product')
exp_res4 = exp_stats2(expo)

toy_census = xp.load_toy_census()
toy_res = (ExposedDF.expose_py(toy_census, end_date="2022-12-31",
                               target_status="Surrender").
           groupby("pol_yr").
           exp_stats())


class TestPlotWorks():

    def test_exp_1(self):
        assert isinstance(exp_res.plot(), ggplot)

    def test_exp_2(self):
        assert isinstance(exp_res2.plot(), ggplot)

    def test_exp_3(self):
        assert isinstance(exp_res3.plot(), ggplot)

    def test_exp_4(self):
        assert isinstance(exp_res4.plot(), ggplot)


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
