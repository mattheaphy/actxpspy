
<!-- README.md is generated from README.Rmd. Please edit that file -->

# actxps <a href="https://github.com/mattheaphy/actxpspy/"><img src="doc/images/logo.png" align="right" height="138" /></a>

The actxps package provides a set of tools to assist with the creation
of actuarial experience studies. Experience studies are used by
actuaries to explore historical experience across blocks of business and
to inform assumption setting for projection models.

-   The `ExposedDF` class converts census-level records into policy or
    calendar year exposure records.
-   The `exp_stats()` method of `ExposedDF` creates `ExpStats`
    experience summary objects containing observed termination rates and
    claims. Optionally, expected termination rates, actual-to-expected
    ratios, and limited fluctuation credibility estimates can also be
    returned.
-   The `add_transactions()` method attaches summarized transactions to
    an `ExposedDF` object.
-   The `trx_stats()` method of `ExposedDF` creates `TrxStats`
    transaction summary objects containing transaction counts, amounts,
    frequencies, and utilization. Optionally, transaction amounts can be
    expressed as a percentage of one or more variables to calculate
    rates or actual-to-expected ratios.
-   The `plot()` and `table()` methods of `ExpStats` and `TrxStats`
    create plots and tables for reporting.

## Basic usage

The actxps package includes simulated census data for a theoretical
deferred annuity product with an optional guaranteed income rider. The
grain of this data is one row *per policy*.

``` python
import actxps as xp
import numpy as np

census_dat = xp.load_census_dat()
census_dat
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pol_num</th>
      <th>status</th>
      <th>issue_date</th>
      <th>inc_guar</th>
      <th>qual</th>
      <th>age</th>
      <th>product</th>
      <th>gender</th>
      <th>wd_age</th>
      <th>premium</th>
      <th>term_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Active</td>
      <td>2014-12-17</td>
      <td>True</td>
      <td>False</td>
      <td>56</td>
      <td>b</td>
      <td>F</td>
      <td>77</td>
      <td>370.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Surrender</td>
      <td>2007-09-24</td>
      <td>False</td>
      <td>False</td>
      <td>71</td>
      <td>a</td>
      <td>F</td>
      <td>71</td>
      <td>708.0</td>
      <td>2019-03-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Active</td>
      <td>2012-10-06</td>
      <td>False</td>
      <td>True</td>
      <td>62</td>
      <td>b</td>
      <td>F</td>
      <td>63</td>
      <td>466.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Surrender</td>
      <td>2005-06-27</td>
      <td>True</td>
      <td>True</td>
      <td>62</td>
      <td>c</td>
      <td>M</td>
      <td>62</td>
      <td>485.0</td>
      <td>2018-11-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Active</td>
      <td>2019-11-22</td>
      <td>False</td>
      <td>False</td>
      <td>62</td>
      <td>c</td>
      <td>F</td>
      <td>67</td>
      <td>978.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>19996</td>
      <td>Active</td>
      <td>2014-08-11</td>
      <td>True</td>
      <td>True</td>
      <td>55</td>
      <td>b</td>
      <td>F</td>
      <td>75</td>
      <td>3551.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>19997</td>
      <td>Surrender</td>
      <td>2006-11-20</td>
      <td>False</td>
      <td>False</td>
      <td>68</td>
      <td>c</td>
      <td>F</td>
      <td>77</td>
      <td>336.0</td>
      <td>2017-07-09</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>19998</td>
      <td>Surrender</td>
      <td>2017-02-20</td>
      <td>True</td>
      <td>False</td>
      <td>68</td>
      <td>c</td>
      <td>F</td>
      <td>68</td>
      <td>1222.0</td>
      <td>2018-08-03</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>19999</td>
      <td>Active</td>
      <td>2015-04-11</td>
      <td>False</td>
      <td>True</td>
      <td>67</td>
      <td>a</td>
      <td>M</td>
      <td>78</td>
      <td>2138.0</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>20000</td>
      <td>Active</td>
      <td>2009-04-29</td>
      <td>True</td>
      <td>True</td>
      <td>72</td>
      <td>c</td>
      <td>M</td>
      <td>72</td>
      <td>5751.0</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
<p>20000 rows × 11 columns</p>
</div>

Convert census records to exposure records with one row *per policy per
year*.

``` python
exposed_data = xp.ExposedDF(census_dat,
                            end_date="2019-12-31",
                            target_status="Surrender")

exposed_data
```

    Exposure data

    Exposure type: policy_year
    Target status: Surrender
    Study range: 1900-01-01 to 2019-12-31

    A DataFrame: 141,252 x 15
       pol_num  status issue_date  inc_guar   qual  age product gender  wd_age  \
    0        1  Active 2014-12-17      True  False   56       b      F      77   
    1        1  Active 2014-12-17      True  False   56       b      F      77   
    2        1  Active 2014-12-17      True  False   56       b      F      77   
    3        1  Active 2014-12-17      True  False   56       b      F      77   
    4        1  Active 2014-12-17      True  False   56       b      F      77   
    5        1  Active 2014-12-17      True  False   56       b      F      77   
    6        2  Active 2007-09-24     False  False   71       a      F      71   
    7        2  Active 2007-09-24     False  False   71       a      F      71   
    8        2  Active 2007-09-24     False  False   71       a      F      71   
    9        2  Active 2007-09-24     False  False   71       a      F      71   

       premium term_date  pol_yr pol_date_yr pol_date_yr_end  exposure  
    0    370.0       NaT       1  2014-12-17      2015-12-16  1.000000  
    1    370.0       NaT       2  2015-12-17      2016-12-16  1.000000  
    2    370.0       NaT       3  2016-12-17      2017-12-16  1.000000  
    3    370.0       NaT       4  2017-12-17      2018-12-16  1.000000  
    4    370.0       NaT       5  2018-12-17      2019-12-16  1.000000  
    5    370.0       NaT       6  2019-12-17      2020-12-16  0.040984  
    6    708.0       NaT       1  2007-09-24      2008-09-23  1.000000  
    7    708.0       NaT       2  2008-09-24      2009-09-23  1.000000  
    8    708.0       NaT       3  2009-09-24      2010-09-23  1.000000  
    9    708.0       NaT       4  2010-09-24      2011-09-23  1.000000  

Create a summary grouped by policy year and the presence of a guaranteed
income rider.

``` python
exp_res = (exposed_data.
           groupby('pol_yr', 'inc_guar').
           exp_stats())

exp_res
```

    Experience study results

    Groups: pol_yr, inc_guar
    Target status: Surrender
    Study range: 1900-01-01 to 2019-12-31


    A DataFrame: 30 x 6
       pol_yr  inc_guar  n_claims  claims      exposure     q_obs
    0       1     False        56      56   7719.807740  0.007254
    1       1      True        46      46  11532.404626  0.003989
    2       2     False        92      92   7102.813160  0.012953
    3       2      True        68      68  10611.967258  0.006408
    4       3     False        67      67   6446.916146  0.010393
    5       3      True        57      57   9650.221229  0.005907
    6       4     False       123     123   5798.909986  0.021211
    7       4      True        45      45   8736.954420  0.005151
    8       5     False        97      97   5105.875799  0.018998
    9       5      True        67      67   7809.650445  0.008579

Calculate actual-to-expected ratios.

First, attach one or more columns of expected termination rates to the
exposure data. Then, pass these column names to the `expected` argument
of `exp_stats()`.

``` python
expected_table = np.concatenate((
    np.linspace(0.005, 0.03, 10), 
    np.array([0.2, 0.15]), 
    np.repeat(0.05, 3)
    ))

# using 2 different expected termination rates
exposed_data.data['expected_1'] = \
    expected_table[exposed_data.data.pol_yr - 1]
exposed_data.data['expected_2'] = \
    np.where(exposed_data.data.inc_guar, 0.015, 0.03)

exp_res = (exposed_data.
           groupby('pol_yr', 'inc_guar').
           exp_stats(expected = ["expected_1", "expected_2"]))

exp_res
```

    Experience study results

    Groups: pol_yr, inc_guar
    Target status: Surrender
    Study range: 1900-01-01 to 2019-12-31
    Expected values: expected_1, expected_2


    A DataFrame: 30 x 10
       pol_yr  inc_guar  n_claims  claims      exposure     q_obs  expected_1  \
    0       1     False        56      56   7719.807740  0.007254    0.005000   
    1       1      True        46      46  11532.404626  0.003989    0.005000   
    2       2     False        92      92   7102.813160  0.012953    0.007778   
    3       2      True        68      68  10611.967258  0.006408    0.007778   
    4       3     False        67      67   6446.916146  0.010393    0.010556   
    5       3      True        57      57   9650.221229  0.005907    0.010556   
    6       4     False       123     123   5798.909986  0.021211    0.013333   
    7       4      True        45      45   8736.954420  0.005151    0.013333   
    8       5     False        97      97   5105.875799  0.018998    0.016111   
    9       5      True        67      67   7809.650445  0.008579    0.016111   

       expected_2  ae_expected_1  ae_expected_2  
    0       0.030       1.450813       0.241802  
    1       0.015       0.797752       0.265917  
    2       0.030       1.665336       0.431754  
    3       0.015       0.823868       0.427191  
    4       0.030       0.984559       0.346419  
    5       0.015       0.559573       0.393773  
    6       0.030       1.590816       0.707029  
    7       0.015       0.386290       0.343369  
    8       0.030       1.179169       0.633257  
    9       0.015       0.532498       0.571942  

Create visualizations using the `plot()` and `table()` methods.

``` python
from plotnine import ggplot, scale_color_manual, labs
from plotnine.themes import theme_light

colors = ["#eb15e4", "#7515eb"]
# theme_set(theme_light())

(exp_res.plot() +
  scale_color_manual(values = colors) + 
  labs(title = "Observed Surrender Rates by Policy Year and Income Guarantee Presence"))
```

<img src="README_files\figure-gfm/plots-output-1.png" id="plots-1" />

    <ggplot: (123366203852)>

``` python
exp_res.table()
```

<style type="text/css">
#T_1b374 th {
  font-weight: bold;
  font-size: 100%;
}
#T_1b374 tr {
  font-size: 100%;
}
#T_1b374 caption {
  font-size: 100%;
}
#T_1b374 th.col_heading {
  text-align: center;
}
#T_1b374 th.col_heading.level0 {
  font-size: 1.1em;
}
#T_1b374_row0_col2, #T_1b374_row11_col2 {
  background-color: #f5fbee;
  color: #000000;
}
#T_1b374_row0_col4 {
  background-color: #f7f6f6;
  color: #000000;
}
#T_1b374_row0_col6, #T_1b374_row1_col6, #T_1b374_row7_col4 {
  background-color: #053061;
  color: #f1f1f1;
}
#T_1b374_row1_col2 {
  background-color: #f7fcf0;
  color: #000000;
}
#T_1b374_row1_col4 {
  background-color: #408fc1;
  color: #f1f1f1;
}
#T_1b374_row2_col2 {
  background-color: #f1faeb;
  color: #000000;
}
#T_1b374_row2_col4 {
  background-color: #fddbc7;
  color: #000000;
}
#T_1b374_row2_col6, #T_1b374_row3_col6, #T_1b374_row11_col4 {
  background-color: #0a3b70;
  color: #f1f1f1;
}
#T_1b374_row3_col2 {
  background-color: #f6fbef;
  color: #000000;
}
#T_1b374_row3_col4 {
  background-color: #4695c4;
  color: #f1f1f1;
}
#T_1b374_row4_col2 {
  background-color: #f3faec;
  color: #000000;
}
#T_1b374_row4_col4 {
  background-color: #84bcd9;
  color: #000000;
}
#T_1b374_row4_col6 {
  background-color: #08366a;
  color: #f1f1f1;
}
#T_1b374_row5_col2, #T_1b374_row7_col2 {
  background-color: #f6fcef;
  color: #000000;
}
#T_1b374_row5_col4, #T_1b374_row18_col6 {
  background-color: #1b5a9c;
  color: #f1f1f1;
}
#T_1b374_row5_col6 {
  background-color: #09386d;
  color: #f1f1f1;
}
#T_1b374_row6_col2, #T_1b374_row10_col2, #T_1b374_row17_col2 {
  background-color: #ebf7e5;
  color: #000000;
}
#T_1b374_row6_col4 {
  background-color: #fbe5d8;
  color: #000000;
}
#T_1b374_row6_col6, #T_1b374_row10_col6 {
  background-color: #134c87;
  color: #f1f1f1;
}
#T_1b374_row7_col6 {
  background-color: #073467;
  color: #f1f1f1;
}
#T_1b374_row8_col2 {
  background-color: #edf8e7;
  color: #000000;
}
#T_1b374_row8_col4 {
  background-color: #c0dceb;
  color: #000000;
}
#T_1b374_row8_col6 {
  background-color: #114781;
  color: #f1f1f1;
}
#T_1b374_row9_col2 {
  background-color: #f4fbed;
  color: #000000;
}
#T_1b374_row9_col4 {
  background-color: #185493;
  color: #f1f1f1;
}
#T_1b374_row9_col6 {
  background-color: #0f437b;
  color: #f1f1f1;
}
#T_1b374_row10_col4, #T_1b374_row24_col6 {
  background-color: #b1d5e7;
  color: #000000;
}
#T_1b374_row11_col6 {
  background-color: #0e4179;
  color: #f1f1f1;
}
#T_1b374_row12_col2 {
  background-color: #eaf7e4;
  color: #000000;
}
#T_1b374_row12_col4 {
  background-color: #a5cee3;
  color: #000000;
}
#T_1b374_row12_col6 {
  background-color: #15508d;
  color: #f1f1f1;
}
#T_1b374_row13_col2 {
  background-color: #f2faeb;
  color: #000000;
}
#T_1b374_row13_col4 {
  background-color: #1a5899;
  color: #f1f1f1;
}
#T_1b374_row13_col6 {
  background-color: #175290;
  color: #f1f1f1;
}
#T_1b374_row14_col2 {
  background-color: #e4f5df;
  color: #000000;
}
#T_1b374_row14_col4 {
  background-color: #d8e9f1;
  color: #000000;
}
#T_1b374_row14_col6 {
  background-color: #1e61a5;
  color: #f1f1f1;
}
#T_1b374_row15_col2 {
  background-color: #eef9e8;
  color: #000000;
}
#T_1b374_row15_col4 {
  background-color: #2f79b5;
  color: #f1f1f1;
}
#T_1b374_row15_col6 {
  background-color: #2065ab;
  color: #f1f1f1;
}
#T_1b374_row16_col2 {
  background-color: #e4f4de;
  color: #000000;
}
#T_1b374_row16_col4 {
  background-color: #c2ddec;
  color: #000000;
}
#T_1b374_row16_col6 {
  background-color: #1f63a8;
  color: #f1f1f1;
}
#T_1b374_row17_col4 {
  background-color: #4291c2;
  color: #f1f1f1;
}
#T_1b374_row17_col6 {
  background-color: #2e77b5;
  color: #f1f1f1;
}
#T_1b374_row18_col2 {
  background-color: #e7f6e2;
  color: #000000;
}
#T_1b374_row18_col4, #T_1b374_row29_col4 {
  background-color: #6eaed2;
  color: #f1f1f1;
}
#T_1b374_row19_col2 {
  background-color: #e8f6e2;
  color: #000000;
}
#T_1b374_row19_col4 {
  background-color: #5fa5cd;
  color: #f1f1f1;
}
#T_1b374_row19_col6 {
  background-color: #3a87bd;
  color: #f1f1f1;
}
#T_1b374_row20_col2 {
  background-color: #084081;
  color: #f1f1f1;
}
#T_1b374_row20_col4, #T_1b374_row26_col6 {
  background-color: #e4eef4;
  color: #000000;
}
#T_1b374_row20_col6, #T_1b374_row26_col4 {
  background-color: #67001f;
  color: #f1f1f1;
}
#T_1b374_row21_col2 {
  background-color: #86d0c0;
  color: #000000;
}
#T_1b374_row21_col4 {
  background-color: #276eb0;
  color: #f1f1f1;
}
#T_1b374_row21_col6 {
  background-color: #8a0b25;
  color: #f1f1f1;
}
#T_1b374_row22_col2 {
  background-color: #2d8fbf;
  color: #f1f1f1;
}
#T_1b374_row22_col4 {
  background-color: #e3edf3;
  color: #000000;
}
#T_1b374_row22_col6 {
  background-color: #e98b6e;
  color: #f1f1f1;
}
#T_1b374_row23_col2 {
  background-color: #bde5be;
  color: #000000;
}
#T_1b374_row23_col4 {
  background-color: #1c5c9f;
  color: #f1f1f1;
}
#T_1b374_row23_col6 {
  background-color: #fbd0b9;
  color: #000000;
}
#T_1b374_row24_col2 {
  background-color: #addfb7;
  color: #000000;
}
#T_1b374_row24_col4 {
  background-color: #e48066;
  color: #f1f1f1;
}
#T_1b374_row25_col2 {
  background-color: #dcf1d7;
  color: #000000;
}
#T_1b374_row25_col4 {
  background-color: #5ca3cb;
  color: #f1f1f1;
}
#T_1b374_row25_col6 {
  background-color: #96c7df;
  color: #000000;
}
#T_1b374_row26_col2 {
  background-color: #8ad2bf;
  color: #000000;
}
#T_1b374_row27_col2 {
  background-color: #daf1d5;
  color: #000000;
}
#T_1b374_row27_col4 {
  background-color: #78b4d5;
  color: #000000;
}
#T_1b374_row27_col6 {
  background-color: #a7d0e4;
  color: #000000;
}
#T_1b374_row28_col2 {
  background-color: #a2dbb7;
  color: #000000;
}
#T_1b374_row28_col4 {
  background-color: #c94741;
  color: #f1f1f1;
}
#T_1b374_row28_col6 {
  background-color: #c7e0ed;
  color: #000000;
}
#T_1b374_row29_col2 {
  background-color: #dbf1d5;
  color: #000000;
}
#T_1b374_row29_col6 {
  background-color: #a2cde3;
  color: #000000;
}
</style>
<table id="T_1b374">
  <caption><h1>Experience Study Results</h1>Target status: Surrender<br>Study range: 1900-01-01 to 2019-12-31</caption>
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1b374_level0_col0" class="col_heading level0 col0" colspan="3"></th>
      <th id="T_1b374_level0_col3" class="col_heading level0 col3" colspan="2">expected_1</th>
      <th id="T_1b374_level0_col5" class="col_heading level0 col5" colspan="2">expected_2</th>
    </tr>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level1" >&nbsp;</th>
      <th id="T_1b374_level1_col0" class="col_heading level1 col0" >Claims</th>
      <th id="T_1b374_level1_col1" class="col_heading level1 col1" >Exposures</th>
      <th id="T_1b374_level1_col2" class="col_heading level1 col2" ><em>q<sup>obs</sup></em></th>
      <th id="T_1b374_level1_col3" class="col_heading level1 col3" ><em>q<sup>exp</sup></em></th>
      <th id="T_1b374_level1_col4" class="col_heading level1 col4" ><em>A/E</em></th>
      <th id="T_1b374_level1_col5" class="col_heading level1 col5" ><em>q<sup>exp</sup></em></th>
      <th id="T_1b374_level1_col6" class="col_heading level1 col6" ><em>A/E</em></th>
    </tr>
    <tr>
      <th class="index_name level0" >pol_yr</th>
      <th class="index_name level1" >inc_guar</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1b374_level0_row0" class="row_heading level0 row0" rowspan="2">1</th>
      <th id="T_1b374_level1_row0" class="row_heading level1 row0" >False</th>
      <td id="T_1b374_row0_col0" class="data row0 col0" >56</td>
      <td id="T_1b374_row0_col1" class="data row0 col1" >7,720</td>
      <td id="T_1b374_row0_col2" class="data row0 col2" >0.7%</td>
      <td id="T_1b374_row0_col3" class="data row0 col3" >0.5%</td>
      <td id="T_1b374_row0_col4" class="data row0 col4" >145.1%</td>
      <td id="T_1b374_row0_col5" class="data row0 col5" >3.0%</td>
      <td id="T_1b374_row0_col6" class="data row0 col6" >24.2%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row1" class="row_heading level1 row1" >True</th>
      <td id="T_1b374_row1_col0" class="data row1 col0" >46</td>
      <td id="T_1b374_row1_col1" class="data row1 col1" >11,532</td>
      <td id="T_1b374_row1_col2" class="data row1 col2" >0.4%</td>
      <td id="T_1b374_row1_col3" class="data row1 col3" >0.5%</td>
      <td id="T_1b374_row1_col4" class="data row1 col4" >79.8%</td>
      <td id="T_1b374_row1_col5" class="data row1 col5" >1.5%</td>
      <td id="T_1b374_row1_col6" class="data row1 col6" >26.6%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row2" class="row_heading level0 row2" rowspan="2">2</th>
      <th id="T_1b374_level1_row2" class="row_heading level1 row2" >False</th>
      <td id="T_1b374_row2_col0" class="data row2 col0" >92</td>
      <td id="T_1b374_row2_col1" class="data row2 col1" >7,103</td>
      <td id="T_1b374_row2_col2" class="data row2 col2" >1.3%</td>
      <td id="T_1b374_row2_col3" class="data row2 col3" >0.8%</td>
      <td id="T_1b374_row2_col4" class="data row2 col4" >166.5%</td>
      <td id="T_1b374_row2_col5" class="data row2 col5" >3.0%</td>
      <td id="T_1b374_row2_col6" class="data row2 col6" >43.2%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row3" class="row_heading level1 row3" >True</th>
      <td id="T_1b374_row3_col0" class="data row3 col0" >68</td>
      <td id="T_1b374_row3_col1" class="data row3 col1" >10,612</td>
      <td id="T_1b374_row3_col2" class="data row3 col2" >0.6%</td>
      <td id="T_1b374_row3_col3" class="data row3 col3" >0.8%</td>
      <td id="T_1b374_row3_col4" class="data row3 col4" >82.4%</td>
      <td id="T_1b374_row3_col5" class="data row3 col5" >1.5%</td>
      <td id="T_1b374_row3_col6" class="data row3 col6" >42.7%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row4" class="row_heading level0 row4" rowspan="2">3</th>
      <th id="T_1b374_level1_row4" class="row_heading level1 row4" >False</th>
      <td id="T_1b374_row4_col0" class="data row4 col0" >67</td>
      <td id="T_1b374_row4_col1" class="data row4 col1" >6,447</td>
      <td id="T_1b374_row4_col2" class="data row4 col2" >1.0%</td>
      <td id="T_1b374_row4_col3" class="data row4 col3" >1.1%</td>
      <td id="T_1b374_row4_col4" class="data row4 col4" >98.5%</td>
      <td id="T_1b374_row4_col5" class="data row4 col5" >3.0%</td>
      <td id="T_1b374_row4_col6" class="data row4 col6" >34.6%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row5" class="row_heading level1 row5" >True</th>
      <td id="T_1b374_row5_col0" class="data row5 col0" >57</td>
      <td id="T_1b374_row5_col1" class="data row5 col1" >9,650</td>
      <td id="T_1b374_row5_col2" class="data row5 col2" >0.6%</td>
      <td id="T_1b374_row5_col3" class="data row5 col3" >1.1%</td>
      <td id="T_1b374_row5_col4" class="data row5 col4" >56.0%</td>
      <td id="T_1b374_row5_col5" class="data row5 col5" >1.5%</td>
      <td id="T_1b374_row5_col6" class="data row5 col6" >39.4%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row6" class="row_heading level0 row6" rowspan="2">4</th>
      <th id="T_1b374_level1_row6" class="row_heading level1 row6" >False</th>
      <td id="T_1b374_row6_col0" class="data row6 col0" >123</td>
      <td id="T_1b374_row6_col1" class="data row6 col1" >5,799</td>
      <td id="T_1b374_row6_col2" class="data row6 col2" >2.1%</td>
      <td id="T_1b374_row6_col3" class="data row6 col3" >1.3%</td>
      <td id="T_1b374_row6_col4" class="data row6 col4" >159.1%</td>
      <td id="T_1b374_row6_col5" class="data row6 col5" >3.0%</td>
      <td id="T_1b374_row6_col6" class="data row6 col6" >70.7%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row7" class="row_heading level1 row7" >True</th>
      <td id="T_1b374_row7_col0" class="data row7 col0" >45</td>
      <td id="T_1b374_row7_col1" class="data row7 col1" >8,737</td>
      <td id="T_1b374_row7_col2" class="data row7 col2" >0.5%</td>
      <td id="T_1b374_row7_col3" class="data row7 col3" >1.3%</td>
      <td id="T_1b374_row7_col4" class="data row7 col4" >38.6%</td>
      <td id="T_1b374_row7_col5" class="data row7 col5" >1.5%</td>
      <td id="T_1b374_row7_col6" class="data row7 col6" >34.3%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row8" class="row_heading level0 row8" rowspan="2">5</th>
      <th id="T_1b374_level1_row8" class="row_heading level1 row8" >False</th>
      <td id="T_1b374_row8_col0" class="data row8 col0" >97</td>
      <td id="T_1b374_row8_col1" class="data row8 col1" >5,106</td>
      <td id="T_1b374_row8_col2" class="data row8 col2" >1.9%</td>
      <td id="T_1b374_row8_col3" class="data row8 col3" >1.6%</td>
      <td id="T_1b374_row8_col4" class="data row8 col4" >117.9%</td>
      <td id="T_1b374_row8_col5" class="data row8 col5" >3.0%</td>
      <td id="T_1b374_row8_col6" class="data row8 col6" >63.3%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row9" class="row_heading level1 row9" >True</th>
      <td id="T_1b374_row9_col0" class="data row9 col0" >67</td>
      <td id="T_1b374_row9_col1" class="data row9 col1" >7,810</td>
      <td id="T_1b374_row9_col2" class="data row9 col2" >0.9%</td>
      <td id="T_1b374_row9_col3" class="data row9 col3" >1.6%</td>
      <td id="T_1b374_row9_col4" class="data row9 col4" >53.2%</td>
      <td id="T_1b374_row9_col5" class="data row9 col5" >1.5%</td>
      <td id="T_1b374_row9_col6" class="data row9 col6" >57.2%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row10" class="row_heading level0 row10" rowspan="2">6</th>
      <th id="T_1b374_level1_row10" class="row_heading level1 row10" >False</th>
      <td id="T_1b374_row10_col0" class="data row10 col0" >96</td>
      <td id="T_1b374_row10_col1" class="data row10 col1" >4,494</td>
      <td id="T_1b374_row10_col2" class="data row10 col2" >2.1%</td>
      <td id="T_1b374_row10_col3" class="data row10 col3" >1.9%</td>
      <td id="T_1b374_row10_col4" class="data row10 col4" >113.1%</td>
      <td id="T_1b374_row10_col5" class="data row10 col5" >3.0%</td>
      <td id="T_1b374_row10_col6" class="data row10 col6" >71.2%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row11" class="row_heading level1 row11" >True</th>
      <td id="T_1b374_row11_col0" class="data row11 col0" >56</td>
      <td id="T_1b374_row11_col1" class="data row11 col1" >6,882</td>
      <td id="T_1b374_row11_col2" class="data row11 col2" >0.8%</td>
      <td id="T_1b374_row11_col3" class="data row11 col3" >1.9%</td>
      <td id="T_1b374_row11_col4" class="data row11 col4" >43.1%</td>
      <td id="T_1b374_row11_col5" class="data row11 col5" >1.5%</td>
      <td id="T_1b374_row11_col6" class="data row11 col6" >54.2%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row12" class="row_heading level0 row12" rowspan="2">7</th>
      <th id="T_1b374_level1_row12" class="row_heading level1 row12" >False</th>
      <td id="T_1b374_row12_col0" class="data row12 col0" >92</td>
      <td id="T_1b374_row12_col1" class="data row12 col1" >3,899</td>
      <td id="T_1b374_row12_col2" class="data row12 col2" >2.4%</td>
      <td id="T_1b374_row12_col3" class="data row12 col3" >2.2%</td>
      <td id="T_1b374_row12_col4" class="data row12 col4" >108.9%</td>
      <td id="T_1b374_row12_col5" class="data row12 col5" >3.0%</td>
      <td id="T_1b374_row12_col6" class="data row12 col6" >78.7%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row13" class="row_heading level1 row13" >True</th>
      <td id="T_1b374_row13_col0" class="data row13 col0" >72</td>
      <td id="T_1b374_row13_col1" class="data row13 col1" >6,018</td>
      <td id="T_1b374_row13_col2" class="data row13 col2" >1.2%</td>
      <td id="T_1b374_row13_col3" class="data row13 col3" >2.2%</td>
      <td id="T_1b374_row13_col4" class="data row13 col4" >55.2%</td>
      <td id="T_1b374_row13_col5" class="data row13 col5" >1.5%</td>
      <td id="T_1b374_row13_col6" class="data row13 col6" >79.8%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row14" class="row_heading level0 row14" rowspan="2">8</th>
      <th id="T_1b374_level1_row14" class="row_heading level1 row14" >False</th>
      <td id="T_1b374_row14_col0" class="data row14 col0" >103</td>
      <td id="T_1b374_row14_col1" class="data row14 col1" >3,287</td>
      <td id="T_1b374_row14_col2" class="data row14 col2" >3.1%</td>
      <td id="T_1b374_row14_col3" class="data row14 col3" >2.4%</td>
      <td id="T_1b374_row14_col4" class="data row14 col4" >128.2%</td>
      <td id="T_1b374_row14_col5" class="data row14 col5" >3.0%</td>
      <td id="T_1b374_row14_col6" class="data row14 col6" >104.4%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row15" class="row_heading level1 row15" >True</th>
      <td id="T_1b374_row15_col0" class="data row15 col0" >87</td>
      <td id="T_1b374_row15_col1" class="data row15 col1" >5,161</td>
      <td id="T_1b374_row15_col2" class="data row15 col2" >1.7%</td>
      <td id="T_1b374_row15_col3" class="data row15 col3" >2.4%</td>
      <td id="T_1b374_row15_col4" class="data row15 col4" >69.0%</td>
      <td id="T_1b374_row15_col5" class="data row15 col5" >1.5%</td>
      <td id="T_1b374_row15_col6" class="data row15 col6" >112.4%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row16" class="row_heading level0 row16" rowspan="2">9</th>
      <th id="T_1b374_level1_row16" class="row_heading level1 row16" >False</th>
      <td id="T_1b374_row16_col0" class="data row16 col0" >87</td>
      <td id="T_1b374_row16_col1" class="data row16 col1" >2,684</td>
      <td id="T_1b374_row16_col2" class="data row16 col2" >3.2%</td>
      <td id="T_1b374_row16_col3" class="data row16 col3" >2.7%</td>
      <td id="T_1b374_row16_col4" class="data row16 col4" >119.1%</td>
      <td id="T_1b374_row16_col5" class="data row16 col5" >3.0%</td>
      <td id="T_1b374_row16_col6" class="data row16 col6" >108.0%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row17" class="row_heading level1 row17" >True</th>
      <td id="T_1b374_row17_col0" class="data row17 col0" >94</td>
      <td id="T_1b374_row17_col1" class="data row17 col1" >4,275</td>
      <td id="T_1b374_row17_col2" class="data row17 col2" >2.2%</td>
      <td id="T_1b374_row17_col3" class="data row17 col3" >2.7%</td>
      <td id="T_1b374_row17_col4" class="data row17 col4" >80.8%</td>
      <td id="T_1b374_row17_col5" class="data row17 col5" >1.5%</td>
      <td id="T_1b374_row17_col6" class="data row17 col6" >146.6%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row18" class="row_heading level0 row18" rowspan="2">10</th>
      <th id="T_1b374_level1_row18" class="row_heading level1 row18" >False</th>
      <td id="T_1b374_row18_col0" class="data row18 col0" >60</td>
      <td id="T_1b374_row18_col1" class="data row18 col1" >2,156</td>
      <td id="T_1b374_row18_col2" class="data row18 col2" >2.8%</td>
      <td id="T_1b374_row18_col3" class="data row18 col3" >3.0%</td>
      <td id="T_1b374_row18_col4" class="data row18 col4" >92.7%</td>
      <td id="T_1b374_row18_col5" class="data row18 col5" >3.0%</td>
      <td id="T_1b374_row18_col6" class="data row18 col6" >92.7%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row19" class="row_heading level1 row19" >True</th>
      <td id="T_1b374_row19_col0" class="data row19 col0" >92</td>
      <td id="T_1b374_row19_col1" class="data row19 col1" >3,448</td>
      <td id="T_1b374_row19_col2" class="data row19 col2" >2.7%</td>
      <td id="T_1b374_row19_col3" class="data row19 col3" >3.0%</td>
      <td id="T_1b374_row19_col4" class="data row19 col4" >88.9%</td>
      <td id="T_1b374_row19_col5" class="data row19 col5" >1.5%</td>
      <td id="T_1b374_row19_col6" class="data row19 col6" >177.9%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row20" class="row_heading level0 row20" rowspan="2">11</th>
      <th id="T_1b374_level1_row20" class="row_heading level1 row20" >False</th>
      <td id="T_1b374_row20_col0" class="data row20 col0" >457</td>
      <td id="T_1b374_row20_col1" class="data row20 col1" >1,694</td>
      <td id="T_1b374_row20_col2" class="data row20 col2" >27.0%</td>
      <td id="T_1b374_row20_col3" class="data row20 col3" >20.0%</td>
      <td id="T_1b374_row20_col4" class="data row20 col4" >134.9%</td>
      <td id="T_1b374_row20_col5" class="data row20 col5" >3.0%</td>
      <td id="T_1b374_row20_col6" class="data row20 col6" >899.5%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row21" class="row_heading level1 row21" >True</th>
      <td id="T_1b374_row21_col0" class="data row21 col0" >347</td>
      <td id="T_1b374_row21_col1" class="data row21 col1" >2,697</td>
      <td id="T_1b374_row21_col2" class="data row21 col2" >12.9%</td>
      <td id="T_1b374_row21_col3" class="data row21 col3" >20.0%</td>
      <td id="T_1b374_row21_col4" class="data row21 col4" >64.3%</td>
      <td id="T_1b374_row21_col5" class="data row21 col5" >1.5%</td>
      <td id="T_1b374_row21_col6" class="data row21 col6" >857.8%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row22" class="row_heading level0 row22" rowspan="2">12</th>
      <th id="T_1b374_level1_row22" class="row_heading level1 row22" >False</th>
      <td id="T_1b374_row22_col0" class="data row22 col0" >180</td>
      <td id="T_1b374_row22_col1" class="data row22 col1" >895</td>
      <td id="T_1b374_row22_col2" class="data row22 col2" >20.1%</td>
      <td id="T_1b374_row22_col3" class="data row22 col3" >15.0%</td>
      <td id="T_1b374_row22_col4" class="data row22 col4" >134.1%</td>
      <td id="T_1b374_row22_col5" class="data row22 col5" >3.0%</td>
      <td id="T_1b374_row22_col6" class="data row22 col6" >670.3%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row23" class="row_heading level1 row23" >True</th>
      <td id="T_1b374_row23_col0" class="data row23 col0" >150</td>
      <td id="T_1b374_row23_col1" class="data row23 col1" >1,768</td>
      <td id="T_1b374_row23_col2" class="data row23 col2" >8.5%</td>
      <td id="T_1b374_row23_col3" class="data row23 col3" >15.0%</td>
      <td id="T_1b374_row23_col4" class="data row23 col4" >56.6%</td>
      <td id="T_1b374_row23_col5" class="data row23 col5" >1.5%</td>
      <td id="T_1b374_row23_col6" class="data row23 col6" >565.5%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row24" class="row_heading level0 row24" rowspan="2">13</th>
      <th id="T_1b374_level1_row24" class="row_heading level1 row24" >False</th>
      <td id="T_1b374_row24_col0" class="data row24 col0" >50</td>
      <td id="T_1b374_row24_col1" class="data row24 col1" >503</td>
      <td id="T_1b374_row24_col2" class="data row24 col2" >9.9%</td>
      <td id="T_1b374_row24_col3" class="data row24 col3" >5.0%</td>
      <td id="T_1b374_row24_col4" class="data row24 col4" >198.9%</td>
      <td id="T_1b374_row24_col5" class="data row24 col5" >3.0%</td>
      <td id="T_1b374_row24_col6" class="data row24 col6" >331.5%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row25" class="row_heading level1 row25" >True</th>
      <td id="T_1b374_row25_col0" class="data row25 col0" >49</td>
      <td id="T_1b374_row25_col1" class="data row25 col1" >1,117</td>
      <td id="T_1b374_row25_col2" class="data row25 col2" >4.4%</td>
      <td id="T_1b374_row25_col3" class="data row25 col3" >5.0%</td>
      <td id="T_1b374_row25_col4" class="data row25 col4" >87.7%</td>
      <td id="T_1b374_row25_col5" class="data row25 col5" >1.5%</td>
      <td id="T_1b374_row25_col6" class="data row25 col6" >292.4%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row26" class="row_heading level0 row26" rowspan="2">14</th>
      <th id="T_1b374_level1_row26" class="row_heading level1 row26" >False</th>
      <td id="T_1b374_row26_col0" class="data row26 col0" >33</td>
      <td id="T_1b374_row26_col1" class="data row26 col1" >263</td>
      <td id="T_1b374_row26_col2" class="data row26 col2" >12.6%</td>
      <td id="T_1b374_row26_col3" class="data row26 col3" >5.0%</td>
      <td id="T_1b374_row26_col4" class="data row26 col4" >251.3%</td>
      <td id="T_1b374_row26_col5" class="data row26 col5" >3.0%</td>
      <td id="T_1b374_row26_col6" class="data row26 col6" >418.9%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row27" class="row_heading level1 row27" >True</th>
      <td id="T_1b374_row27_col0" class="data row27 col0" >29</td>
      <td id="T_1b374_row27_col1" class="data row27 col1" >609</td>
      <td id="T_1b374_row27_col2" class="data row27 col2" >4.8%</td>
      <td id="T_1b374_row27_col3" class="data row27 col3" >5.0%</td>
      <td id="T_1b374_row27_col4" class="data row27 col4" >95.2%</td>
      <td id="T_1b374_row27_col5" class="data row27 col5" >1.5%</td>
      <td id="T_1b374_row27_col6" class="data row27 col6" >317.3%</td>
    </tr>
    <tr>
      <th id="T_1b374_level0_row28" class="row_heading level0 row28" rowspan="2">15</th>
      <th id="T_1b374_level1_row28" class="row_heading level1 row28" >False</th>
      <td id="T_1b374_row28_col0" class="data row28 col0" >8</td>
      <td id="T_1b374_row28_col1" class="data row28 col1" >74</td>
      <td id="T_1b374_row28_col2" class="data row28 col2" >10.8%</td>
      <td id="T_1b374_row28_col3" class="data row28 col3" >5.0%</td>
      <td id="T_1b374_row28_col4" class="data row28 col4" >216.1%</td>
      <td id="T_1b374_row28_col5" class="data row28 col5" >3.0%</td>
      <td id="T_1b374_row28_col6" class="data row28 col6" >360.1%</td>
    </tr>
    <tr>
      <th id="T_1b374_level1_row29" class="row_heading level1 row29" >True</th>
      <td id="T_1b374_row29_col0" class="data row29 col0" >9</td>
      <td id="T_1b374_row29_col1" class="data row29 col1" >194</td>
      <td id="T_1b374_row29_col2" class="data row29 col2" >4.6%</td>
      <td id="T_1b374_row29_col3" class="data row29 col3" >5.0%</td>
      <td id="T_1b374_row29_col4" class="data row29 col4" >92.7%</td>
      <td id="T_1b374_row29_col5" class="data row29 col5" >1.5%</td>
      <td id="T_1b374_row29_col6" class="data row29 col6" >309.1%</td>
    </tr>
  </tbody>
</table>

<br> **Logo**

<a href="https://www.freepik.com/free-vector/shine-old-wooden-chest-realistic-composition-transparent-background-with-vintage-coffer-sparkling-particles_7497397.htm#query=treasure&position=7&from_view=search&track=sph">Image
by macrovector</a> on Freepik
