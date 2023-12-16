# actxps
<a href="https://github.com/mattheaphy/actxpspy/"><img src="docs/images/logo.png" align="right" height="138" /></a>

<!-- README.md is generated from README.Rmd. Please edit that file -->

The actxps package provides a set of tools to assist with the creation
of actuarial experience studies. Experience studies are used by
actuaries to explore historical experience across blocks of business and
to inform assumption setting for projection models.

- The `ExposedDF` class converts census-level records into policy or
  calendar year exposure records.
- The `exp_stats()` method of `ExposedDF` creates `ExpStats` experience
  summary objects containing observed termination rates and claims.
  Optionally, expected termination rates, actual-to-expected ratios, and
  limited fluctuation credibility estimates can also be returned.
- The `add_transactions()` method attaches summarized transactions to an
  `ExposedDF` object.
- The `trx_stats()` method of `ExposedDF` creates `TrxStats` transaction
  summary objects containing transaction counts, amounts, frequencies,
  and utilization. Optionally, transaction amounts can be expressed as a
  percentage of one or more variables to calculate rates or
  actual-to-expected ratios.
- The `plot()` and `table()` methods of `ExpStats` and `TrxStats` create
  plots and tables for reporting.

## Basic usage

The actxps package includes simulated census data for a theoretical
deferred annuity product with an optional guaranteed income rider. The
grain of this data is one row *per policy*.

``` python
import actxps as xp
import numpy as np

census_dat = xp.load_census_dat()
print(census_dat)
```

           pol_num     status issue_date  inc_guar   qual  age product gender  \
    0            1     Active 2014-12-17      True  False   56       b      F   
    1            2  Surrender 2007-09-24     False  False   71       a      F   
    2            3     Active 2012-10-06     False   True   62       b      F   
    3            4  Surrender 2005-06-27      True   True   62       c      M   
    4            5     Active 2019-11-22     False  False   62       c      F   
    ...        ...        ...        ...       ...    ...  ...     ...    ...   
    19995    19996     Active 2014-08-11      True   True   55       b      F   
    19996    19997  Surrender 2006-11-20     False  False   68       c      F   
    19997    19998  Surrender 2017-02-20      True  False   68       c      F   
    19998    19999     Active 2015-04-11     False   True   67       a      M   
    19999    20000     Active 2009-04-29      True   True   72       c      M   

           wd_age  premium  term_date  
    0          77    370.0        NaT  
    1          71    708.0 2019-03-08  
    2          63    466.0        NaT  
    3          62    485.0 2018-11-29  
    4          67    978.0        NaT  
    ...       ...      ...        ...  
    19995      75   3551.0        NaT  
    19996      77    336.0 2017-07-09  
    19997      68   1222.0 2018-08-03  
    19998      78   2138.0        NaT  
    19999      72   5751.0        NaT  

    [20000 rows x 11 columns]

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

(exp_res.plot() +
  scale_color_manual(values = colors) + 
  labs(title = "Observed Surrender Rates by Policy Year and Income Guarantee Presence") + 
  theme_light())
```

<img src="README_files/figure-commonmark/plots-output-1.png"
id="plots-1" />

    <Figure Size: (640 x 480)>

``` python
exp_res.table()
```

<style type="text/css">
#T_fa9a6 th {
  font-weight: bold;
  font-size: 100%;
}
#T_fa9a6 tr {
  font-size: 100%;
}
#T_fa9a6 caption {
  font-size: 100%;
}
#T_fa9a6 th.col_heading {
  text-align: center;
}
#T_fa9a6 th.col_heading.level0 {
  font-size: 1.1em;
}
#T_fa9a6 caption {
  caption-side: top;
}
#T_fa9a6_row0_col2, #T_fa9a6_row11_col2 {
  background-color: #f5fbee;
  color: #000000;
}
#T_fa9a6_row0_col4 {
  background-color: #f7f6f6;
  color: #000000;
}
#T_fa9a6_row0_col6, #T_fa9a6_row1_col6, #T_fa9a6_row7_col4 {
  background-color: #053061;
  color: #f1f1f1;
}
#T_fa9a6_row1_col2 {
  background-color: #f7fcf0;
  color: #000000;
}
#T_fa9a6_row1_col4 {
  background-color: #408fc1;
  color: #f1f1f1;
}
#T_fa9a6_row2_col2 {
  background-color: #f1faeb;
  color: #000000;
}
#T_fa9a6_row2_col4 {
  background-color: #fddbc7;
  color: #000000;
}
#T_fa9a6_row2_col6, #T_fa9a6_row3_col6, #T_fa9a6_row11_col4 {
  background-color: #0a3b70;
  color: #f1f1f1;
}
#T_fa9a6_row3_col2 {
  background-color: #f6fbef;
  color: #000000;
}
#T_fa9a6_row3_col4 {
  background-color: #4695c4;
  color: #f1f1f1;
}
#T_fa9a6_row4_col2 {
  background-color: #f3faec;
  color: #000000;
}
#T_fa9a6_row4_col4 {
  background-color: #84bcd9;
  color: #000000;
}
#T_fa9a6_row4_col6 {
  background-color: #08366a;
  color: #f1f1f1;
}
#T_fa9a6_row5_col2, #T_fa9a6_row7_col2 {
  background-color: #f6fcef;
  color: #000000;
}
#T_fa9a6_row5_col4, #T_fa9a6_row18_col6 {
  background-color: #1b5a9c;
  color: #f1f1f1;
}
#T_fa9a6_row5_col6 {
  background-color: #09386d;
  color: #f1f1f1;
}
#T_fa9a6_row6_col2, #T_fa9a6_row10_col2, #T_fa9a6_row17_col2 {
  background-color: #ebf7e5;
  color: #000000;
}
#T_fa9a6_row6_col4 {
  background-color: #fbe5d8;
  color: #000000;
}
#T_fa9a6_row6_col6, #T_fa9a6_row10_col6 {
  background-color: #134c87;
  color: #f1f1f1;
}
#T_fa9a6_row7_col6 {
  background-color: #073467;
  color: #f1f1f1;
}
#T_fa9a6_row8_col2 {
  background-color: #edf8e7;
  color: #000000;
}
#T_fa9a6_row8_col4 {
  background-color: #c0dceb;
  color: #000000;
}
#T_fa9a6_row8_col6 {
  background-color: #114781;
  color: #f1f1f1;
}
#T_fa9a6_row9_col2 {
  background-color: #f4fbed;
  color: #000000;
}
#T_fa9a6_row9_col4 {
  background-color: #185493;
  color: #f1f1f1;
}
#T_fa9a6_row9_col6 {
  background-color: #0f437b;
  color: #f1f1f1;
}
#T_fa9a6_row10_col4, #T_fa9a6_row24_col6 {
  background-color: #b1d5e7;
  color: #000000;
}
#T_fa9a6_row11_col6 {
  background-color: #0e4179;
  color: #f1f1f1;
}
#T_fa9a6_row12_col2 {
  background-color: #eaf7e4;
  color: #000000;
}
#T_fa9a6_row12_col4 {
  background-color: #a5cee3;
  color: #000000;
}
#T_fa9a6_row12_col6 {
  background-color: #15508d;
  color: #f1f1f1;
}
#T_fa9a6_row13_col2 {
  background-color: #f2faeb;
  color: #000000;
}
#T_fa9a6_row13_col4 {
  background-color: #1a5899;
  color: #f1f1f1;
}
#T_fa9a6_row13_col6 {
  background-color: #175290;
  color: #f1f1f1;
}
#T_fa9a6_row14_col2 {
  background-color: #e4f5df;
  color: #000000;
}
#T_fa9a6_row14_col4 {
  background-color: #d8e9f1;
  color: #000000;
}
#T_fa9a6_row14_col6 {
  background-color: #1e61a5;
  color: #f1f1f1;
}
#T_fa9a6_row15_col2 {
  background-color: #eef9e8;
  color: #000000;
}
#T_fa9a6_row15_col4 {
  background-color: #2f79b5;
  color: #f1f1f1;
}
#T_fa9a6_row15_col6 {
  background-color: #2065ab;
  color: #f1f1f1;
}
#T_fa9a6_row16_col2 {
  background-color: #e4f4de;
  color: #000000;
}
#T_fa9a6_row16_col4 {
  background-color: #c2ddec;
  color: #000000;
}
#T_fa9a6_row16_col6 {
  background-color: #1f63a8;
  color: #f1f1f1;
}
#T_fa9a6_row17_col4 {
  background-color: #4291c2;
  color: #f1f1f1;
}
#T_fa9a6_row17_col6 {
  background-color: #2e77b5;
  color: #f1f1f1;
}
#T_fa9a6_row18_col2 {
  background-color: #e7f6e2;
  color: #000000;
}
#T_fa9a6_row18_col4, #T_fa9a6_row29_col4 {
  background-color: #6eaed2;
  color: #f1f1f1;
}
#T_fa9a6_row19_col2 {
  background-color: #e8f6e2;
  color: #000000;
}
#T_fa9a6_row19_col4 {
  background-color: #5fa5cd;
  color: #f1f1f1;
}
#T_fa9a6_row19_col6 {
  background-color: #3a87bd;
  color: #f1f1f1;
}
#T_fa9a6_row20_col2 {
  background-color: #084081;
  color: #f1f1f1;
}
#T_fa9a6_row20_col4, #T_fa9a6_row26_col6 {
  background-color: #e4eef4;
  color: #000000;
}
#T_fa9a6_row20_col6, #T_fa9a6_row26_col4 {
  background-color: #67001f;
  color: #f1f1f1;
}
#T_fa9a6_row21_col2 {
  background-color: #86d0c0;
  color: #000000;
}
#T_fa9a6_row21_col4 {
  background-color: #276eb0;
  color: #f1f1f1;
}
#T_fa9a6_row21_col6 {
  background-color: #8a0b25;
  color: #f1f1f1;
}
#T_fa9a6_row22_col2 {
  background-color: #2d8fbf;
  color: #f1f1f1;
}
#T_fa9a6_row22_col4 {
  background-color: #e3edf3;
  color: #000000;
}
#T_fa9a6_row22_col6 {
  background-color: #e98b6e;
  color: #f1f1f1;
}
#T_fa9a6_row23_col2 {
  background-color: #bde5be;
  color: #000000;
}
#T_fa9a6_row23_col4 {
  background-color: #1c5c9f;
  color: #f1f1f1;
}
#T_fa9a6_row23_col6 {
  background-color: #fbd0b9;
  color: #000000;
}
#T_fa9a6_row24_col2 {
  background-color: #addfb7;
  color: #000000;
}
#T_fa9a6_row24_col4 {
  background-color: #e48066;
  color: #f1f1f1;
}
#T_fa9a6_row25_col2 {
  background-color: #dcf1d7;
  color: #000000;
}
#T_fa9a6_row25_col4 {
  background-color: #5ca3cb;
  color: #f1f1f1;
}
#T_fa9a6_row25_col6 {
  background-color: #96c7df;
  color: #000000;
}
#T_fa9a6_row26_col2 {
  background-color: #8ad2bf;
  color: #000000;
}
#T_fa9a6_row27_col2 {
  background-color: #daf1d5;
  color: #000000;
}
#T_fa9a6_row27_col4 {
  background-color: #78b4d5;
  color: #000000;
}
#T_fa9a6_row27_col6 {
  background-color: #a7d0e4;
  color: #000000;
}
#T_fa9a6_row28_col2 {
  background-color: #a2dbb7;
  color: #000000;
}
#T_fa9a6_row28_col4 {
  background-color: #c94741;
  color: #f1f1f1;
}
#T_fa9a6_row28_col6 {
  background-color: #c7e0ed;
  color: #000000;
}
#T_fa9a6_row29_col2 {
  background-color: #dbf1d5;
  color: #000000;
}
#T_fa9a6_row29_col6 {
  background-color: #a2cde3;
  color: #000000;
}
</style>

|        |          |        |           |                   | expected_1        |        | expected_2        |        |
|--------|----------|--------|-----------|-------------------|-------------------|--------|-------------------|--------|
|        |          | Claims | Exposures | *q<sup>obs</sup>* | *q<sup>exp</sup>* | *A/E*  | *q<sup>exp</sup>* | *A/E*  |
| pol_yr | inc_guar |        |           |                   |                   |        |                   |        |
| 1      | False    | 56     | 7,720     | 0.7%              | 0.5%              | 145.1% | 3.0%              | 24.2%  |
|        | True     | 46     | 11,532    | 0.4%              | 0.5%              | 79.8%  | 1.5%              | 26.6%  |
| 2      | False    | 92     | 7,103     | 1.3%              | 0.8%              | 166.5% | 3.0%              | 43.2%  |
|        | True     | 68     | 10,612    | 0.6%              | 0.8%              | 82.4%  | 1.5%              | 42.7%  |
| 3      | False    | 67     | 6,447     | 1.0%              | 1.1%              | 98.5%  | 3.0%              | 34.6%  |
|        | True     | 57     | 9,650     | 0.6%              | 1.1%              | 56.0%  | 1.5%              | 39.4%  |
| 4      | False    | 123    | 5,799     | 2.1%              | 1.3%              | 159.1% | 3.0%              | 70.7%  |
|        | True     | 45     | 8,737     | 0.5%              | 1.3%              | 38.6%  | 1.5%              | 34.3%  |
| 5      | False    | 97     | 5,106     | 1.9%              | 1.6%              | 117.9% | 3.0%              | 63.3%  |
|        | True     | 67     | 7,810     | 0.9%              | 1.6%              | 53.2%  | 1.5%              | 57.2%  |
| 6      | False    | 96     | 4,494     | 2.1%              | 1.9%              | 113.1% | 3.0%              | 71.2%  |
|        | True     | 56     | 6,882     | 0.8%              | 1.9%              | 43.1%  | 1.5%              | 54.2%  |
| 7      | False    | 92     | 3,899     | 2.4%              | 2.2%              | 108.9% | 3.0%              | 78.7%  |
|        | True     | 72     | 6,018     | 1.2%              | 2.2%              | 55.2%  | 1.5%              | 79.8%  |
| 8      | False    | 103    | 3,287     | 3.1%              | 2.4%              | 128.2% | 3.0%              | 104.4% |
|        | True     | 87     | 5,161     | 1.7%              | 2.4%              | 69.0%  | 1.5%              | 112.4% |
| 9      | False    | 87     | 2,684     | 3.2%              | 2.7%              | 119.1% | 3.0%              | 108.0% |
|        | True     | 94     | 4,275     | 2.2%              | 2.7%              | 80.8%  | 1.5%              | 146.6% |
| 10     | False    | 60     | 2,156     | 2.8%              | 3.0%              | 92.7%  | 3.0%              | 92.7%  |
|        | True     | 92     | 3,448     | 2.7%              | 3.0%              | 88.9%  | 1.5%              | 177.9% |
| 11     | False    | 457    | 1,694     | 27.0%             | 20.0%             | 134.9% | 3.0%              | 899.5% |
|        | True     | 347    | 2,697     | 12.9%             | 20.0%             | 64.3%  | 1.5%              | 857.8% |
| 12     | False    | 180    | 895       | 20.1%             | 15.0%             | 134.1% | 3.0%              | 670.3% |
|        | True     | 150    | 1,768     | 8.5%              | 15.0%             | 56.6%  | 1.5%              | 565.5% |
| 13     | False    | 50     | 503       | 9.9%              | 5.0%              | 198.9% | 3.0%              | 331.5% |
|        | True     | 49     | 1,117     | 4.4%              | 5.0%              | 87.7%  | 1.5%              | 292.4% |
| 14     | False    | 33     | 263       | 12.6%             | 5.0%              | 251.3% | 3.0%              | 418.9% |
|        | True     | 29     | 609       | 4.8%              | 5.0%              | 95.2%  | 1.5%              | 317.3% |
| 15     | False    | 8      | 74        | 10.8%             | 5.0%              | 216.1% | 3.0%              | 360.1% |
|        | True     | 9      | 194       | 4.6%              | 5.0%              | 92.7%  | 1.5%              | 309.1% |

Experience Study Results  
Target status: Surrender  
Study range: 1900-01-01 to 2019-12-31

Launch a shiny app to interactively explore experience data.

``` python
exposed_data.exp_shiny(expected=['expected_1', 'expected_2'])
```

<img src="docs/images/exp_shiny.png" width="100%" />

<br> **Logo**

<a href="https://www.freepik.com/free-vector/shine-old-wooden-chest-realistic-composition-transparent-background-with-vintage-coffer-sparkling-particles_7497397.htm#query=treasure&position=7&from_view=search&track=sph">Image
by macrovector</a> on Freepik
