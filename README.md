
<!-- README.md is generated from README.Rmd. Please edit that file -->

# actxps <a href="https://github.com/mattheaphy/actxpspy/"><img src="doc/images/logo.png" align="right" height="138" /></a>

The actxps package provides a set of tools to assist with the creation
of actuarial experience studies. Experience studies are used by
actuaries to explore historical experience across blocks of business and
to inform assumption setting for projection models.

-   The `ExposedDF()` class converts census-level records into policy or
    calendar year exposure records.

## Basic usage

The actxps package includes simulated census data for a theoretical
deferred annuity product with an optional guaranteed income rider. The
grain of this data is one row *per policy*.

``` python
import actxps as xp

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

    A DataFrame: 141252 x 15
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

<br> **Logo**

<a href="https://www.freepik.com/free-vector/shine-old-wooden-chest-realistic-composition-transparent-background-with-vintage-coffer-sparkling-particles_7497397.htm#query=treasure&position=7&from_view=search&track=sph">Image
by macrovector</a> on Freepik
