
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
```

Convert census records to exposure records with one row *per policy per
year*.

``` python
exposed_data = xp.ExposedDF(census_dat,
                            end_date="2019-12-31",
                            target_status="Surrender")

exposed_data.data
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
      <th>pol_yr</th>
      <th>pol_date_yr</th>
      <th>pol_date_yr_end</th>
      <th>exposure</th>
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
      <td>1</td>
      <td>2014-12-17</td>
      <td>2015-12-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>2</td>
      <td>2015-12-17</td>
      <td>2016-12-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>3</td>
      <td>2016-12-17</td>
      <td>2017-12-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>4</td>
      <td>2017-12-17</td>
      <td>2018-12-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>5</td>
      <td>2018-12-17</td>
      <td>2019-12-16</td>
      <td>1.000000</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141247</th>
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
      <td>7</td>
      <td>2015-04-29</td>
      <td>2016-04-28</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>141248</th>
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
      <td>8</td>
      <td>2016-04-29</td>
      <td>2017-04-28</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>141249</th>
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
      <td>9</td>
      <td>2017-04-29</td>
      <td>2018-04-28</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>141250</th>
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
      <td>10</td>
      <td>2018-04-29</td>
      <td>2019-04-28</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>141251</th>
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
      <td>11</td>
      <td>2019-04-29</td>
      <td>2020-04-28</td>
      <td>0.674863</td>
    </tr>
  </tbody>
</table>
<p>141252 rows × 15 columns</p>
</div>

<br> **Logo**

<a href="https://www.freepik.com/free-vector/shine-old-wooden-chest-realistic-composition-transparent-background-with-vintage-coffer-sparkling-particles_7497397.htm#query=treasure&position=7&from_view=search&track=sph">Image
by macrovector</a> on Freepik
