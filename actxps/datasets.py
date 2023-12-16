from importlib import resources
import pandas as pd
from actxps.tools import document
from joblib import load


def load_toy_census() -> pd.DataFrame:
    """
    # Toy policy census data

    A tiny dataset containing 3 policies: one active, one terminated due to
    death, and one terminated due to surrender.    

    ## Returns:

    A data frame with 3 rows and 4 columns:

    - `pol_num` = policy number
    - `status` = policy status
    - `issue_date` = issue date
    - `term_date` = termination date

    """
    stream = resources.files('actxps').joinpath('data/toy_census.csv')
    return pd.read_csv(stream,
                       index_col=0,
                       dtype={'pol_num': int,
                              'status': 'category'},
                       parse_dates=['issue_date', 'term_date'])


_sim_doc = """
# Simulated annuity data

Simulated data for a theoretical deferred annuity product with
an optional guaranteed income rider. This data is theoretical only and
does not represent the experience on any specific product.

Three data frames contain census records (`census_dat`),
withdrawal transactions (`withdrawals`), and historical account values
(`account_vals`).

## Returns

### Census data (`load_census_dat()`)

A data frame with 20,000 rows and 11 columns:

- `pol_num` = policy number
- `status` - policy status: Active, Surrender, or Death
- `issue_date` - issue date
- `inc_guar` - indicates whether the policy was issued with an income guarantee
- `qual` - indicates whether the policy was purchased with tax-qualified funds
- `age` - issue age
- `product` - product: a, b, or c
- `gender` - M (Male) or F (Female)
- `wd_age` - Age that withdrawals commence
- `premium` - Single premium deposit
- `term_date` - termination date upon death or surrender

### Withdrawal data (`load_withdrawals()`)

A data frame with 4 columns:

- `pol_num` - policy number
- `trx_date` - withdrawal transaction date
- `trx_type` - withdrawal transaction type, either Base or Rider
- `trx_amt` - withdrawal transaction amount

### Account values data (`load_account_vals()`)

A data frame with 3 columns:

- `pol_num` - policy number
- `pol_date_yr` - policy anniversary date (beginning of year)
- `av_anniv` - account value on the policy anniversary date
"""


@document(_sim_doc)
def load_census_dat() -> pd.DataFrame:
    stream = resources.files('actxps').joinpath('data/census_dat')
    return load(stream)


@document(_sim_doc)
def load_withdrawals() -> pd.DataFrame:
    stream = resources.files('actxps').joinpath('data/withdrawals')
    return load(stream)


@document(_sim_doc)
def load_account_vals() -> pd.DataFrame:
    stream = resources.files('actxps').joinpath('data/account_vals')
    return load(stream)


_mort_doc = """
# 2012 Individual Annuity Mortality Table and Projection Scale G2

Mortality rates and mortality improvement rates from the 2012 Individual
Annuity Mortality Basic (IAMB) Table and Projection Scale G2.

## Returns

### 2012 IAMB table (`load_qx_iamb()`)

A data frame with 242 rows and 3 columns:

- `age` - attained age
- `qx` - mortality rate
- `gender` - Female or Male


### Projection Scale G2 (`load_scale_g2()`)

A data frame with 242 rows and 3 columns:

- `age` - attained age
- `mi` - mortality improvement rate
- `gender` - Female or Male


## Source

 - https://mort.soa.org/
 - https://www.actuary.org/sites/default/files/files/publications/Payout_Annuity_Report_09-28-11.pdf
"""


@document(_mort_doc)
def load_qx_iamb():
    stream = resources.files('actxps').joinpath('data/qx_iamb.csv')
    return pd.read_csv(stream, index_col=0,
                       dtype={'age': int,
                              'qx': float,
                              'gender': str})


@document(_mort_doc)
def load_scale_g2():
    stream = resources.files('actxps').joinpath('data/scaleG2.csv')
    return pd.read_csv(stream, index_col=0,
                       dtype={'age': int,
                              'mi': float,
                              'gender': str})
