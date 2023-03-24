## code to prepare example datasets goes here
import pandas as pd
import numpy as np

toy_census = pd.DataFrame({
  'pol_num': np.arange(1, 4),
  'status': pd.Series(["Active", "Death", "Surrender"], dtype='category'),
  'issue_date': pd.to_datetime(["2010-01-01", "2011-05-27", "2009-11-10"]),
  'term_date': pd.to_datetime([np.nan, "2020-09-14", "2022-02-25"])
})

toy_census.to_csv('toy_census.csv')