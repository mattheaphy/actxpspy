# code to prepare example datasets goes here
import polars as pl

toy_census = pl.DataFrame({
    'pol_num': pl.int_range(1, 4, eager=True),
    'status': pl.Series(["Active", "Death", "Surrender"], dtype=pl.Categorical),
    'issue_date': pl.Series(["2010-01-01", "2011-05-27", "2009-11-10"]),
    'term_date': pl.Series(["", "2020-09-14", "2022-02-25"])
}).with_columns(
    pl.col('issue_date', 'term_date').str.to_date('%Y-%m-%d', strict=False))

toy_census.write_csv('toy_census.csv')
