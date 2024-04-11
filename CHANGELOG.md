## v1.0.2

- Small correction to the final policy year exposure for leap years

## v1.0.0

- To improve speed and efficiency, the data frame backend was changed from Pandas to Polars. 
  - The `data` property of `ExposedDF`, `ExpStats`, and `TrxStats` is now a Polars data frame.
  - `ExposedDF`, `ExpStats.from_DataFrame`, and `TrxStats.from_DataFrame` will accept both Polars and Pandas data frames. However, Pandas data frames are immediately converted to Polars.
- Removed all add_* date functions which are no longer needed under Polars
- `ExposedDF`'s `groupby` method was renamed to `group_by`.

## v0.0.1 (2024-02-18)

- Several updates to align with the R version of actxps
  
  - Added optional confidence interval outputs to `ExpStats` and `TrxStats` objects
  - Added an `expose_split()` method to `ExposedDF` classes, which divides calendar period exposures into pre- and post-policy anniversary segments. This creates a `SplitExposedDF` object, which is a subclass of `ExposedDF`.
  - Added special plotting functions `ExpStats.plot_termination_rates()`, `ExpStats.plot_actual_to_expected()`, and `TrxStats.plot_utilization_rates()`
  - Added `from_DataFrame` class methods to ExpStats and TrxStats objects that enables the creation of experience summary objects pre-aggregated data sets.
  - Added the `great_tables` package as the backend for table() methods
  - Shiny UI updates  
  - Added a 'points' geom, log 10 y-axis, and confidence interval options to plot methods
  - Added sample data for pre-aggregated exposures (`load_agg_sim_dat()`)
  - Bug fix - the target status wasn't being fully exposed when using calendar period exposures
  - The assumed default status on `ExposedDF` objects is now the most common status instead of the first observed status.
  - Added helper background functions for column selection
  - Added several articles to the package's website

- Current differences from the R version are:

  - No support for plotting a second variable on the y-axis for `.plot()` methods
  - No predictive modeling support function (`add_predictions()`, `step_expose()`)
  - Table output cannot be saved to a png directly in the shiny app
  - functions like `pol_interval()` don't accept arbitrary durations. Only 'year', 'quarter', 'month', or 'week' are allowed.


## v0.0.0.9000 (2023-12-14)

- Development version of `actxps`