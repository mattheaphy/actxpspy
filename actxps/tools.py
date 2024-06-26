# This module contains helper functions used by other modules
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
from itertools import product
from plotnine import (
    ggplot,
    geom_point,
    geom_line,
    geom_col,
    geom_errorbar,
    aes,
    facet_wrap,
    scale_y_continuous,
    scale_color_manual,
    scale_fill_manual)
from warnings import warn
from scipy.stats import binom, norm
_use_default_colors = False


def document(docstring: str):
    """
    Decorator function factory for adding docstrings

    Parameters
    ----------
    docstring : str
      A docstring to add to a function

    Returns
    ----------
    A decorator function that adds a docstring to another function
    """

    def decorator(func):
        func.__doc__ = docstring
        return func

    return decorator


def arg_match(name: str, x, allowed):
    """
    Verify that an argument contains one of several allowed values.

    A `ValueError` exception is raised if the argument value `x` is not allowed.

    Parameters
    -----------
    name : str
        Argument name
    x : Any
        Argument value
    allowed : Any
        A list of allowed argument values

    References
    -----------
    This function is inspired by the R language's `arg.match()` and 
    `rlang::arg_match()` functions.
    """
    if x not in allowed:
        allowed = ", ".join([f'"{a}"' for a in allowed[:-1]]) + \
            f', or "{allowed[-1]}"'
        raise ValueError(f'`{name}` must be one of {allowed}. '
                         f'"{x}" is not allowed.')


def _plot_experience(xp_obj,
                     x: str = None,
                     y: str = "q_obs",
                     color: str = None,
                     mapping: aes = None,
                     scales: str = "fixed",
                     geoms: str = "lines",
                     y_labels: callable = lambda l: [
                         f"{v * 100:.1f}%" for v in l],
                     facets: list | str = None,
                     y_log10: bool = False,
                     conf_int_bars: bool = False,
                     alt_data=None,
                     group_insert=1):
    """
    This helper function is used by `ExpStats.plot()` and `TrxStats.plot()`. 
    It is not meant to be called directly.
    """

    from actxps.trx_stats import TrxStats
    from actxps.exp_stats import ExpStats

    assert isinstance(xp_obj, TrxStats | ExpStats)

    groups = xp_obj.groups
    if groups is None:
        groups = []
    else:
        groups = groups.copy()

    # handling for special plotting functions
    if alt_data is None:
        data = xp_obj.data.clone()
    else:
        data = alt_data
        groups.insert(group_insert, 'Series')

    if groups == []:
        groups = ["All"]
        data = data.with_columns(All=pl.lit(""))

    def auto_aes(var, default, if_none):
        if var is None:
            if len(groups) < default:
                return if_none
            else:
                return groups[default - 1]
        else:
            return var

    arg_match("geoms", geoms, ["lines", "bars", "points"])

    # set up aesthetics
    if mapping is None:
        x = auto_aes(x, 1, "All")
        if color != "_no_color":
            color = auto_aes(color, 2, None)
        else:
            color = None
        if color is None:
            mapping = aes(x, y)
        else:
            mapping = aes(x, y,
                          color=color, fill=color, group=color)

    if facets is None:
        facets = groups[2:]
        if len(facets) == 0:
            facets = None

    if y_log10:
        y_trans = "log10"
    else:
        y_trans = "identity"

    p = (ggplot(data, mapping) +
         scale_y_continuous(labels=y_labels, trans=y_trans))

    global _use_default_colors
    if _use_default_colors and (color or 'color' in mapping.keys()) is not None:
        colors = ["#1367D4", "#7515EB", "#EB15E4", "#1AC4F2",
                  "#1FF2C1", "#C6E531", "#FFA13D", "#FF7647"]
        p = (p +
             scale_color_manual(colors) +
             scale_fill_manual(colors))

    if geoms == "lines":
        p = p + geom_point() + geom_line()
    elif geoms == "points":
        p = p + geom_point()
    else:
        p = p + geom_col(position="dodge")

    if conf_int_bars:

        if not xp_obj.xp_params['conf_int']:
            _conf_int_warning()
        else:

            y_chr = p.mapping['y']
            y_min_max = [y_chr + "_lower", y_chr + "_upper"]
            if all([_ in data.columns for _ in y_min_max]):
                p = p + geom_errorbar(aes(ymin=y_min_max[0],
                                          ymax=y_min_max[1]))
            else:
                warn("Confidence intervals are not available for the " +
                     "selected y-variable.")

    if facets is None:
        return p
    else:
        return p + facet_wrap(facets, scales=scales)


def _pivot_plot_special(xp_obj, piv_cols: set, values_to="Rate") -> pl.DataFrame:
    """
    This internal function is used to pivot `ExpStats` or `TrxStats` data frames
    before they're passed to special plotting functions.

    Parameters
    ----------
    xp_obj : ExpStats | TrxStats
        An experience summary xp_obj
    piv_cols : list | set
        A primary set of columns to pivot longer
    values_to : str, default="Rate"
        Name of the values column in the pivoted xp_obj.

    Returns
    -------
    pl.DataFrame
        A pivoted dataframe
    """

    data = xp_obj.data
    piv_cols = set(piv_cols).intersection(data.columns)
    id_cols = set(data.columns).difference(piv_cols)

    if not xp_obj.xp_params['conf_int']:
        data = data.melt(id_vars=id_cols, value_vars=piv_cols,
                         variable_name='Series', value_name=values_to)
    else:
        extra_piv_cols = [x + y for x, y in
                          product(piv_cols, ["_upper", "_lower"])]
        extra_piv_cols = set(extra_piv_cols).intersection(data.columns)
        id_cols = id_cols.difference(extra_piv_cols)

        data = (
            data.
            melt(id_vars=id_cols,
                 value_vars=list(piv_cols) + list(extra_piv_cols),
                 variable_name='Series', value_name=values_to).
            with_columns(
                val_type=(pl.col('Series').str.
                          extract('_(upper|lower)$').
                          fill_null(values_to)),
                Series=pl.col('Series').str.replace('_(upper|lower)$', '')).
            pivot(index=list(id_cols) + ['Series'],
                  columns='val_type', values=values_to).
            rename({'lower': values_to + '_lower',
                    'upper': values_to + '_upper'})
        )

    return data


def _set_actxps_plot_theme():
    """
    Plotting theme for vignettes

    This is an internal function used to set a plotting theme in vignettes
    and articles.
    """

    from plotnine import (
        theme_set,
        theme_light,
        theme,
        element_rect
    )

    global _use_default_colors
    _use_default_colors = True

    theme_set(theme_light() +
              theme(strip_background=element_rect(fill="#1367D4"))
              )


def _conf_int_warning():
    """
    This internal function provides a common warning that is used by multiple
    functions.
    """
    warn("This object has no confidence intervals. Hint: pass " +
         "`conf_int = True` to `exp_stats()` or `trx_stats()` to calculate " +
         "confidence intervals.")


def _verify_exposed_df(expo):
    from actxps import ExposedDF
    assert isinstance(expo, ExposedDF), \
        "An `ExposedDF` object is required."


def _verify_col_names(x_names, required: set):
    """
    Internal function to verify that required names exist and to 
    send an error if not.
    """
    unmatched = required.difference(x_names)
    assert len(unmatched) == 0, \
        f"The following columns are missing: {', '.join(unmatched)}. " + \
        "Hint: create these columns or use the `col_*` arguments to " + \
        "specify existing columns that should be mapped to these elements."


def _date_str(x) -> str:
    """
    Internal function for converting dates to ISO-8601 format. If x is not a 
    date object, it is returned as-is.
    """
    try:
        return x.strftime('%Y-%m-%d')
    except:
        return x


def _qbinom(q: float, obs: str = "exposure", prob: str = "q_obs") -> pl.Expr:
    """
    Internal function for the binomial distribution expressed as probabilities

    Parameters
    ----------
    q : float
        Percentile
    obs : str, default="exposure"
        A column name containing exposures or the number of observations
    prob : str, default="q_obs"
        A column name containing the probability of an event

    Returns
    -------
    pl.Expr
    """
    return (pl.struct(pl.col(obs).round(), pl.col(prob)).
            map_batches(lambda x: binom.ppf(q, x.struct[0], x.struct[1])) /
            pl.col(obs))


def _qnorm(q: float, mean: str, sd: str | pl.Expr) -> pl.Expr:
    """
    Internal function for the inverse cumulative normal distribution
    that returns the mean when the standard deviation is zero.

    Parameters
    ----------
    q : float
        Percentile
    mean : str
        A column name containing means
    sd : str | pl.Expr
        A column name containing standard deviations

    Returns
    -------
    pl.Expr
    """
    if isinstance(sd, str):
        sd = pl.col(sd)

    return (pl.struct(pl.col(mean), pl.max_horizontal(sd, 1E-16)).
            map_batches(lambda x: norm.ppf(q, x.struct[0], x.struct[1])))


def relocate(data: pl.DataFrame,
             x: str | list | np.ndarray,
             before: str = None,
             after: str = None) -> pl.DataFrame:
    """
    Reorder columns in a data frame

    Move the columns in `x` before or after a given column. If neither `before`
    or `after` are specified, `x` will be moved to the left. If both `before`
    and `after` are specified, an error is returned.

    Parameters
    ----------
    data : pl.DataFrame
        A data frame
    x : str | list | np.ndarray
        Column names to to move
    before : str, default=None
        A column in `data`
    after : str, default=None
        A column in `data`

    Returns
    -------
    pl.DataFrame
        A data frame with reordered columns.
    """
    assert before is None or after is None, \
        'One of `before` and `after` must be specified, but not both.'
    x = np.atleast_1d(x)
    columns = data.columns
    columns2 = [col for col in columns if col not in x]

    if before is None and after is None:
        return data.select(list(x) + columns2)
    if before is None:
        pos = list(columns2).index(after) + 1
    else:
        pos = list(columns2).index(before)
    return data.select(list(columns2[:pos]) + list(x) + list(columns2[pos:]))


def _check_convert_df(data: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """
    Internal function to check if `data` is a pandas or polars data frame. If a 
    pandas data frame is passed, it will be converted to a polars data frame and 
    all datetime columns will be converted to dates.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame

    Returns
    -------
    pl.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data).with_columns(
            cs.datetime().cast(pl.Date)
        )

    assert isinstance(data, pl.DataFrame), \
        '`data` must be a DataFrame'

    return data


def _check_missing_dates(x: pl.Series):
    assert x.is_not_null().all(), \
        f"Missing values are not allowed in the `{x.name}` column.\n" + \
        "Make sure all dates are in YYYY-MM-DD format."
