# This module contains helper functions used by other modules
import numpy as np
import pandas as pd
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
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
from great_tables import style, loc, from_column, GT
from warnings import warn
from scipy.stats import norm
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
        data = xp_obj.data.copy()
    else:
        data = alt_data
        groups.insert(group_insert, 'Series')

    if groups == []:
        groups = ["All"]
        data["All"] = ""

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
            if all(np.isin(y_min_max, data.columns)):
                p = p + geom_errorbar(aes(ymin=y_min_max[0],
                                          ymax=y_min_max[1]))
            else:
                warn("Confidence intervals are not available for the " +
                     "selected y-variable.")

    if facets is None:
        return p
    else:
        return p + facet_wrap(facets, scales=scales)


def _pivot_plot_special(xp_obj, piv_cols, values_to="Rate"):
    """
    This internal function is used to pivot `ExpStats` or `TrxStats` data frames
    before they're passed to special plotting functions.

    Parameters
    ----------
    xp_obj : ExpStats | TrxStats
        An experience summary xp_obj
    piv_cols : list
        A primary set of columns to pivot longer
    values_to : str, default="Rate"
        Name of the values column in the pivoted xp_obj.

    Returns
    -------
    pd.DataFrame
        A pivoted dataframe
    """

    data = xp_obj.data.copy()
    piv_cols = np.intersect1d(piv_cols, data.columns)
    id_cols = np.setdiff1d(data.columns, piv_cols)

    if not xp_obj.xp_params['conf_int']:
        data = data.melt(id_vars=id_cols, value_vars=piv_cols,
                         var_name='Series', value_name=values_to)
    else:
        extra_piv_cols = np.concatenate((piv_cols + "_upper",
                                         piv_cols + "_lower"))
        extra_piv_cols = np.intersect1d(extra_piv_cols, data.columns)
        id_cols = np.setdiff1d(id_cols, extra_piv_cols)
        piv_cols_rename = {x: f'{x}_{values_to}' for
                           x in data.columns if
                           x in piv_cols}

        data = (data.rename(columns=piv_cols_rename).
                melt(id_vars=id_cols,
                     value_vars=list(piv_cols_rename.values()
                                     ) + list(extra_piv_cols),
                     var_name='Series', value_name=values_to))
        data[['Series', 'val_type']] = \
            data.Series.str.rsplit("_", expand=True, n=1)
        data = (data.pivot(index=list(id_cols) + ['Series'],
                           columns='val_type',
                           values=values_to).
                reset_index().
                rename(columns={'lower': values_to + '_lower',
                                'upper': values_to + '_upper'}))

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


# safe version of normal ppf when standard deviation is zero
def _qnorm(p, mean = 0, sd = 1):
    """
    Internal function for the inverse cumulative normal distribution
    that returns the mean when the standard deviation is zero.

    Parameters
    ----------
    p : np.ndarray
        A vector of probabilities
    mean : np.ndarray
        A vector of means
    sd : np.ndarray
        A vector of standard deviations

    Returns
    -------
    np.ndarray
        A vector of quantiles
    """
    sd = np.maximum(sd, 1E-16)
    return norm.ppf(p, mean, sd)


def relocate(data: pd.DataFrame, 
             x: str | list | np.ndarray, 
             before: str=None, 
             after: str=None):
    """
    Reorder columns in a data frame
    
    Move the columns in `x` before or after a given column. If neither `before`
    or `after` are specified, `x` will be moved to the left. If both `before`
    and `after` are specified, an error is returned.

    Parameters
    ----------
    data : pd.DataFrame
        A data frame
    x : str | list | np.ndarray
        Column names to to move
    before : str, default=None
        A column in `data`
    after : str, default=None
        A column in `data`

    Returns
    -------
    pd.DataFrame
        A data frame with reordered columns.
    """
    assert before is None or after is None, \
        'One of `before` and `after` must be specified, but not both.'
    x = np.atleast_1d(x)
    columns = data.columns
    columns2 = columns[~columns.isin(x)]
    if before is None and after is None:
        return data[list(x) + list(columns2)]
    if before is None:
        pos = list(columns2).index(after) + 1
    else:
        pos = list(columns2).index(before)
    return data[list(columns2[:pos]) + list(x) + list(columns2[pos:])]