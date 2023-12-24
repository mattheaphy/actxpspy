# This module contains helper functions used by other modules
import numpy as np
from plotnine import (
    ggplot,
    geom_point,
    geom_line,
    geom_col,
    aes,
    facet_wrap,
    scale_y_continuous,
    scale_color_manual,
    scale_fill_manual)
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


def _plot_experience(xp_obj, x, y, color, mapping, scales,
                     geoms, y_labels, facets, alt_data=None):
    """
    This helper function is used by `ExpStats.plot()` and `TrxStats.plot()`. 
    It is not meant to be called directly.
    """

    from actxps.trx_stats import TrxStats
    from actxps.exp_stats import ExpStats

    assert isinstance(xp_obj, TrxStats | ExpStats)

    if alt_data is None:
        data = xp_obj.data.copy()
    else:
        data = alt_data

    groups = xp_obj.groups
    if groups is None or groups == []:
        groups = ["All"]
        data["All"] = ""

    def auto_aes(var, default, if_none):
        if (var is None):
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
        color = auto_aes(color, 2, None)
        if color is None:
            mapping = aes(x, y)
        else:
            mapping = aes(x, y,
                          color=color, fill=color, group=color)

    if facets is None:
        facets = groups[2:]
        if len(facets) == 0:
            facets = None

    p = (ggplot(data, mapping) +
         scale_y_continuous(labels=y_labels))

    global _use_default_colors
    if _use_default_colors:
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
    values_to : str, default="Rate
        Name of the values column in the pivoted xp_obj.

    Returns
    -------
    pd.DataFrame
        A pivoted dataframe
    """

    xp_params = xp_obj.xp_params
    data = xp_obj.data.copy()
    piv_cols = np.intersect1d(piv_cols, data.columns)
    id_cols = np.setdiff1d(data.columns, piv_cols)

    if not xp_params['conf_int']:
        data = data.melt(id_vars=id_cols, value_vars=piv_cols,
                         var_name='Series', value_name=values_to)
    else:
        pass
        # TODO - confidence intervals

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
