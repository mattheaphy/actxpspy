# This module contains helper functions used by other modules
from plotnine import (
    ggplot,
    geom_point,
    geom_line,
    geom_col,
    aes,
    facet_wrap,
    scale_y_continuous)


def document(docstring: str):
    """
    Decorator function factory for adding docstrings

    ## Parameters

    `docstring`: str
      A docstring to add to a function

    ## Returns

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

    ## Parameters

    `name`: str
        Argument name
    `x`: Any
        Argument value
    `allowed`: Any
        A list of allowed argument values

    ## Source

    This function is inspired by the R language's `arg.match()` and 
    `rlang::arg_match()` functions.
    """
    if x not in allowed:
        allowed = ", ".join([f'"{a}"' for a in allowed[:-1]]) + \
            f', or "{allowed[-1]}"'
        raise ValueError(f'`{name}` must be one of {allowed}. '
                         f'"{x}" is not allowed.')


def _plot_experience(object, x, y, color, mapping, scales,
                     geoms, y_labels, facets):
    """
    This helper function is used by `ExpStats.plot()` and `TrxStats.plot()`. 
    It is not meant to be called directly.
    """

    from actxps.trx_stats import TrxStats
    from actxps.exp_stats import ExpStats

    assert isinstance(object, TrxStats | ExpStats)

    data = object.data.copy()

    groups = object.groups
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

    arg_match("geoms", geoms, ["lines", "bars"])

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

    if geoms == "lines":
        p = p + geom_point() + geom_line()
    else:
        p = p + geom_col(position="dodge")

    if facets is None:
        return p
    else:
        return p + facet_wrap(facets, scales=scales)
