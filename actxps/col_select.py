import pandas as pd


def col_starts_with(data: pd.DataFrame,
                    pat: str,
                    **kwargs):
    """
    Return a list of column names that start with a given pattern.

    Parameters
    ----------
    data : pd.DataFrame
        A data frame
    pat : str
        A string pattern
    kwargs
        Additional arguments passed to pd.Series.str.contains

    Returns
    ----------
    List
    
    See Also
    ----------
    col_ends_with, col_contains
    
    Examples
    ----------
    ```{python}
    import actxps as xp
    dat = xp.load_toy_census()
    xp.col_starts_with(dat, 'pol')
    ```
    """
    return list(data.columns[data.columns.str.startswith(pat, **kwargs)])


def col_ends_with(data: pd.DataFrame,
                  pat: str,
                  **kwargs):
    """
    Return a list of column names that end with a given pattern.

    Parameters
    ----------
    data : pd.DataFrame
        A data frame
    pat : str
        A string pattern
    kwargs 
        Additional arguments passed to pd.Series.str.contains

    Returns
    ----------
    List
    
    See Also
    ----------
    col_starts_with, col_contains

    
    Examples
    ----------
    ```{python}
    import actxps as xp
    dat = xp.load_toy_census()
    xp.col_ends_with(dat, 'date')
    ```

    """
    return list(data.columns[data.columns.str.endswith(pat, **kwargs)])


def col_contains(data: pd.DataFrame,
                 pat: str,
                 **kwargs):
    """
    Return a list of column names that match a regular expression.

    Parameters
    ----------
    data : pd.DataFrame
        A data frame
    pat : str
        A string pattern
    kwargs 
        Additional arguments passed to pd.Series.str.contains

    Returns
    ----------
    List
    
    See Also
    ----------
    col_starts_with, col_ends_with
    
    Examples
    ----------
    ```{python}
    import actxps as xp
    dat = xp.load_toy_census()
    xp.col_contains(dat, 'at')
    ```

    """
    return list(data.columns[data.columns.str.contains(pat, **kwargs)])
