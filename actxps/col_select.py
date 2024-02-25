import polars as pl
import polars.selectors as cs


def col_starts_with(data: pl.DataFrame, pat: str):
    """
    Return a list of column names that start with a given pattern.

    Parameters
    ----------
    data : pl.DataFrame
        A data frame
    pat : str
        A string pattern

    Returns
    ----------
    List
    
    See Also
    ----------
    col_ends_with, col_matches
    
    Examples
    ----------
    ```{python}
    import actxps as xp
    dat = xp.load_toy_census()
    xp.col_starts_with(dat, 'pol')
    ```
    """
    return data.select(cs.starts_with(pat)).columns


def col_ends_with(data: pl.DataFrame, pat: str):
    """
    Return a list of column names that end with a given pattern.

    Parameters
    ----------
    data : pl.DataFrame
        A data frame
    pat : str
        A string pattern

    Returns
    ----------
    List
    
    See Also
    ----------
    col_starts_with, col_matches

    
    Examples
    ----------
    ```{python}
    import actxps as xp
    dat = xp.load_toy_census()
    xp.col_ends_with(dat, 'date')
    ```

    """
    return data.select(cs.ends_with(pat)).columns


def col_matches(data: pl.DataFrame, pat: str):
    """
    Return a list of column names that match a regular expression.

    Parameters
    ----------
    data : pl.DataFrame
        A data frame
    pat : str
        A string pattern

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
    xp.col_matches(dat, 'at')
    ```

    """
    return data.select(cs.matches(pat)).columns
