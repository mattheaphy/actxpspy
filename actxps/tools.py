# This module contains helper functions used by other modules

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
