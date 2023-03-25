# This module contains helper functions used by other modules

def document(docstring):
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
