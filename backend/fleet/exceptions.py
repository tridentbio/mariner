"""
Exceptions for the package.
"""


class FitError(Exception):
    """
    Exception raised when a fit fails.
    """


class LayerForwardError(Exception):
    """
    Exception raised when a layer's forward pass fails.
    """
