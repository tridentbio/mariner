"""
Authentication related exceptions
"""


class InvalidOAuthState(Exception):
    """Exception raised when there is an error during oauth authentication."""


class InvalidGithubCode(Exception):
    """Exception raised when the github token from the request is invalid
    i.e. it didn't came from github."""
