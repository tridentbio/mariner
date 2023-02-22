"""
Authentication related exceptions
"""


class InvalidOAuthState(Exception):
    pass


class InvalidGithubCode(Exception):
    pass
