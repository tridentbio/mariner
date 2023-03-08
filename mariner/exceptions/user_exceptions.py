"""
User related exceptions
"""


class NotCreatorOwner(Exception):
    """Exception raised when user is not allowed for an operation because of
    not being the owner of that resource"""

    pass


class UserNotActive(Exception):
    """Raised when the user from request is not active anymore"""

    pass


class UserNotSuperUser(Exception):
    """Raised when the user is not a super user and tries to perform
    a super user action

    .. deprecated:: 0.1.0
        The application has no super user actions
    """

    pass


class UserNotFound(Exception):
    """Raised when the application can't proceed because of a user not
    being found"""

    pass


class UserAlreadyExists(Exception):
    """Raised when the account creation includes unique information
    (.e.g, email) that's already in the app's database"""

    pass


class UserEmailNotAllowed(Exception):
    """Raised when the email of a request is not allowed"""

    pass
