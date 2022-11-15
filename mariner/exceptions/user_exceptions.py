class NotCreatorOwner(Exception):
    pass


class UserNotActive(Exception):
    pass


class UserNotSuperUser(Exception):
    pass


class UserNotFound(Exception):
    pass


class UserAlreadyExists(Exception):
    pass


class UserEmailNotAllowed(Exception):
    pass
