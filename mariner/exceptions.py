class UserNotFound(Exception):
    pass


class UserAlreadyExists(Exception):
    pass


class InvalidCategoricalColumn(Exception):
    pass


class NotCreatorOfDataset(Exception):
    pass


class DatasetAlreadyExists(Exception):
    pass


class DatasetNotFound(Exception):
    pass


class ModelVersionNotFound(Exception):
    pass


class ModelNotFound(Exception):
    pass


class ModelNameAlreadyUsed(Exception):
    pass


class ExperimentNotFound(Exception):
    pass
