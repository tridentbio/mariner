"""
Dataset related exceptions
"""


class InvalidCategoricalColumn(Exception):
    """Invalid categorical column exception"""


class DatasetAlreadyExists(Exception):
    """Dataset already exists exception"""


class DatasetNotFound(Exception):
    """Dataset not found exception"""
