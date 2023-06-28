import numpy as np


class NpConcatenate:
    def fit(self, *_):
        """
        Necessary because this method is called in all transformers.
        """
        return self

    def transform(self, *args):
        """
        Performs the numpy concatenation in all inputs.
        """
        return np.concatenate(*args, axis=-1)

    fit_transform = transform  # Necessary because this method is called in all transformers.
    __call__ = transform  # Necessary because a transform needs to be callable.
