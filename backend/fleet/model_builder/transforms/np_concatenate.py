import numpy as np
from sklearn.base import TransformerMixin


class NpConcatenate(TransformerMixin):
    def fit(self, *_):
        """
        Necessary because this method is called in all transformers.
        """
        return self

    def transform(self, *args):
        """
        Performs the numpy concatenation in all inputs.
        """
        return np.concatenate(args, axis=-1)

    fit_transform = transform  # Necessary because this method is called in all transformers.
