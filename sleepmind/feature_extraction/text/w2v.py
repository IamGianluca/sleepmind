import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


class Word2VecEncoder:

    def __init__(self):
        self.w2v = KeyedVectors.load_word2vec_format(
            './GoogleNews-vectors-negative300.bin',
            binary=True,
        )
        self.vector_length = 300

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Args:
            X (array-like, sparse matrix): Input data.
                Shape (n_samples, n_features)
            y (array-like): Target vector. Shape (n_samples, )
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        try:
            n_rows, n_cols = X.shape
        except ValueError:
            n_rows, n_cols = X.shape[0], 1

        values = []
        for row in range(n_rows):
            try:
                new_row = [self.w2v[word] for word in X[row, :]]
            except ValueError:
                new_row = [self.w2v[word] for word in X[row]]
            values.append(new_row)
        return np.array(values).reshape(-1, self.vector_length)
