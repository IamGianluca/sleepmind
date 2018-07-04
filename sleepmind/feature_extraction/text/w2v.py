import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from sleepmind.base import BaseTransformer


class Word2VecEncoder(BaseTransformer):

    def __init__(self):
        self.w2v = KeyedVectors.load_word2vec_format(
            './GoogleNews-vectors-negative300.bin',
            binary=True,
        )
        self.fitted = False
        self.vector_length = 300

    def fit(self, X, y=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        try:
            n_rows, n_cols = X.shape
        except ValueError:
            n_rows, n_cols = X.shape[0], 1

        values = []
        for row_idx in range(n_rows):
            try:
                new_row = [self.get_vector(word) for word in X[row_idx, :]]
            except ValueError:
                new_row = [self.get_vector(word) for word in X[row_idx]]
            except KeyError:
                pass
            values.append(new_row)
        space = np.array(values).reshape(-1, self.vector_length)
        self.unknown = space.mean(axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        """
        Args:
            X (array-like, sparse matrix): Input data.
                Shape (n_samples, n_features)
            y (array-like): Target vector. Shape (n_samples, )
        """
        if not self.fitted:
            raise ValueError('Word2VecEncoder transformer need to be trained')
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        try:
            n_rows, n_cols = X.shape
        except ValueError:
            n_rows, n_cols = X.shape[0], 1

        values = []
        for row_idx in range(n_rows):
            try:
                new_row = [self.get_vector(phrase) for phrase in X[row_idx, :]]
            except ValueError:
                container = [self.get_vector(phrase) for phrase in X[row_idx]]
                new_row = [item for sub_list in container for item in sub_list]
            values.append(new_row)

        return np.array(values).reshape(-1, self.vector_length * n_cols)

    def get_vector(self, phrase):
        words = phrase.split(' ')
        v = np.zeros(self.vector_length)
        for word in words:
            try:
                v += self.w2v[word]
            except KeyError:
                if self.fitted:
                    v += self.unknown
        return v.tolist()
