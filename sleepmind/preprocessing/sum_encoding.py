from collections import defaultdict

import numpy as np
import pandas as pd

from sleepmind.base import BaseTransformer


class SumEncoder(BaseTransformer):

    def __init__(self):
        self.encoding = {}
        self.col_names = None

    def fit(self, X, y):
        """
        Args:
            X (array-like, sparse matrix): Input data.
                Shape (n_samples, n_features)
            y (array-like): Target vector. Shape (n_samples, )
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        try:
            n_cols = X.shape[1]
        except IndexError:
            n_cols = 1
            X = X.reshape(-1, n_cols)

        self.col_names = list(range(n_cols))
        store = defaultdict(get_defaultdict_of_list)
        for row, label in zip(X, y):
            # if not n_cols > 1:
                # row = [row]
            for col, level in enumerate(row):
                store[col][level].append(label)

        self.encoding = defaultdict(get_defaultdict_of_list)
        for col in self.col_names:
            for level, y in store[col].items():
                numerator = np.nanmean(y)
                others = []
                for other_level, other_y in store[col].items():
                    if other_level != level:
                        others.extend(other_y)
                if others == []:  # cardinality 1
                    self.encoding[col][level] = 1
                else:
                    denominator = np.nanmean(others)
                    self.encoding[col][level] = numerator / denominator
        return self

    def transform(self, X):
        """
        Args:
            X (array-like, sparse matrix): Input data.
                Shape (n_samples, n_features)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        try:
            n_rows, n_cols = X.shape
        except ValueError:
            n_rows, n_cols = X.shape[0], 1

        values = []
        for row in range(n_rows):
            try:
                new_row = [
                    self.encoding[col].get(level, 1)
                    for col, level in enumerate(X[row, :])
                ]
            except ValueError:
                new_row = [self.encoding[0].get(X[row], 1)]
            values.append(new_row)
        return np.array(values).reshape(-1, n_cols)


def get_defaultdict_of_list():
    return defaultdict(list)
