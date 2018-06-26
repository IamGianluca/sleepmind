from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sleepmind.base import BaseTransformer


def most_frequent(a):
    return Counter(a).most_common()[0][0]


class SumEncoder(BaseTransformer):
    def __init__(self, strategy="sum"):
        self.strategy = strategy
        self.statistics = {}
        self.col_names = None

    def fit(self, X, y):
        """
        Args:
            X (array-like, sparse matrix): Input data.
                Shape (n_samples, n_features)
            y (array-like): Target vector. Shape (n_samples, )
        """
        allowed_strategies = ["sum", "mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Strategy {self.strategy} is not supported.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        n_cols = X.shape[0]
        self.col_names = list(range(n_cols))

        storage = defaultdict(list)
        for row, target in zip(X, y):
            for col, item in enumerate(row):
                storage[(col, item)].append(target)

        functions = {
            "sum": np.nansum,
            "mean": np.nanmean,
            "median": np.nanmedian,
            "most_frequent": most_frequent,
        }

        for key, values in storage.items():
            col, label = key
            self.statistics[(col, label)] = functions[self.strategy](values)
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
        # import ipdb; ipdb.set_trace()
        for row in range(n_rows):
            try:
                new_row = [
                    self.statistics[(col, item)]
                    for col, item in enumerate(X[row, :])
                ]
            except IndexError:
                new_row = [
                    self.statistics[(col, item)]
                    for col, item in enumerate(X[row])
                ]
            values.append(new_row)
        return np.array(values).reshape(-1, n_cols)
