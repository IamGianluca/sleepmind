import re

import numpy as np

from sleepmind.base import BaseTransformer


class RegexExtractor(BaseTransformer):

    def __init__(self, pattern, else_all=False):
        """Use regular expression to extract part of a string.

        Args:
            pattern (str): A valid regular expression.
            else_all (bool): Whether return the entire string in case nothing
                was matched.
        """
        self.pattern = pattern
        self.else_all = else_all

    def transform(self, X):
        """Extract text from array of documents.

        Args:
            X (np.array): shape (n_samples, 1). The documents.

        Returns:
            (np.array) shape (n_samples, 1). The matches.
        """
        results = []
        for doc in X:
            if len(doc) > 0:
                match = re.findall(self.pattern, doc[0].lower())
                if len(match) > 0:  # regex matched
                    result = [match[0][1]]
                else:  # regex didn't match anything
                    if self.else_all:
                        result = [doc[0].lower()]
                    else:
                        result = []
            else:  # empty document
                result = []
            results.append(result)

        return np.array(results)


class RegexCounter(BaseTransformer):
    """Count the number of occurrences of a regular expression in a text."""

    def __init__(self, pattern=''):
        self.pattern = pattern

    def transform(self, X):
        results = []
        for doc in X:
            match = re.findall(self.pattern, doc[0].lower())
            results.append(len(match))
        return np.array(results)
