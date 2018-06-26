import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.compose import make_column_transformer

from sleepmind.preprocessing import SumEncoder


@pytest.mark.parametrize('strategy,expected', [
    ('sum', np.array([1, 6, 6, 6, 1]).reshape(-1, 1)),
    ('mean', np.array([1, 2, 2, 2, 1]).reshape(-1, 1)),
    ('median', np.array([1, 2, 2, 2, 1]).reshape(-1, 1)),
    ('most_frequent', np.array([1, 1, 1, 1, 1]).reshape(-1, 1)),
])
def test_sum_encoding(strategy, expected):
    # given
    X = np.array(['a', 'b', 'b', 'b', 'a'])
    y = np.array([1, 1, 2, 3, np.NaN])

    # when
    transformer = SumEncoder(strategy=strategy).fit(X=X, y=y)
    result = transformer.transform(X=X)

    # then
    assert_array_equal(result, expected)


def test_multiple_columns():
    # given
    data = pd.DataFrame({
        'first': ['a', 'a', 'b', 'b', 'b'],
        'second': ['dog', 'cat', 'cat', 'dog', 'dog'],
        'target': [1, 1, 2, 2, 3]})
    preprocess = make_column_transformer(
        (['first', 'second'], SumEncoder())
    )

    # when
    results = preprocess.fit_transform(
        X=data.drop('target', axis=1),
        y=data.target
    )

    # then
    assert_array_equal(
        results,
        np.array([[2, 6],
                  [2, 3],
                  [7, 3],
                  [7, 6],
                  [7, 6]]))
