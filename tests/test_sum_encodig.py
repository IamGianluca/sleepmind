import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sleepmind.preprocessing import SumEncoder


def test_one_level():
    # given
    train = pd.DataFrame({
        'animal': ['dog', 'dog', 'dog'],
        'label': [1, 0, 2],
    })
    transformer = SumEncoder().fit(X=train.animal, y=train.label)

    # when
    test = pd.DataFrame({
        'animal': ['dog', 'dog', 'dog']
    })
    result = transformer.transform(X=test.animal)

    # then
    assert_array_almost_equal(
        result,
        np.array([[1], [1], [1]]),
        decimal=4,
    )


def test_sum_encoding_fit_transform_and_new_level():
    # given
    train = pd.DataFrame({
        'animal': ['dog', 'dog', 'cat', 'cat', 'cat'],
        'label': [1, 0, 1, 1, 0],
    })
    transformer = SumEncoder().fit(X=train.animal, y=train.label)

    # when
    test = pd.DataFrame({
        'animal': ['dog', 'cat', 'elephant']
    })
    result = transformer.transform(X=test.animal)

    # then
    assert_array_almost_equal(
        result,
        np.array([[0.75], [1.3333], [1]]),
        decimal=4,
    )


def test_sum_encoding_multiple_columns():
    # given
    train = pd.DataFrame({
        'animal': ['dog', 'dog', 'cat', 'cat', 'cat'],
        'age': ['old', 'young', 'middle', 'young', 'old'],
        'label': [1, 0, 1, 1, 0],
    })
    transformer = SumEncoder().fit(
        X=train.drop(['label'], axis=1),
        y=train.label
    )

    # when
    test = pd.DataFrame({
        'animal': ['dog', 'cat', 'elephant'],
        'age': ['old', 'young', 'infant'],
    })
    result = transformer.transform(X=test)

    # then
    assert_array_almost_equal(
        result,
        np.array([[0.75, 0.75], [1.3333, 0.75], [1, 1]]),
        decimal=4,
    )


def test_sum_encoding_pipeline():
    # given
    train = pd.DataFrame({'animal': ['dog', 'dog', 'cat', 'cat', 'cat']})
    label = pd.DataFrame({'label': [1, 0, 1, 1, 0]})
    pipe = make_column_transformer(
        (['animal'], make_pipeline(
            SumEncoder())),
        remainder='drop',
    )

    # when
    result = pipe.fit_transform(X=train, y=label)

    # then
    assert_array_almost_equal(
        result,
        np.array([[0.75], [0.75], [1.3333], [1.3333], [1.3333]]),
        decimal=4,
    )


def test_sum_encoding_pickle():
    # given
    train = pd.DataFrame(data={
        'animal': ['dog', 'dog', 'cat', 'cat', 'cat'],
        'label': [1, 0, 1, 1, 0],
    })
    transformer = SumEncoder().fit(X=train.animal, y=train.label)

    # when
    pickle_file = BytesIO(pickle.dumps(obj=transformer))
    unpickled_transformer = pickle.load(file=pickle_file)
    test = pd.DataFrame(data={
        'animal': ['dog', 'cat', 'elephant']
    })
    result = unpickled_transformer.transform(X=test.animal)

    # then
    assert_array_almost_equal(
        result,
        np.array([[0.75], [1.3333], [1]]),
        decimal=4,
    )
