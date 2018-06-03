import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
)
from sleepmind.preprocessing import ModifiedLabelEncoder


INPUT_DATA = ['cat', 'dog', 'dolphin', 'dog']

def test_modified_label_encoder():
    encoder = ModifiedLabelEncoder()
    result = encoder.fit_transform(INPUT_DATA)
    assert_array_equal(
        result,
        np.array([[0],
                  [1],
                  [2],
                  [1]])
    )


def test_onehotencoder_fail():
    """OneHotEncoder cannot take categorical data."""
    encoder = OneHotEncoder(sparse=False)
    with pytest.raises(ValueError):
        encoder.fit_transform(INPUT_DATA)


# TODO: fix this test
def test_labelencoder_onehotencoder():
    """LabelEncoder output 1-D array, and OneHotEncoder doesn't take as input
    data in such format.
    """
    # pipe = Pipeline([
        # ('le', LabelEncoder()),
        # ('ohe', OneHotEncoder()),
    # ])
    # with pytest.raises(ValueError):
        # pipe.fit_transform(INPUT_DATA)


def test_categoricalencoder_not_available():
    """CategoricalEncoder should make ModifiedLabelEncoder redundant. This new
    class will become available with sklearn 0.20.

    Notes: As soon as this tool becomes available, deprecate
        ModifiedLabelEncoder.
    """
    with pytest.raises(ImportError):
        from sklearn.preprocessing import CategoricalEncoder


def test_modified_label_encoder_pipeline():
    """ModifiedLabelEncoder can be used in combination with OneHotEncoder to
    transform categorical features.
    """
    pipe = Pipeline([
        ('le', ModifiedLabelEncoder()),
        ('ohe', OneHotEncoder(sparse=False))
    ])
    result = pipe.fit_transform(INPUT_DATA)
    assert_array_equal(
        result,
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 1, 0]])
    )
