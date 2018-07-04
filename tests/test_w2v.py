import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors
from numpy.testing import assert_array_equal
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sleepmind.feature_extraction.text import Word2VecEncoder


@pytest.fixture
def fake_pretrained_model(monkeypatch):
    def mocked_return(*args, **kwargs):
        return {
            'cat': np.zeros(300),
            'dog': np.ones(300),
            'male': np.array([5] * 300),
            'female': np.array([-1]* 300),
        }
    monkeypatch.setattr(KeyedVectors, 'load_word2vec_format', mocked_return)
    return


def test_fitted(fake_pretrained_model):
    # given
    new_data = np.array([['dog'], ['cat'], ['dog cat'], ['elephant']])
    encoder = Word2VecEncoder()

    # when
    with pytest.raises(ValueError):
        result = encoder.transform(X=new_data)


def test_w2v_multi_columns(fake_pretrained_model):
    # given
    new_data = pd.DataFrame({
        'animal': ['dog', 'cat', 'dog'],
        'gender': ['female', 'female', 'male'],
    })
    encoder = Word2VecEncoder()

    # when
    result = encoder.fit_transform(X=new_data)

    # then
    assert_array_equal(
        result,
        np.array(
            [[1] * 300 + [-1] * 300,
             [0] * 300 + [-1] * 300,
             [1] * 300 + [5] * 300,]
        )
    )

def test_w2v(fake_pretrained_model):
    # given
    new_data = np.array([['dog'], ['cat'], ['dog cat'], ['elephant']])
    encoder = Word2VecEncoder()

    # when
    result = encoder.fit_transform(X=new_data)

    # then
    assert_array_equal(
        result,
        np.array(
            [[1] * 300,
             [0] * 300,
             [1] * 300,
             [0.5] * 300]  # words not in the vocabulary should be the as the mean of the h
        )
    )


def test_w2v_pipeline(fake_pretrained_model):
    # given
    train = pd.DataFrame({'animal': ['dog', 'cat']})
    label = pd.DataFrame({'label': [1, 0]})
    pipe = make_column_transformer(
        (['animal'], make_pipeline(
            Word2VecEncoder())),
        remainder='drop',
    )

    # when
    result = pipe.fit_transform(X=train, y=label)

    # then
    assert_array_equal(
        result,
        np.array(
            [[1] * 300,
             [0] * 300]
        ),
    )
