import numpy as np
import pandas as pd
import pytest

from sleepmind.preprocessing import CategoricalImputer
from pandas.util.testing import assert_frame_equal


@pytest.mark.parametrize(
    "df,missing,strategy,expected",
    [
        (
            pd.DataFrame(
                {"a": ["hi", np.NaN, "hi"], "b": [np.NaN, "dog", "dog"]}
            ),
            "NaN",
            "most_frequent",
            pd.DataFrame(
                {"a": ["hi", "hi", "hi"], "b": ["dog", "dog", "dog"]}
            ),
        )
    ],
)
def test_categorical_imputer(df, missing, strategy, expected):
    transformer = CategoricalImputer(missing_values=missing, strategy=strategy)
    result = transformer.fit_transform(X=df)
    assert_frame_equal(result, expected)


def test_multimode():
    """When column is multimodal, CategoricalImputer will sort the modes and
    choose the first element.
    """
    df = pd.DataFrame(
        {"a": ["hi", np.NaN, "bye"], "b": [np.NaN, "cat", "dog"]}
    )
    transformer = CategoricalImputer()
    result = transformer.fit_transform(df)
    assert_frame_equal(
        result,
        pd.DataFrame({"a": ["hi", "bye", "bye"], "b": ["cat", "cat", "dog"]}),
    )
