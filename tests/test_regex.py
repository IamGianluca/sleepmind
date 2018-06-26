import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sleepmind.feature_extraction.text import RegexExtractor
from sleepmind.feature_extraction.text import RegexCounter


@pytest.mark.parametrize(
    "pattern,docs,expected,else_all",
    [
        ("(results  |discussion  )(.*?)($)", [[]], [[]], False),
        (
            "(results  |discussion  )(.*?)($)",
            [["Results  Hello my name is Luca."]],
            [["hello my name is luca."]],
            True,
        ),
        (
            "(results  |discussion  )(.*?)($)",
            [["I don't want this.Results  END"]],
            [["end"]],
            True,
        ),
        (
            "(results)(.*?)($)",
            [
                ["Results  Hello my name is Luca."],
                ["Something that does not match"],
            ],
            [["  hello my name is luca."], ["something that does not match"]],
            True,
        ),
        (
            "(results)(.*?)($)",
            [
                ["Results  Hello my name is Luca."],
                ["Something that does not match"],
            ],
            [["  hello my name is luca."], []],
            False,
        ),
    ],
)
def test_regex_extractor(pattern, docs, expected, else_all):
    transformer = RegexExtractor(pattern=pattern, else_all=else_all)
    assert_array_equal(
        transformer.transform(np.array(docs)), np.array(expected)
    )


@pytest.mark.parametrize(
    "pattern,docs,expected",
    [
        (
            "brig",
            [["Brig just landed at Gatwick airport. Brig is here!!"]],
            [2],
        ),
        (
            "luca",
            [["gianluca wants to move to NY."], ["lu ca"], ["lucaaa"]],
            [1, 0, 1],
        ),
    ],
)
def test_regex_counter(pattern, docs, expected):
    transformer = RegexCounter(pattern=pattern)
    assert_array_equal(
        transformer.transform(np.array(docs)), np.array(expected)
    )
