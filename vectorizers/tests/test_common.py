import pytest

from sklearn.utils.estimator_checks import check_estimator

import scipy.sparse
import numpy as np

from vectorizers import TokenCooccurrenceVectorizer
from vectorizers import NgramVectorizer
from vectorizers import DistributionVectorizer

from vectorizers._vectorizers import (
    harmonic_kernel,
    triangle_kernel,
    flat_kernel,
    ngrams_of,
)


token_data = [
    [1, 3, 1, 4, 2],
    [2, 1, 2, 3, 4, 1, 2, 1, 3, 2, 4],
    [4, 1, 1, 3, 2, 4, 2],
    [1, 2, 2, 1, 2, 1, 3, 4, 3, 2, 4],
    [3, 4, 2, 1, 3, 1, 4, 4, 1, 3, 2],
    [2, 1, 3, 1, 4, 4, 1, 4, 1, 3, 2, 4],
]

text_token_data = [
    ["foo", "pok", "foo", "wer", "bar"],
    ["bar", "foo", "bar", "pok", "wer", "foo", "bar", "foo", "pok", "bar", "wer"],
    ["wer", "foo", "foo", "pok", "bar", "wer", "bar"],
    ["foo", "bar", "bar", "foo", "bar", "foo", "pok", "wer", "pok", "bar", "wer"],
    ["pok", "wer", "bar", "foo", "pok", "foo", "wer", "wer", "foo", "pok", "bar"],
    [
        "bar",
        "foo",
        "pok",
        "foo",
        "wer",
        "wer",
        "foo",
        "wer",
        "foo",
        "pok",
        "bar",
        "wer",
    ],
]


mixed_token_data = [
    [1, "pok", 1, 3.1415, "bar"],
    ["bar", 1, "bar", "pok", 3.1415, 1, "bar", 1, "pok", "bar", 3.1415],
    [3.1415, 1, 1, "pok", "bar", 3.1415, "bar"],
    [1, "bar", "bar", 1, "bar", 1, "pok", 3.1415, "pok", "bar", 3.1415],
    ["pok", 3.1415, "bar", 1, "pok", 1, 3.1415, 3.1415, 1, "pok", "bar"],
    ["bar", 1, "pok", 1, 3.1415, 3.1415, 1, 3.1415, 1, "pok", "bar", 3.1415],
]


def test_harmonic_kernel():
    kernel = harmonic_kernel([0, 0, 0, 0])
    assert kernel[0] == 1.0
    assert kernel[-1] == 1.0 / 4.0
    assert kernel[1] == 1.0 / 2.0


def test_triangle_kernel():
    kernel = triangle_kernel([0, 0, 0, 0], 4)
    assert kernel[0] == 4.0
    assert kernel[-1] == 1.0
    assert kernel[1] == 3.0


def test_flat_kernel():
    kernel = flat_kernel([0] * np.random.randint(2, 10))
    assert np.all(kernel == 1.0)


def test_ngrams_of():
    for ngram_size in (1, 2, 4):
        tokens = np.random.randint(10, size=np.random.poisson(5 + ngram_size))
        ngrams = ngrams_of(tokens, ngram_size)
        assert len(ngrams) == len(tokens) - (ngram_size - 1)
        assert np.all(
            [ngrams[i][0] == tokens[i] for i in range(len(tokens) - (ngram_size - 1))]
        )
        assert np.all(
            [
                ngrams[i][-1] == tokens[i + (ngram_size - 1)]
                for i in range(len(tokens) - (ngram_size - 1))
            ]
        )


def test_token_cooccurrence_vectorizer_basic():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(window_args=(1,))
    result = vectorizer.fit_transform(token_data)
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_text():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(window_args=(1,))
    result = vectorizer.fit_transform(text_token_data)
    assert result[1, 2] == 8
    assert result[0, 1] == 6


def test_token_cooccurrence_vectorizer_mixed():
    vectorizer = TokenCooccurrenceVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit_transform(mixed_token_data)


def test_ngram_vectorizer_basic():
    vectorizer = NgramVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)


def test_ngram_vectorizer_text():
    vectorizer = NgramVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)


def test_ngram_vectorizer_mixed():
    vectorizer = NgramVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit_transform(mixed_token_data)
