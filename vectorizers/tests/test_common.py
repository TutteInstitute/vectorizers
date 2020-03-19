import pytest

from sklearn.utils.estimator_checks import check_estimator

import scipy.sparse

from vectorizers import TokenCooccurrenceVectorizer
from vectorizers import NgramVectorizer
from vectorizers import DistributionVectorizer
#
#
# @pytest.mark.parametrize(
#     "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
# )
# def test_all_estimators(Estimator):
#     return check_estimator(Estimator)

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
    ["foo", "bar", "bar", "foo", "bar", "foo", "pok", "wer","pok", "bar", "wer"],
    ["pok", "wer", "bar", "foo", "pok", "foo", "wer", "wer", "foo", "pok", "bar"],
    ["bar", "foo", "pok", "foo", "wer", "wer", "foo", "wer", "foo", "pok", "bar", "wer"],
]


mixed_token_data = [
    [1, "pok", 1, 3.1415, "bar"],
    ["bar", 1, "bar", "pok", 3.1415, 1, "bar", 1, "pok", "bar", 3.1415],
    [3.1415, 1, 1, "pok", "bar", 3.1415, "bar"],
    [1, "bar", "bar", 1, "bar", 1, "pok", 3.1415,"pok", "bar", 3.1415],
    ["pok", 3.1415, "bar", 1, "pok", 1, 3.1415, 3.1415, 1, "pok", "bar"],
    ["bar", 1, "pok", 1, 3.1415, 3.1415, 1, 3.1415, 1, "pok", "bar", 3.1415],
]


def test_token_cooccurrence_vectorizer_basic():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)


def test_token_cooccurrence_vectorizer_text():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)

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