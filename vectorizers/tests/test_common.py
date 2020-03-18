import pytest

from sklearn.utils.estimator_checks import check_estimator

import scipy.sparse

from vectorizers import TokenCooccurrenceVectorizer
from vectorizers import NgramVectorizer
# from vectorizers import TemplateClassifier
# from vectorizers import TemplateTransformer
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
    [ 1, 2, 2, 1, 2, 1, 3, 4,3, 2, 4],
    [3, 4, 2, 1, 3, 1, 4, 4, 1, 3, 2],
    [2, 1, 3, 1, 4, 4, 1, 4, 1, 3, 2, 4],
]


def test_token_cooccurrence_vectorizer_basic():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)

def test_ngram_vectorizer_basic():
    vectorizer = NgramVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)