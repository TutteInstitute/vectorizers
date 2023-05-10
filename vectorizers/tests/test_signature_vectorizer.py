import pytest

from vectorizers import SignatureVectorizer

import numpy as np
iisignature = pytest.importorskip("iisignature")
import re

NUMPY_SHAPE_ERROR_MSG = """
Error: SignatureVectorizer expects numpy arrays to be of shape (num_samples x path_len x path_dim).
"""
LIST_SHAPE_ERROR_MSG = """
Error: Expecting list entries to be numpy arrays of shape (path_len x path_dim).
"""

# Check numpy and list vectorizers return the same output as iisignature
@pytest.mark.parametrize("truncation_level", [2, 3, 5])
@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("basepoint", [True, False])
def test_numpy_vs_list_vs_iisig(truncation_level, log, basepoint, seed=1):

    n_paths = 100
    path_len = 50
    path_dim = 5

    np.random.seed(seed)
    test_paths_list = [
        np.random.normal(size=(path_len, path_dim)) for i in range(n_paths)
    ]
    test_paths_numpy = np.array(test_paths_list)

    sigs_numpy = SignatureVectorizer(
        truncation_level=truncation_level, log=log, basepoint=basepoint
    ).fit_transform(test_paths_numpy)
    sigs_list = SignatureVectorizer(
        truncation_level=truncation_level, log=log, basepoint=basepoint
    ).fit_transform(test_paths_list)

    if basepoint:
        concat_shape = (test_paths_numpy.shape[0], 1, test_paths_numpy.shape[2])
        X = np.concatenate([np.zeros(shape=concat_shape), test_paths_numpy], axis=1)
    else:
        X = test_paths_numpy

    if log:
        s = iisignature.prepare(X.shape[-1], truncation_level)
        sigs_iisig = iisignature.logsig(X, s)
    else:
        sigs_iisig = iisignature.sig(X, truncation_level)
    assert np.all(np.isclose(sigs_numpy, sigs_list))
    assert np.all(np.isclose(sigs_list, sigs_iisig))


# Check bad initialisation returns appropriate error messages
def test_bad_init_params():
    with pytest.raises(
        AssertionError, match="Error: expecting int type for truncation_level."
    ):
        vectorizer = SignatureVectorizer(truncation_level="three")

    with pytest.raises(AssertionError, match="Error: expecting bool type for log."):
        vectorizer = SignatureVectorizer(log=1)

    with pytest.raises(
        AssertionError, match="Error: expecting bool type for basepoint"
    ):
        vectorizer = SignatureVectorizer(basepoint=np.zeros(10))


# Check bad fit returns appropriate error messages
def test_bad_fit_params():

    vectorizer = SignatureVectorizer()

    with pytest.raises(AssertionError, match=re.escape(NUMPY_SHAPE_ERROR_MSG)):
        vectorizer.fit(np.random.random(size=(2, 10, 3, 5)))

    with pytest.raises(AssertionError, match=re.escape(NUMPY_SHAPE_ERROR_MSG)):
        vectorizer.fit(np.random.random(size=(2, 10)))

    with pytest.raises(
        AssertionError, match="Error: Expecting numpy array or list of paths."
    ):
        vectorizer.fit("Not a list or numpy array")

    with pytest.raises(
        AssertionError, match="Error: Expecting list entries to be numpy arrays."
    ):
        vectorizer.fit(["List", "of", "nonsense"])

    with pytest.raises(AssertionError, match=re.escape(LIST_SHAPE_ERROR_MSG)):
        vectorizer.fit([np.random.random(size=(3, 10, 5))])


# Check bad transform returns appropriate error messages
def test_bad_transform_parameters():

    vectorizer = SignatureVectorizer()
    vectorizer.fit(np.random.random(size=(20, 50, 3)))
    with pytest.raises(AssertionError, match=re.escape(NUMPY_SHAPE_ERROR_MSG)):
        vectorizer.transform(np.random.random(size=(2, 10, 3, 5)))

    with pytest.raises(AssertionError, match=re.escape(NUMPY_SHAPE_ERROR_MSG)):
        vectorizer.transform(np.random.random(size=(2, 10)))

    with pytest.raises(
        AssertionError, match="Error: Expecting numpy array or list of paths."
    ):
        vectorizer.transform("Not a list or numpy array")

    with pytest.raises(
        AssertionError, match="Error: Expecting list entries to be numpy arrays."
    ):
        vectorizer.transform(["List", "of", "nonsense"])

    with pytest.raises(AssertionError, match=re.escape(LIST_SHAPE_ERROR_MSG)):
        vectorizer.transform([np.random.random(size=(3, 10, 5))])

    # Mismatch from fit shape
    with pytest.raises(AssertionError, match="Error: Expecting path_dim to be"):
        vectorizer.transform([np.random.random(size=(50, 5))])
    with pytest.raises(AssertionError, match="Error: Expecting path_dim to be "):
        vectorizer.transform(np.random.random(size=(30, 50, 5)))
    with pytest.raises(
        AssertionError, match="Error: Not all paths share the same dimension."
    ):
        vectorizer.transform(
            [
                np.random.random(size=(10, 3)),
                np.random.random(size=(10, 3)),
                np.random.random(size=(10, 5)),
                np.random.random(size=(10, 3)),
            ]
        )
