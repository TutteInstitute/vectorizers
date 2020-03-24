import pytest

import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize

from vectorizers.distances import hellinger, sparse_hellinger
from vectorizers.distances import total_variation, sparse_total_variation
from vectorizers.distances import (
    jensen_shannon_divergence,
    sparse_jensen_shannon_divergence,
)


def test_hellinger():
    assert hellinger(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert hellinger(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert hellinger(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert hellinger(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == 0.5411961001461969
    assert hellinger(np.array([0.1, 0.9]), np.array([1.0, 0.0])) == 0.8269052146305295


def test_sparse_hellinger():
    assert np.isclose(
        sparse_hellinger(
            np.array([7, 12]),
            np.array([0.0, 0.0]),
            np.array([8, 13]),
            np.array([0.0, 0.0]),
        ),
        0.0,
    )
    assert np.isclose(
        sparse_hellinger(
            np.array([7, 12]),
            np.array([0.0, 0.0]),
            np.array([8, 13]),
            np.array([1.0, 0.0]),
        ),
        1.0,
    )
    assert np.isclose(
        sparse_hellinger(
            np.array([7, 12]),
            np.array([0.5, 0.5]),
            np.array([7, 12]),
            np.array([0.5, 0.5]),
        ),
        0.0,
    )
    assert np.isclose(
        sparse_hellinger(
            np.array([7, 12]),
            np.array([0.5, 0.5]),
            np.array([7, 12]),
            np.array([1.0, 0.0]),
        ),
        0.5411961001461969,
    )
    assert np.isclose(
        sparse_hellinger(
            np.array([7, 12]),
            np.array([0.1, 0.9]),
            np.array([7, 12]),
            np.array([1.0, 0.0]),
        ),
        0.8269052146305295,
    )


# Test using inequalities with Hellinger distance from Wikipedia
# https://en.wikipedia.org/wiki/Hellinger_distance#Connection_with_the_statistical_distance
def test_total_variation():
    test_data = np.random.random(size=(10, 50))
    test_data = normalize(test_data, norm="l1")
    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            hd = hellinger(test_data[i], test_data[j])
            tvd = total_variation(test_data[i], test_data[j])
            assert hd ** 2 <= tvd
            assert tvd <= np.sqrt(2) * hd


# Test using inequalities with Hellinger distance from Wikipedia
# https://en.wikipedia.org/wiki/Hellinger_distance#Connection_with_the_statistical_distance
def test_sparse_total_variation():
    test_data = np.random.random(size=(10, 100))
    # sparsify
    test_data[test_data <= 0.5] = 0.0
    test_data = scipy.sparse.csr_matrix(test_data)
    test_data = normalize(test_data, norm="l1")

    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            hd = sparse_hellinger(
                test_data[i].indices,
                test_data[i].data,
                test_data[j].indices,
                test_data[j].data,
            )
            tvd = sparse_total_variation(
                test_data[i].indices,
                test_data[i].data,
                test_data[j].indices,
                test_data[j].data,
            )
            assert hd ** 2 <= tvd
            assert tvd <= np.sqrt(2) * hd


def test_jensen_shannon():
    test_data = np.random.random(size=(10, 50))
    test_data = normalize(test_data, norm="l1")
    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            m = (test_data[i] + test_data[j]) / 2.0
            p = test_data[i]
            q = test_data[j]
            d = (
                -np.sum(m * np.log(m))
                + (np.sum(p * np.log(p)) + np.sum(q * np.log(q))) / 2.0
            )
            assert np.isclose(d, jensen_shannon_divergence(p, q))


def test_sparse_jensen_shannon():
    test_data = np.random.random(size=(10, 100))
    # sparsify
    test_data[test_data <= 0.5] = 0.0
    sparse_test_data = scipy.sparse.csr_matrix(test_data)
    sparse_test_data = normalize(sparse_test_data, norm="l1")
    test_data = normalize(test_data, norm="l1")

    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            m = (test_data[i] + test_data[j]) / 2.0
            p = test_data[i]
            q = test_data[j]
            d = (
                -np.sum(m[m > 0] * np.log(m[m > 0]))
                + (
                    np.sum(p[p > 0] * np.log(p[p > 0]))
                    + np.sum(q[q > 0] * np.log(q[q > 0]))
                )
                / 2.0
            )
            assert np.isclose(
                d,
                sparse_jensen_shannon_divergence(
                    sparse_test_data[i].indices,
                    sparse_test_data[i].data,
                    sparse_test_data[j].indices,
                    sparse_test_data[j].data,
                ),
            )
