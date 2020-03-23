import pytest

import numpy as np

from vectorizers.distances import hellinger, sparse_hellinger


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
