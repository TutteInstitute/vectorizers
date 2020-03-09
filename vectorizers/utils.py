import numpy as np
import numba
import scipy.stats
import itertools
from collections.abc import Iterable
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from typing import Union, Sequence, AnyStr

from warnings import warn


def flatten(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if type(list_of_seq[0]) in (list, tuple):
        return tuple(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq


def gmm_component_likelihood(
    component_mean: np.ndarray, component_covar: np.ndarray, diagram: np.ndarray
) -> np.ndarray:
    """Generate the vector of likelihoods of observing points in a diagram
    for a single gmm components (i.e. a single Gaussian). That is, evaluate
    the given Gaussian PDF at all points in the diagram.

    Parameters
    ----------
    component_mean: array of shape (2,)
        The mean of the Gaussian

    component_covar: array of shape (2,2)
        The covariance matrix of the Gaussian

    diagram: array of shape (n_top_features, 2)
        The persistence diagram

    Returns
    -------
    likelihoods: array of shape (n_top_features,)
        The likelihood of observing each topological feature in the diagram
        under the provided Gaussian
    """
    return scipy.stats.multivariate_normal.pdf(
        diagram, mean=component_mean, cov=component_covar,
    )


def vectorize_diagram(diagram: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """Given a diagram and a Guassian Mixture Model, produce the vectorized
    representation of the diagram as a vector of weights associated to
    each component of the GMM.

    Parameters
    ----------
    diagram: array of shape (n_top_features, 2)
        The persistence diagram to be vectorized

    gmm: sklearn.mixture.GaussianMixture
        The Gaussian Mixture Model to use for vectorization

    Returns
    -------
    vect: array of shape (gmm.n_components,)
        The vector representation of the persistence diagram
    """
    interim_matrix = np.zeros((gmm.n_components, diagram.shape[0]))
    for i in range(interim_matrix.shape[0]):
        interim_matrix[i] = gmm_component_likelihood(
            gmm.means_[i], gmm.covariances_[i], diagram
        )
    normalize(interim_matrix, norm="l1", axis=0, copy=False)
    return interim_matrix.sum(axis=1)


@numba.njit()
def mat_sqrt(mat: np.ndarray) -> np.ndarray:
    """Closed form solution for the square root of a 2x2 matrix

    Parameters
    ----------
    mat: array of shape (2,2)
        The matrix to take the square root of

    Returns
    -------
    root: array of shape (2,2)
        The matrix such that root * root == mat (up to precision)
    """
    result = mat.copy()
    s = np.sqrt(mat[0, 0] * mat[1, 1] - mat[1, 0] * mat[0, 1])
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2.0 * s)
    result[0, 0] += s
    result[1, 1] += s
    result /= t
    return result


@numba.njit()
def wasserstein2_gaussian(
    m1: np.ndarray, C1: np.ndarray, m2: np.ndarray, C2: np.ndarray
) -> float:
    """Compute the Wasserstein_2 distance between two 2D Gaussians. This can be
    computed via the closed form formula:

    $$W_{2} (\mu_1, \mu_2)^2 = \| m_1 - m_2 \|_2^2 + \mathop{\mathrm{trace}} \bigl( C_1 + C_2 - 2 \bigl( C_2^{1/2} C_1 C_2^{1/2} \bigr)^{1/2} \bigr)$$

    Parameters
    ----------
    m1: array of shape (2,)
        Mean of the first Gaussian

    C1: array of shape (2,2)
        Covariance matrix of the first Gaussian

    m1: array of shape (2,)
        Mean of the second Gaussian

    C2: array of shape (2,2)
        Covariance matrix of the second Gaussian

    Returns
    -------
    dist: float
        The Wasserstein_2 distance between the two Gaussians
    """
    result = np.sum((m1 - m2) ** 2)
    sqrt_C2 = np.ascontiguousarray(mat_sqrt(C2))
    prod_matrix = sqrt_C2 @ C1 @ sqrt_C2
    sqrt_prod_matrix = mat_sqrt(prod_matrix)
    correction_matrix = C1 + C2 - 2 * sqrt_prod_matrix
    result += correction_matrix[0, 0] + correction_matrix[1, 1]
    return np.sqrt(np.maximum(result, 0))


@numba.njit()
def pairwise_gaussian_ground_distance(
    means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Compute pairwise distances between a list of Gaussians. This can be
    used as the ground distance for an earth-mover distance computation on
    vectorized persistence diagrams.

    Parameters
    ----------
    means: array of shape (n_gaussians, 2)
        The means for the Gaussians

    covariances: array of shape (n_gaussians, 2, 2)
        The covariance matrrices of the Gaussians

    Returns
    -------
    dist_matrix: array of shape (n_gaussians, n_gaussians)
        The pairwise Wasserstein_2 distance between the Gaussians
    """
    n_components = means.shape[0]

    result = np.zeros((n_components, n_components), dtype=np.float32)
    for i in range(n_components):
        for j in range(i + 1, n_components):
            result[i, j] = wasserstein2_gaussian(
                means[i], covariances[i], means[j], covariances[j]
            )
            result[j, i] = result[i, j]

    return result
