import numpy as np
import numba
import scipy.stats
import scipy.sparse
import itertools
from collections.abc import Iterable
from collections import namedtuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, LabelBinarizer
from typing import Union, Sequence, AnyStr
from collections import Counter

from warnings import warn

CooArray = namedtuple("CooArray", ["row", "col", "val", "key", "ind", "min"])


@numba.njit(nogil=True, inline="always")
def coo_sum_duplicates(coo, kind):

    upper_lim = coo.ind[0]
    lower_lim = coo.min[0]

    perm = np.argsort(coo.key[lower_lim:upper_lim], kind=kind)

    coo.row[lower_lim:upper_lim] = coo.row[lower_lim:upper_lim][perm]
    coo.col[lower_lim:upper_lim] = coo.col[lower_lim:upper_lim][perm]
    coo.val[lower_lim:upper_lim] = coo.val[lower_lim:upper_lim][perm]
    coo.key[lower_lim:upper_lim] = coo.key[lower_lim:upper_lim][perm]

    sum_ind = lower_lim
    this_row = coo.row[lower_lim]
    this_col = coo.col[lower_lim]
    this_val = np.float32(0)
    this_key = coo.key[lower_lim]

    for i in range(lower_lim, upper_lim):
        if coo.key[i] == this_key:
            this_val += coo.val[i]
        else:
            coo.row[sum_ind] = this_row
            coo.col[sum_ind] = this_col
            coo.val[sum_ind] = this_val
            coo.key[sum_ind] = this_key
            this_row = coo.row[i]
            this_col = coo.col[i]
            this_val = coo.val[i]
            this_key = coo.key[i]
            sum_ind += 1

    if this_key != coo.key[upper_lim]:
        coo.row[sum_ind] = this_row
        coo.col[sum_ind] = this_col
        coo.val[sum_ind] = this_val
        coo.key[sum_ind] = this_key
        sum_ind += 1

    coo.ind[0] = sum_ind
    coo.min[0] = coo.ind[0]


@numba.njit(nogil=True, inline="always")
def coo_append(coo, tup):
    coo.row[coo.ind[0]] = tup[0]
    coo.col[coo.ind[0]] = tup[1]
    coo.val[coo.ind[0]] = tup[2]
    coo.key[coo.ind[0]] = tup[3]
    coo.ind[0] += 1

    if (coo.ind[0] - coo.min[0]) >= 1 << 18:
        coo_sum_duplicates(coo, kind="quicksort")
        if coo.key.shape[0] - coo.min[0] <= 1 << 18:
            coo.min[0] = 0.0
            coo_sum_duplicates(coo, kind="mergesort")
            if coo.ind[0] >= 0.95 * coo.key.shape[0]:
                print(
                    "The coo matrix array is near memory limit.  Increasing coo_max_bytes should increase performance."
                )

    if coo.ind[0] == coo.key.shape[0]:
        raise ValueError(
            f"The coo matrix array is over memory limit.  Increase coo_max_bytes to process data."
        )


def flatten(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if type(list_of_seq[0]) in (list, tuple, np.ndarray):
        return tuple(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq

def sparse_collapse(matrix, labels, sparse=True):
    """
    Groups the rows and columns of a matrix by the the labels array.
    The group by operation is summation.
    Parameters
    ----------
    matrix: sparse matrix of shape (n, n)
        a square matrix to group by
    labels: array of length (n)
        An array of the labels of the rows and columns to group by
    sparse: bool (default=True)
        Should I maintain sparsity?
    Returns
    -------
    matrix: sparse matrix of shape (unique_labels, unique_labels)
        This is the matrix of the summation of values between unique labels
    labels: array of length (unique_labels)
        This is the array of the labels of the rows and columns of our matrix.
    """
    if len(labels) == 0:
        return matrix, labels

    transformer = LabelBinarizer(sparse_output=sparse)
    trans = transformer.fit_transform(labels)
    if (trans.shape[1]) == 1:
        trans = trans.toarray()
        if len(transformer.classes_) == 1:
            trans ^= 1
        else:
            trans = np.hstack([trans ^ 1, trans])
        trans = scipy.sparse.csr_matrix(trans, dtype=np.float32, shape=trans.shape)
    result = trans.T @ matrix @ trans
    return result, transformer.classes_


def cast_tokens_to_strings(data):
    """Given token data (either an iterator of tokens, or an iterator of iterators
    of tokens) this will convert all the tokens to strings, so that all tokens
    are of a consistent homogeneous type.

    Parameters
    ----------
    data: iterator of tokens or iterator of iterators of tokens
        The token data to convert to string based tokens

    Returns
    -------
    new_data: iterator of tokens or iterator of iterators of tokens
        The token data in the same format as passed but with all tokens as strings.
    """
    result = []
    for item in data:
        if type(item) in (list, tuple, np.ndarray):
            result.append([str(x) for x in item])
        else:
            result.append(str(item))

    return result


def validate_homogeneous_token_types(data):
    """Validate that all tokens are of homogeneous type.

    Parameters
    ----------
    data: iterator of tokens or iterator of iterators of tokens
        The token data to convert to string based tokens

    Returns
    -------
    valid: True if valid; will raise an exception if tokens are heterogeneous.
    """
    types = Counter([type(x) for x in flatten(data)])
    if len(types) > 1:
        warn(f"Non-homogeneous token types encountered. Token type counts are: {types}")
        raise ValueError(
            "Heterogeneous token types are not supported -- please cast "
            "your tokens to a single type. You can use "
            '"X = vectorizers.cast_tokens_to_string(X)" to achieve '
            "this."
        )
    else:
        return True


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
        diagram,
        mean=component_mean,
        cov=component_covar,
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
