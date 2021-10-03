import numpy as np
import numba
import pandas as pd
import scipy.stats
import scipy.sparse
import itertools
from collections.abc import Iterable
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize, LabelBinarizer
from collections import Counter
import re
from warnings import warn

import os

if "NUMBA_DISABLE_JIT" in os.environ and not os.environ["NUMBA_DISABLE_JIT"] in (
    0,
    "0",
):

    def to_fixed_tuple(iterable, size):
        return tuple(iterable)[:size]


else:
    from numba.np.unsafe.ndarray import to_fixed_tuple

from scipy.special import digamma


@numba.vectorize()
def digamma(x):
    if x <= 0:
        return -np.inf
    result = 0.0
    for i in range(x, 7):
        result -= 1 / x
        x += 1
    x -= 0.5
    xx = 1.0 / x
    xx2 = xx * xx
    xx4 = xx2 * xx2
    result += (
        np.log(x)
        + (1.0 / 24.0) * xx2
        - (7.0 / 960.0) * xx4
        + ((31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4)
    )
    return result


@numba.njit()
def dp_normalize(indptr, data, sums):
    data = digamma(data)
    sums = digamma(sums)
    for i, this_sum in enumerate(sums):
        for j in range(indptr[i], indptr[i + 1]):
            data[j] = np.exp(data[j] - this_sum)
    return data


@numba.njit()
def dp_normalize_vector(vec):
    data = digamma(vec)
    this_sum = digamma(np.sum(vec))
    return np.exp(data - this_sum)


@numba.njit()
def l1_normalize_vector(vec):
    return vec / np.sum(vec)


def dirichlet_process_normalize(X, axis=0, norm="l1"):
    sums = np.array(X.sum(axis=axis)).flatten()
    if axis == 0:
        X = X.tocsc()
        new_data = dp_normalize(X.indptr, X.data, sums)
        X.data = new_data
        return X.tocsr()
    else:
        X = X.tocsr()
        new_data = dp_normalize(X.indptr, X.data, sums)
        X.data = new_data
        return X.tocsr()


@numba.njit(nogil=True)
def pair_to_tuple(pair):
    return to_fixed_tuple(pair, 2)


def make_tuple_converter(tuple_size):
    @numba.njit(nogil=True)
    def tuple_converter(to_convert):
        return to_fixed_tuple(to_convert, tuple_size)

    return tuple_converter


def str_to_bytes(size_str):
    """Convert a string description of a memory size (e.g. 1GB, or 100M, or 20kB)
    into a number of bytes to be used in computations and memory size limits. This is
    not particularly fancy or robust, but is enough for most standard cases up to
    terabyte sizes.

    Parameters
    ----------
    size_str: str
        A description of a memory size. Support k, M, G, T and ki, Mi, Gi, Ti sizes
        with optional B/b characters appended, and possible space between the number
        and the unit.

    Returns
    -------
    bytes: int
        The memory size explicitly in bytes.
    """
    parse_match = re.match(r"(\d+\.?\d*)\s*([kMGT]?i?[Bb]?)$", size_str)
    if parse_match is None:
        raise ValueError(
            f"Invalid memory size string {size_str}; should be of the form '200M', '2G', etc."
        )

    if parse_match.group(2) in ("k", "kB", "kb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024))
    elif parse_match.group(2) in ("M", "MB", "Mb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 2))
    elif parse_match.group(2) in ("G", "GB", "Gb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 3))
    elif parse_match.group(2) in ("T", "TB", "Tb"):
        return int(np.ceil(float(parse_match.group(1)) * 1024 ** 4))
    elif parse_match.group(2) in ("ki", "kiB", "kib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000))
    elif parse_match.group(2) in ("Mi", "MiB", "Mib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 2))
    elif parse_match.group(2) in ("Gi", "GiB", "Gib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 3))
    elif parse_match.group(2) in ("Ti", "TiB", "Tib"):
        return int(np.ceil(float(parse_match.group(1)) * 1000 ** 4))
    else:
        return int(np.ceil(float(parse_match.group(1))))


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
    """Given a diagram and a Gaussian Mixture Model, produce the vectorized
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


def procrustes_align(
    e1: np.ndarray, e2: np.ndarray, scale_to: str = "both"
) -> np.ndarray:
    """Given two embeddings ``e1`` and ``e2`` attempt to align them
    via a combination of shift, uniform scaling, and orthogonal
    transformations.

    Note that ``e1`` and ``e2`` are assumed to be matched row by row
    with each other, and thus must have the same shape.

    Parameters
    ----------
    e1: ndarray
        An embedding with a row per sample.

    e2: ndarray
        An embedding with a row per sample.

    scale_to: string (optional, default="both")
        When scaling the results, scale to match either the
        first argument, the second argument, or both. Should
        be one of:
            * "first"
            * "second"
            * "both"

    Returns
    -------
    e1_aligned, e2_aligned: ndarray
        The aligned copies of embeddings ``e1`` and ``e2``
    """
    e1_shift = e1 - np.mean(e1, axis=0)
    e2_shift = e2 - np.mean(e2, axis=0)
    e1_scale_factor = np.sqrt(np.mean(e1_shift ** 2))
    e2_scale_factor = np.sqrt(np.mean(e2_shift ** 2))
    if scale_to == "first":
        rescale_factor = e1_scale_factor
    elif scale_to == "second":
        rescale_factor = e2_scale_factor
    elif scale_to == "both":
        rescale_factor = np.sqrt(e1_scale_factor * e2_scale_factor)
    else:
        raise ValueError(
            f"Invalid value {scale_to} for scale_to. scale_to should be one of 'first', 'second', or 'both'"
        )
    e1_scaled = e1_shift / e1_scale_factor
    e2_scaled = e2_shift / e2_scale_factor
    covariance = e2_scaled.T @ e1_scaled
    u, s, v = np.linalg.svd(covariance)
    rotation = u @ v
    return e1_scaled * rescale_factor, (e2_scaled @ rotation) * rescale_factor


def summarize_embedding(
    weight_matrix,
    column_index_dictionary,
    k=3,
    return_type="list",
    include_values=False,
    sep=",",
):
    """
    Summarizes each of the rows of a weight matrix via the k column names associated with the largest k values
    in each row.

    Parameters
    ----------
    weight_matrix: matrix (often sparse)
    column_index_dictionary: dict
        A dictionary mapping from the column numbers to their column names
    k: int (default 3)
        The number of column names with which to represent each row
    return_type: string (default: 'list')
        The type of summary you'd like to return this is chosen from ['list', 'string']
    include_values: bool (default: False)
        Should the values associated with the top k columns also be included in our summary?
    sep: string (default: ',')
        if the return type is string what separator should be used to create the string.

    Returns
    -------
    A list of short summaries of length weight_matrix.shape[0].
    By default each summary is a list of k column names.
    If include_values==True then a second list is also returned with the corresponding sorted values.
    If return_type=='string' then this list (or these lists) are packed together into a single string representation
    per row.
    """

    # Quick and dirty casting to a dense matrix to get started.
    # I'll likely need to juggle some csr data structures to do this right.
    # One thing I'll need to deal with is the variable number of non-zeros in each row may mean that the
    # there are less non-zeros than k in many of our rows.
    if include_values:
        values = []
    if scipy.sparse.issparse(weight_matrix):
        row_summary = []
        for row in weight_matrix:
            row_summary.append(
                [
                    column_index_dictionary[row.indices[x]]
                    for x in np.argsort(row.data)[::-1][:k]
                ]
            )
            if include_values:
                values.append([row.data[x] for x in np.argsort(row.data)[::-1][:k]])
    else:
        largest_indices = np.array(np.argsort(weight_matrix, axis=1))
        row_summary = [
            list(map(column_index_dictionary.get, x[::-1][:k])) for x in largest_indices
        ]
        if include_values:
            for i, row in enumerate(largest_indices):
                values.append([weight_matrix[i, x] for x in row[::-1][:k]])

    if include_values:
        if return_type == "string":
            if not isinstance(sep, str):
                raise ValueError("sep must be a string")
            summary = []
            for i in range(len(row_summary)):
                summary.append(
                    sep.join([f"{x}:{y}" for x, y in zip(row_summary[i], values[i])])
                )
            return summary
        else:
            return row_summary, values
    else:
        if return_type == "string":
            row_summary = [sep.join(x) for x in row_summary]
        return row_summary


def categorical_columns_to_list(data_frame, column_names):
    """
    Takes a data frame and a set of columns and represents each row a list of the appropriate non-empty columns
    of the form column_name:value respecting the order of the rows.
    Parameters
    ----------
    data_frame: pandas.DataFrame
        A data frame with columns that match the column names provided.
    column_names: list
        A list of column names to be selected from our data frame.

    Returns
    -------
    A list of lists
    with one entry per row of our data_frame.
    Each entry is a list of column_name:column_value string tokens for each selected column that was present.
    """
    if not set(column_names).issubset(data_frame.columns):
        raise ValueError("Selected column_names must be a subset of your data_frame")

    label_list = [
        [f"{k}:{v}" for k, v in zip(column_names, t) if not pd.isnull(v)]
        for t in zip(*map(data_frame.get, column_names))
    ]
    return label_list
