from .ngram_vectorizer import (
    ngrams_of,
)
from .preprocessing import (
    prune_token_dictionary,
    preprocess_token_sequences,
    construct_token_dictionary_and_frequency,
    construct_document_frequency,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd, svd_flip
from collections.abc import Iterable
from scipy.sparse.linalg import svds

import vectorizers.distances as distances

from .utils import (
    validate_homogeneous_token_types,
    flatten,
    str_to_bytes,
    pair_to_tuple,
    make_tuple_converter,
)

from .coo_utils import coo_append, coo_sum_duplicates, CooArray

import numpy as np
import numba
import scipy.sparse
from ._window_kernels import (
    _KERNEL_FUNCTIONS,
    _WINDOW_FUNCTIONS,
    window_at_index,
    update_kernel,
)

MOCK_DICT = numba.typed.Dict()
MOCK_DICT[(-1, -1)] = -1


@numba.njit(nogil=True)
def build_multi_skip_ngrams(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    array_lengths,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    array_to_tuple=pair_to_tuple,
):
    """Generate a matrix of (weighted) counts of co-occurrences of tokens within
    windows in a set of sequences of tokens. Each sequence in the collection of
    sequences provides an effective boundary over which skip-grams may not pass
    (such as sentence boundaries in an NLP context). This is done for a collection
    of different window and kernel types simultaneously.

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_size_array: numpy.ndarray(float, size = (n_windows, n_unique_tokens))
        A collection of window sizes per vocabulary index per window function

    window_reversals: numpy.array(bool, size = (n_windows,))
        Array indicating whether the window is after or not.

    kernel_array: numpy.ndarray(float, size = (n_windows, max_window_radius))
        A collection of kernel values per window index per window funciton

    kernel_args: tuple of tuples
        Arguments to pass through to the kernel functions per function

    mix_weights: numpy.array(bool, size = (n_windows,))
        The scalars values used to combine the values of the kernel functions

    normalize_windows: bool
        Indicates whether or nor to L_1 normalize the kernel values per window occurrence

    array_lengths: numpy.array(int, size = (n_windows,))
        The lengths of the arrays per window used to the store the coo matrix triples.

    ngram_dictionary: dict (optional)
        The dictionary from tuples of token indices to an n_gram index

    ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    array_to_tuple: numba.jitted callable (optional)
        Function that casts arrays of fixed length to tuples

    Returns
    -------
    cooccurrence_matrix: CooArray
        Weight counts of values (kernel weighted counts) that token_head[i] cooccurred with token_tail[i]
    """

    n_windows = window_size_array.shape[0]
    array_mul = n_windows * n_unique_tokens + 1
    kernel_masks = [ker[0] for ker in kernel_args]
    kernel_normalize = [ker[1] for ker in kernel_args]
    window_reversal_const = np.zeros(len(window_reversals)).astype(np.int32)
    window_reversal_const[window_reversals] = 1
    coo_data = [
        CooArray(
            np.zeros(array_lengths[i], dtype=np.int32),
            np.zeros(array_lengths[i], dtype=np.int32),
            np.zeros(array_lengths[i], dtype=np.float32),
            np.zeros(array_lengths[i], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        for i in range(n_windows)
    ]
    for d_i, seq in enumerate(token_sequences):
        for w_i in range(ngram_size - 1, len(seq)):

            ngram = array_to_tuple(seq[w_i - ngram_size + 1 : w_i + 1])

            if ngram in ngram_dictionary:
                target_gram_ind = ngram_dictionary[ngram]
                windows = [
                    window_at_index(
                        seq,
                        window_size_array[i, target_gram_ind],
                        w_i - window_reversal_const[i] * (ngram_size - 1),
                        reverse=window_reversals[i],
                    )
                    for i in range(n_windows)
                ]

                kernels = [
                    mix_weights[i]
                    * update_kernel(
                        windows[i],
                        kernel_array[i],
                        kernel_masks[i],
                        kernel_normalize[i],
                    )
                    for i in range(n_windows)
                ]

                total = 0
                if normalize_windows:
                    sums = np.array([np.sum(ker) for ker in kernels])
                    total = np.sum(sums)
                if total <= 0:
                    total = 1

                for i, window in enumerate(windows):
                    this_ker = kernels[i]
                    for j, context in enumerate(window):
                        val = np.float32(this_ker[j] / total)
                        if val > 0:
                            row = target_gram_ind
                            col = context + i * n_unique_tokens
                            key = col + array_mul * row
                            coo_append(coo_data[i], (row, col, val, key))

    return coo_data


@numba.njit(nogil=True)
def build_multi_skip_grams(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    array_lengths,
):
    """Generate a matrix of (weighted) counts of co-occurrences of tokens within
    windows in a set of sequences of tokens. Each sequence in the collection of
    sequences provides an effective boundary over which skip-grams may not pass
    (such as sentence boundaries in an NLP context). This is done for a collection
    of different window and kernel types simultaneously.

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_size_array: numpy.ndarray(float, size = (n_windows, n_unique_tokens))
        A collection of window sizes per vocabulary index per window function

    window_reversals: numpy.array(bool, size = (n_windows,))
        Array indicating whether the window is after or not.

    kernel_array: numpy.ndarray(float, size = (n_windows, max_window_radius))
        A collection of kernel values per window index per window funciton

    kernel_args: tuple of tuples
        Arguments to pass through to the kernel functions per function

    mix_weights: numpy.array(bool, size = (n_windows,))
        The scalars values used to combine the values of the kernel functions

    normalize_windows: bool
        Indicates whether or nor to L_1 normalize the kernel values per window occurrence

    array_lengths: numpy.array(int, size = (n_windows,))
        The lengths of the arrays per window used to the store the coo matrix triples.

    Returns
    -------
    cooccurrence_matrix: CooArray
        Weight counts of values (kernel weighted counts) that token_head[i] cooccurred with token_tail[i]
    """

    n_windows = window_size_array.shape[0]
    array_mul = n_windows * n_unique_tokens + 1
    kernel_masks = [ker[0] for ker in kernel_args]
    kernel_normalize = [ker[1] for ker in kernel_args]

    coo_data = [
        CooArray(
            np.zeros(array_lengths[i], dtype=np.int32),
            np.zeros(array_lengths[i], dtype=np.int32),
            np.zeros(array_lengths[i], dtype=np.float32),
            np.zeros(array_lengths[i], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        for i in range(n_windows)
    ]
    for d_i, seq in enumerate(token_sequences):
        for w_i, target_word in enumerate(seq):
            windows = [
                window_at_index(
                    seq,
                    window_size_array[i, target_word],
                    w_i,
                    reverse=window_reversals[i],
                )
                for i in range(n_windows)
            ]

            kernels = [
                mix_weights[i]
                * update_kernel(
                    windows[i], kernel_array[i], kernel_masks[i], kernel_normalize[i]
                )
                for i in range(n_windows)
            ]

            total = 0
            if normalize_windows:
                sums = np.array([np.sum(ker) for ker in kernels])
                total = np.sum(sums)
            if total <= 0:
                total = 1

            for i, window in enumerate(windows):
                this_ker = kernels[i]
                for j, context in enumerate(window):
                    val = np.float32(this_ker[j] / total)
                    if val > 0:
                        row = target_word
                        col = context + i * n_unique_tokens
                        key = col + array_mul * row
                        coo_append(coo_data[i], (row, col, val, key))

    return coo_data


@numba.njit(nogil=True)
def sequence_multi_skip_grams(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    array_lengths,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    array_to_tuple=pair_to_tuple,
):
    """Generate a sequence of (weighted) counts of co-occurrences of tokens within
    windows in a set of sequences of tokens. Each sequence in the collection of
    sequences provides an effective boundary over which skip-grams may not pass
    (such as sentence boundaries in an NLP context). This is done for a collection
    of different window and kernel types simultaneously.

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_size_array: numpy.ndarray(float, size = (n_windows, n_unique_tokens))
        A collection of window sizes per vocabulary index per window function

    window_reversals: numpy.array(bool, size = (n_windows,))
        Array indicating whether the window is after or not.

    kernel_array: numpy.ndarray(float, size = (n_windows, max(window_size_array)))
        A collection of kernel values per window index per window funciton

    kernel_args: tuple of tuples
        Arguments to pass through to the kernel functions per function

    mix_weights: numpy.array(bool, size = (n_windows,))
        The scalars values used to combine the values of the kernel functions

    normalize_windows: bool
        Indicates whether or nor to L_1 normalize the kernel values per window occurrence

    array_lengths: numpy.array(int, size = (n_windows,))
        The lengths of the arrays per window used to the store the coo matrix triples.

    ngram_dictionary: dict (optional)
        The dictionary from tuples of token indices to an n_gram index

    ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    array_to_tuple: numba.jitted callable (optional)
        Function that casts arrays of fixed length to tuples

    Returns
    -------
    token_head, token_tail, values: numpy.array, numpy.array, numpy.array:
        Weight counts of values (kernel weighted counts) that token_head[i] cooccurred with token_tail[i]
    """
    if ngram_size > 1:
        coo_list = build_multi_skip_ngrams(
            token_sequences=token_sequences,
            window_size_array=window_size_array,
            window_reversals=window_reversals,
            kernel_array=kernel_array,
            kernel_args=kernel_args,
            mix_weights=mix_weights,
            normalize_windows=normalize_windows,
            n_unique_tokens=n_unique_tokens,
            array_lengths=array_lengths,
            ngram_dictionary=ngram_dictionary,
            ngram_size=ngram_size,
            array_to_tuple=array_to_tuple,
        )
    else:
        coo_list = build_multi_skip_grams(
            token_sequences=token_sequences,
            window_size_array=window_size_array,
            window_reversals=window_reversals,
            kernel_array=kernel_array,
            kernel_args=kernel_args,
            mix_weights=mix_weights,
            normalize_windows=normalize_windows,
            n_unique_tokens=n_unique_tokens,
            array_lengths=array_lengths,
        )

    for coo in coo_list:
        coo.min[0] = 0
        coo_sum_duplicates(coo, kind="mergesort")

    return (
        [coo.row[: coo.ind[0]] for coo in coo_list],
        [coo.col[: coo.ind[0]] for coo in coo_list],
        [coo.val[: coo.ind[0]] for coo in coo_list],
    )


@numba.njit(nogil=True)
def set_array_size(token_sequences, window_array):
    tot_len = np.zeros(window_array.shape[0]).astype(np.float64)
    window_array = window_array.astype(np.float64)
    for seq in token_sequences:
        counts = np.bincount(seq, minlength=window_array.shape[1]).astype(np.float64)
        tot_len += np.dot(
            window_array, counts
        ).T  # NOTE: numba only does dot products with floats
    return tot_len.astype(np.int64)


def multi_token_cooccurrence_matrix(
    token_sequences,
    n_unique_tokens,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args,
    mix_weights,
    normalize_windows,
    array_lengths,
    n_iter,
    epsilon,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    chunk_size=1 << 20,
):
    """Generate a matrix of (weighted) counts of co-occurrences of tokens within
    windows in a set of sequences of tokens. Each sequence in the collection of
    sequences provides an effective boundary over which skip-grams may not pass
    (such as sentence boundaries in an NLP context). This is done for a collection
    of different window and kernel types simultaneously.

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_size_array: numpy.ndarray(float, size = (n_windows, n_unique_tokens))
        A collection of window sizes per vocabulary index per window function

    window_reversals: numpy.array(bool, size = (n_windows,))
        Array indicating whether the window is after or not.

    kernel_array: numpy.ndarray(float, size = (n_windows, max(window_size_array)))
        A collection of kernel values per window index per window funciton

    kernel_args: tuple of tuples
        Arguments to pass through to the kernel functions per function

    mix_weights: numpy.array(bool, size = (n_windows,))
        The scalars values used to combine the values of the kernel functions

    normalize_windows: bool
        Indicates whether or nor to L_1 normalize the kernel values per window occurrence

    array_lengths: numpy.array(int, size = (n_windows,))
        The lengths of the arrays per window used to the store the coo matrix triples.

    n_iter: int
        The number of iterations of EM to perform

    epsilon: float
        Set to zero all coooccurrence matrix values less than epsilon

    ngram_dictionary: dict (optional)
        The dictionary from tuples of token indices to an n_gram index

    ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    chunk_size: int (optional, default=1048576)
        When processing token sequences, break the list of sequences into
        chunks of this size to stream the data through, rather than storing all
        the results at once. This saves on peak memory usage.

    Returns
    -------
    cooccurrence_matrix: scipy.sparse.csr_matrix
        A matrix of shape (n_unique_tokens, n_windows*n_unique_tokens) where the i,j entry gives
        the (weighted) count of the number of times token i cooccurs within a
        window with token (j mod n_unique_tokens) for window/kernel function (j // n_unique_tokens).
    """
    if n_unique_tokens == 0:
        raise ValueError("Token dictionary is empty; try using less extreme contraints")

    if n_unique_tokens == 0:
        raise ValueError("Token dictionary is empty; try using less extreme contraints")

    if len(ngram_dictionary) == 1 or ngram_size == 1:
        n_rows = n_unique_tokens
        array_to_tuple = pair_to_tuple  # Mock function for this case; unused
    else:
        n_rows = len(ngram_dictionary)
        array_to_tuple = make_tuple_converter(ngram_size)

    cooccurrence_matrix = scipy.sparse.coo_matrix(
        (n_rows, window_size_array.shape[0] * n_unique_tokens),
        dtype=np.float32,
    )
    n_chunks = (len(token_sequences) // chunk_size) + 1

    for chunk_index in range(n_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(len(token_sequences), chunk_start + chunk_size)
        coo_rows, coo_cols, coo_vals = sequence_multi_skip_grams(
            token_sequences=token_sequences[chunk_start:chunk_end],
            n_unique_tokens=n_unique_tokens,
            window_size_array=window_size_array,
            window_reversals=window_reversals,
            kernel_array=kernel_array,
            kernel_args=kernel_args,
            mix_weights=mix_weights,
            normalize_windows=normalize_windows,
            array_lengths=array_lengths,
            ngram_dictionary=ngram_dictionary,
            ngram_size=ngram_size,
            array_to_tuple=array_to_tuple,
        )

        cooccurrence_matrix += scipy.sparse.coo_matrix(
            (
                np.hstack(coo_vals),
                (
                    np.hstack(coo_rows),
                    np.hstack(coo_cols),
                ),
            ),
            shape=(n_rows, n_unique_tokens * window_size_array.shape[0]),
            dtype=np.float32,
        )

    cooccurrence_matrix.sum_duplicates()
    cooccurrence_matrix = cooccurrence_matrix.tocsr()

    if n_iter > 0 or epsilon > 0:
        cooccurrence_matrix = normalize(cooccurrence_matrix, axis=0, norm="l1").tocsr()
        cooccurrence_matrix.data[cooccurrence_matrix.data < epsilon] = 0
        cooccurrence_matrix.eliminate_zeros()
        cooccurrence_matrix = normalize(cooccurrence_matrix, axis=0, norm="l1").tocsr()

    # Do the EM
    n_chunks = (len(token_sequences) // chunk_size) + 1

    for iter in range(n_iter):
        new_data = np.zeros_like(cooccurrence_matrix.data)
        for chunk_index in range(n_chunks):
            chunk_start = chunk_index * chunk_size
            chunk_end = min(len(token_sequences), chunk_start + chunk_size)
            new_data += em_cooccurrence_iteration(
                token_sequences=token_sequences[chunk_start:chunk_end],
                n_unique_tokens=n_unique_tokens,
                window_size_array=window_size_array,
                window_reversals=window_reversals,
                kernel_array=kernel_array,
                kernel_args=kernel_args,
                mix_weights=mix_weights,
                prior_data=cooccurrence_matrix.data,
                prior_indices=cooccurrence_matrix.indices,
                prior_indptr=cooccurrence_matrix.indptr,
                ngram_dictionary=ngram_dictionary,
                ngram_size=ngram_size,
                array_to_tuple=array_to_tuple,
            )
        cooccurrence_matrix.data = new_data
        cooccurrence_matrix = normalize(cooccurrence_matrix, axis=0, norm="l1").tocsr()
        cooccurrence_matrix.data[cooccurrence_matrix.data < epsilon] = 0
        cooccurrence_matrix.eliminate_zeros()
        cooccurrence_matrix = normalize(cooccurrence_matrix, axis=0, norm="l1").tocsr()

    return cooccurrence_matrix.tocsr()


@numba.njit(nogil=True, inline="always")
def em_update_matrix(
    posterior_data,
    prior_indices,
    prior_indptr,
    prior_data,
    n_unique_tokens,
    target_gram_ind,
    windows,
    kernels,
):
    """
    Updated the csr matrix from one round of EM on the given (hstack of) n cooccurrence matrices provided in csr format.

    Parameters
    ----------
    posterior_data: numpy.array
        The csr data of the hstacked cooccurrence matrix to be updated

    prior_indices:  numpy.array
        The csr indices of the hstacked cooccurrence matrix

    prior_indptr: numpy.array
        The csr indptr of the hstacked cooccurrence matrix

    prior_data: numpy.array
        The csr data of the hstacked cooccurrence matrix

    n_unique_tokens: int
        The number of unique tokens

    target_gram_ind: int
        The index of the target ngram to update

    windows: List of List of int
        The indices of the tokens in the windows

    kernels: List of List of floats
        The kernel values of the entries in the windows.

    Returns
    -------
    posterior_data: numpy.array
        The data of the updated csr matrix after an update of EM.
    """
    total_win_length = np.sum(np.array([len(w) for w in windows]))
    window_posterior = np.zeros(total_win_length)
    context_ind = np.zeros(total_win_length, dtype=np.int64)
    win_offset = np.append(
        np.zeros(1, dtype=np.int64),
        np.cumsum(np.array([len(w) for w in windows])),
    )[:-1]

    col_ind = prior_indices[
        prior_indptr[target_gram_ind] : prior_indptr[target_gram_ind + 1]
    ]

    for w, window in enumerate(windows):
        for i, context in enumerate(window):
            if kernels[w][i] > 0:
                context_ind[i + win_offset[w]] = np.searchsorted(
                    col_ind, context + w * n_unique_tokens
                )
                # assert(col_ind[context_ind[i + win_offset[w]]] == context+w * n_unique_tokens)
                if (
                    col_ind[context_ind[i + win_offset[w]]]
                    == context + w * n_unique_tokens
                ):
                    window_posterior[i + win_offset[w]] = (
                        kernels[w][i]
                        * prior_data[
                            prior_indptr[target_gram_ind]
                            + context_ind[i + win_offset[w]]
                        ]
                    )
                else:
                    window_posterior[i + win_offset[w]] = 0

    temp = window_posterior.sum()
    if temp > 0:
        window_posterior /= temp

    # Partial M_step - Update the posteriors
    for w, window in enumerate(windows):
        for i, context in enumerate(window):
            val = window_posterior[i + win_offset[w]]
            if val > 0:
                posterior_data[
                    prior_indptr[target_gram_ind] + context_ind[i + win_offset[w]]
                ] += val

    return posterior_data


@numba.njit(nogil=True)
def em_cooccurrence_iteration(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args,
    mix_weights,
    n_unique_tokens,
    prior_indices,
    prior_indptr,
    prior_data,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    array_to_tuple=pair_to_tuple,
):
    """
    Performs one round of EM on the given (hstack of) n cooccurrence matrices provided in csr format.

    Note: The algorithm assumes the matrix is an hstack of cooccurrence matrices with the same vocabulary,
    with kernel and window parameters given in the same order.

    Parameters
    ----------

    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    window_size_array : numpy.ndarray of shape(n, n_vocab)
        The collection of window sizes per token per directed cooccurrence

    window_reversals: numpy.array(bool)
        The collection of indicators whether or not the window is after the target token.

    kernel_array: numpy.array of shape(n, max(window_size_array))
        The n-tuple of evaluated kernel functions of maximal length

    kernel_args: tuple(tuples)
        The n-tuple of update_kernel args per kernel function

    mix_weights: tuple
        The n-tuple of mix weights to apply to the kernel functions

    n_unique_tokens: int
        The number of unique tokens

    prior_indices:  numpy.array
        The csr indices of the hstacked cooccurrence matrix

    prior_indptr: numpy.array
        The csr indptr of the hstacked cooccurrence matrix

    prior_data: numpy.array
        The csr data of the hstacked cooccurrence matrix

    ngram_dictionary: dict (optional)
        The dictionary from tuples of token indices to an n_gram index

    ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    array_to_tuple: numba.jitted callable (optional)
        Function that casts arrays of fixed length to tuples

    Returns
    -------
    posterior_data: numpy.array
        The data of the updated csr matrix after one iteration of EM.

    """

    posterior_data = np.zeros_like(prior_data)
    n_windows = window_size_array.shape[0]
    kernel_masks = [ker[0] for ker in kernel_args]
    kernel_normalize = [ker[1] for ker in kernel_args]
    window_reversal_const = np.zeros(len(window_reversals)).astype(np.int32)
    window_reversal_const[window_reversals] = 1

    if ngram_size > 1:
        for d_i, seq in enumerate(token_sequences):
            for w_i in range(ngram_size - 1, len(seq)):
                ngram = array_to_tuple(seq[w_i - ngram_size + 1 : w_i + 1])
                if ngram in ngram_dictionary:
                    target_gram_ind = ngram_dictionary[ngram]
                    windows = [
                        window_at_index(
                            seq,
                            window_size_array[i, target_gram_ind],
                            w_i - window_reversal_const[i] * (ngram_size - 1),
                            reverse=window_reversals[i],
                        )
                        for i in range(n_windows)
                    ]

                    kernels = [
                        mix_weights[i]
                        * update_kernel(
                            windows[i],
                            kernel_array[i],
                            kernel_masks[i],
                            kernel_normalize[i],
                        )
                        for i in range(n_windows)
                    ]
                    posterior_data = em_update_matrix(
                        posterior_data,
                        prior_indices,
                        prior_indptr,
                        prior_data,
                        n_unique_tokens,
                        target_gram_ind,
                        windows,
                        kernels,
                    )

    else:

        for d_i, seq in enumerate(token_sequences):
            for w_i, target_word in enumerate(seq):
                windows = [
                    window_at_index(
                        seq,
                        window_size_array[i, target_word],
                        w_i,
                        reverse=window_reversals[i],
                    )
                    for i in range(n_windows)
                ]

                kernels = [
                    mix_weights[i]
                    * update_kernel(
                        windows[i],
                        kernel_array[i],
                        kernel_masks[i],
                        kernel_normalize[i],
                    )
                    for i in range(n_windows)
                ]
                posterior_data = em_update_matrix(
                    posterior_data,
                    prior_indices,
                    prior_indptr,
                    prior_data,
                    n_unique_tokens,
                    target_word,
                    windows,
                    kernels,
                )

    return posterior_data


class TokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a collection of directed
    co-occurrence count matrix of tokens. If passed a single sequence of tokens it
    will use windows to determine co-occurrence. If passed a list of sequences of
    tokens it will use windows within each sequence in the list -- with windows not
    extending beyond the boundaries imposed by the individual sequences in the list.

    Upon the construction of the count matrices, it will hstack them together and run
    n_iter iterations of EM to update the counts.

    Parameters
    ----------
    token_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary
        should be learned from the training data.

    min_occurrences: int or None (optional, default=None)
        The minimal number of occurrences of a token for it to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_frequency.

    max_occurrences int or None (optional, default=None)
        The maximal number of occurrences of a token for it to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_frequency.

    min_frequency: float or None (optional, default=None)
        The minimal frequency of occurrence of a token for it to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_occurrences.

    max_frequency: float or None (optional, default=None)
        The maximal frequency of occurrence of a token for it to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_occurrences.

    min_document_occurrences: int or None (optional, default=None)
        The minimal number of documents with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_document_frequency.

    max_document_occurrences int or None (optional, default=None)
        The maximal number of documents with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_document_frequency.

    min_document_frequency: float or None (optional, default=None)
        The minimal frequency of documents with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_document_occurrences.

    max_document_frequency: float or None (optional, default=None)
        The maximal frequency documents with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_document_occurrences.

    ignored_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored entirely. If None then no tokens will
        be ignored in this fashion.

    excluded_token_regex: str or None (optional, default=None)
        The regular expression by which tokens are ignored if re.fullmatch returns True.

    window_functions: (Iterable of) numba.jitted callable or str (optional, default=['fixed'])
        Functions producing a sequence of window radii given a window_radius parameter and term frequencies.
        The string options are ['fixed', 'variable'] for using pre-defined functions.

    kernel_functions: (Iterable of) numba.jitted callable or str (optional, default=['flat'])
        Functions producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'harmonic', 'geometric'] for using pre-defined functions.

    window_radii: (Iterable of) int (optional, default=[5])
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    window_args: (Iterable of) dicts (optional, default = None)
        Optional arguments for the window functions

    kernel_args: (Iterable of) tuple of dicts (optional, default = None)
        Optional arguments for the kernel functions, including 'normalize' which L1 normalizes
        the kernel for each window.

    window_orientations: (Iterable of) strings (['before', 'after', 'directional'])
        The orientations of the cooccurrence windows.  Whether to return all the tokens that
        occurred within a window before, after, or on either side separately.

    mix_weights: (Iterable of) tuple of float (optional, default = None)
        The mix weights to combine the values from the kernel function on each window.
        The default provides no additional rescaling (equivalent to a uniform mixture).

    normalize_windows: bool (optional, default = True)
        Perform L1 normalization on the combined mixture of kernel functions per window.

    chunk_size: int (optional, default=1048576)
        When processing token sequences, break the list of sequences into
        chunks of this size to stream the data through, rather than storing all
        the results at once. This saves on peak memory usage.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.

    skip_ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    nullify_mask: bool (optional, default=False)
        Sets all cooccurrences with the mask_string equal to zero by skipping over them during processing.

    n_iter: int (optional, default = 0)
        Number of EM iterations to perform

    epsilon: float32 (optional default = 0)
        Sets values in the cooccurrence matrix (after l_1 normalizing the columns) less than epsilon to zero

    coo_max_memory: str (optional, default = "2 GiB")
        This value, giving a memory size in k, M, G or T, describes how much memory to set for acculumating the
        (row, col, val) triples of larger data sets.  This should be at least 2 times the number of non-zero
        entries in the final cooccurrence matrix for near optimal performance.  Optimizations to use
        significantly less memory are made for data sets with small expected numbers of non zeros.
    """

    def __init__(
        self,
        token_dictionary=None,
        min_occurrences=None,
        max_occurrences=None,
        min_frequency=None,
        max_frequency=None,
        min_document_occurrences=None,
        max_document_occurrences=None,
        min_document_frequency=None,
        max_document_frequency=None,
        ignored_tokens=None,
        excluded_token_regex=None,
        unknown_token=None,
        window_functions="fixed",
        kernel_functions="flat",
        window_args=None,
        kernel_args=None,
        window_radii=5,
        mix_weights=None,
        skip_ngram_size=1,
        window_orientations="directional",
        chunk_size=1 << 20,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
        normalize_windows=True,
        n_iter=0,
        epsilon=0,
        coo_max_memory="2 GiB",
    ):
        self.token_dictionary = token_dictionary
        self.min_occurrences = min_occurrences
        self.min_frequency = min_frequency
        self.max_occurrences = max_occurrences
        self.max_frequency = max_frequency
        self.min_document_occurrences = min_document_occurrences
        self.min_document_frequency = min_document_frequency
        self.max_document_occurrences = max_document_occurrences
        self.max_document_frequency = max_document_frequency
        self.ignored_tokens = ignored_tokens
        self.excluded_token_regex = excluded_token_regex
        self.unknown_token = unknown_token
        self.window_orientations = window_orientations
        self.window_functions = window_functions
        self.kernel_functions = kernel_functions
        self.window_args = window_args
        self.kernel_args = kernel_args
        self.mix_weights = mix_weights
        self.window_radii = window_radii
        self.chunk_size = chunk_size
        self.skip_ngram_size = skip_ngram_size
        self.validate_data = validate_data
        self.mask_string = mask_string
        self.nullify_mask = nullify_mask
        self.normalize_windows = normalize_windows
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.coo_max_memory = coo_max_memory

        self.token_label_dictionary_ = {}
        self.token_index_dictionary_ = {}
        self._token_frequencies_ = np.array([])

        self.coo_max_bytes = str_to_bytes(self.coo_max_memory)

        # Check the window orientations
        if not isinstance(self.window_radii, Iterable):
            self.window_radii = [self.window_radii]
        if isinstance(self.window_orientations, str):
            self.window_orientations = [self.window_orientations]

        self._window_reversals = []
        self._window_orientations = []
        if self.mix_weights is None:
            self.mix_weights = np.ones(len(self.window_orientations))
        self._mix_weights = []

        for i, w in enumerate(self.window_orientations):
            if w == "directional":
                self._window_reversals.extend([True, False])
                self._window_orientations.extend(["before", "after"])
                self._mix_weights.extend([self.mix_weights[i], self.mix_weights[i]])
            elif w == "before":
                self._window_reversals.append(True)
                self._window_orientations.append("before")
                self._mix_weights.append(self.mix_weights[i])
            elif w == "after":
                self._window_reversals.append(False)
                self._window_orientations.append("after")
                self._mix_weights.append(self.mix_weights[i])
            else:
                raise ValueError(
                    f"Unrecognized window orientations; should be callable or one of 'before','after', or 'directional'."
                )
        self._n_wide = len(self._window_reversals)
        self._mix_weights = np.array(self._mix_weights, dtype=np.float64)
        self._window_reversals = np.array(self._window_reversals)

        # Set kernel functions
        if callable(self.kernel_functions) or isinstance(self.kernel_functions, str):
            self.kernel_functions = [self.kernel_functions]

        self._kernel_functions = []
        for i, ker in enumerate(self.kernel_functions):
            if callable(ker):
                self._kernel_functions.append(ker)
            elif ker in _KERNEL_FUNCTIONS:
                self._kernel_functions.append(_KERNEL_FUNCTIONS[ker])
            else:
                raise ValueError(
                    f"Unrecognized kernel_function; should be callable or one of {_KERNEL_FUNCTIONS.keys()}"
                )
            if self.window_orientations[i] == "directional":
                self._kernel_functions.append(self._kernel_functions[-1])

        # Set window functions
        if callable(self.window_functions) or isinstance(self.window_functions, str):
            self.window_functions = [self.window_functions]

        self._window_functions = []
        for i, win in enumerate(self.window_functions):
            if callable(win):
                self._window_functions.append(win)
            elif win in _WINDOW_FUNCTIONS:
                self._window_functions.append(_WINDOW_FUNCTIONS[win])
            else:
                raise ValueError(
                    f"Unrecognized window_function; should be callable or one of {_WINDOW_FUNCTIONS.keys()}"
                )
            if self.window_orientations[i] == "directional":
                self._window_functions.append(self._window_functions[-1])

        # Set mask nullity
        if self.nullify_mask:
            if self.mask_string is None:
                raise ValueError(f"Cannot nullify mask with mask_string = None")

        # Set window args
        self._window_args = []
        if isinstance(self.window_args, dict):
            self._window_args = tuple(
                [tuple(self.window_args.values()) for _ in range(self._n_wide)]
            )
        elif self.window_args is None:
            self._window_args = tuple([tuple([]) for _ in range(self._n_wide)])
        else:
            for i, args in enumerate(self.window_args):
                self._window_args.append(tuple(args.values()))
                if self.window_orientations[i] == "directional":
                    self._window_args.append(tuple(args.values()))
            self._window_args = tuple(self._window_args)

        # Set initial kernel args
        if isinstance(self.kernel_args, dict):
            self._kernel_args = [self.kernel_args for _ in range(self._n_wide)]
        elif self.kernel_args is None:
            self._kernel_args = [dict([]) for _ in range(self._n_wide)]
        else:
            self._kernel_args = []
            for i, args in enumerate(self.kernel_args):
                self._kernel_args.append(args)
                if self.window_orientations[i] == "directional":
                    self._kernel_args.append(args)

        # Set the window radii
        if not isinstance(self.window_radii, Iterable):
            self.window_radii = [self.window_radii]
        self._window_radii = []
        for i, radius in enumerate(self.window_radii):
            self._window_radii.append(radius)
            if self.window_orientations[i] == "directional":
                self._window_radii.append(radius)
        self._window_radii = np.array(self._window_radii)

        # Check that everything is the same size
        assert len(self._window_radii) == self._n_wide
        assert len(self._mix_weights) == self._n_wide
        assert len(self._window_args) == self._n_wide
        assert len(self._window_orientations) == self._n_wide
        assert len(self._window_functions) == self._n_wide
        assert len(self._kernel_functions) == self._n_wide
        assert len(self._kernel_args) == self._n_wide

    def _set_column_dicts(self):
        self.column_label_dictionary_ = {}
        colonnade = 0
        for i, win in enumerate(self.window_orientations):
            if win == "directional":
                self.column_label_dictionary_.update(
                    {
                        "pre_"
                        + str(i)
                        + "_"
                        + str(token): index
                        + colonnade * len(self.token_label_dictionary_)
                        for token, index in self.token_label_dictionary_.items()
                    }
                )
                colonnade += 1
                self.column_label_dictionary_.update(
                    {
                        "post_"
                        + str(i)
                        + "_"
                        + str(token): index
                        + colonnade * len(self.token_label_dictionary_)
                        for token, index in self.token_label_dictionary_.items()
                    }
                )
                colonnade += 1
            elif win == "before":
                self.column_label_dictionary_.update(
                    {
                        "pre_"
                        + str(i)
                        + "_"
                        + str(token): index
                        + colonnade * len(self.token_label_dictionary_)
                        for token, index in self.token_label_dictionary_.items()
                    }
                )
                colonnade += 1
            else:
                self.column_label_dictionary_.update(
                    {
                        "post_"
                        + str(i)
                        + "_"
                        + str(token): index
                        + colonnade * len(self.token_label_dictionary_)
                        for token, index in self.token_label_dictionary_.items()
                    }
                )
                colonnade += 1

        self.column_index_dictionary_ = {
            item[1]: item[0] for item in self.column_label_dictionary_.items()
        }
        assert len(self.column_index_dictionary_) == self.cooccurrences_.shape[1]

    def _process_n_grams(self, token_sequences):
        if self.skip_ngram_size > 1:
            ngrams = [
                list(map(tuple, ngrams_of(sequence, self.skip_ngram_size, "exact")))
                for sequence in token_sequences
            ]
            (
                raw_ngram_dictionary,
                ngram_frequencies,
                total_ngrams,
            ) = construct_token_dictionary_and_frequency(
                flatten(ngrams), token_dictionary=None
            )

            if {
                self.min_document_frequency,
                self.min_document_occurrences,
                self.max_document_frequency,
                self.max_document_occurrences,
            } != {None}:
                ngram_doc_frequencies = construct_document_frequency(
                    ngrams, raw_ngram_dictionary
                )
            else:
                ngram_doc_frequencies = np.array([])

            raw_ngram_dictionary, ngram_frequencies = prune_token_dictionary(
                raw_ngram_dictionary,
                ngram_frequencies,
                token_doc_frequencies=ngram_doc_frequencies,
                min_frequency=self.min_frequency,
                max_frequency=self.max_frequency,
                min_occurrences=self.min_occurrences,
                max_occurrences=self.max_occurrences,
                min_document_frequency=self.min_document_frequency,
                max_document_frequency=self.max_document_frequency,
                min_document_occurrences=self.min_document_occurrences,
                max_document_occurrences=self.max_document_occurrences,
                total_tokens=total_ngrams,
                total_documents=len(token_sequences),
            )
            self._raw_ngram_dictionary_ = numba.typed.Dict()
            self._raw_ngram_dictionary_.update(raw_ngram_dictionary)
            self._ngram_frequencies = ngram_frequencies

            def joined_tokens(ngram, token_index_dictionary):
                return "_".join([str(token_index_dictionary[index]) for index in ngram])

            self.ngram_label_dictionary_ = {
                joined_tokens(key, self.token_index_dictionary_): value
                for key, value in raw_ngram_dictionary.items()
            }
        else:
            self._raw_ngram_dictionary_ = MOCK_DICT

    def fit_transform(self, X, y=None, **fit_params):

        if self.validate_data:
            validate_homogeneous_token_types(X)

        flat_sequences = flatten(X)
        # noinspection PyTupleAssignmentBalance
        (
            token_sequences,
            self.token_label_dictionary_,
            self.token_index_dictionary_,
            self._token_frequencies_,
        ) = preprocess_token_sequences(
            X,
            flat_sequences,
            self.token_dictionary,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            min_document_occurrences=self.min_document_occurrences,
            max_document_occurrences=self.max_document_occurrences,
            min_document_frequency=self.min_document_frequency,
            max_document_frequency=self.max_document_frequency,
            ignored_tokens=self.ignored_tokens,
            excluded_token_regex=self.excluded_token_regex,
            masking=self.mask_string,
        )

        # Set mask nullity
        if self.nullify_mask:
            mask_index = np.int32(len(self._token_frequencies_))
        else:
            mask_index = None

        # Process the n_grams
        self._process_n_grams(token_sequences)

        # Set the mask n_gram and frequencies
        if self.skip_ngram_size > 1:
            n_gram_frequencies = self._ngram_frequencies
            if self.nullify_mask:
                mask_ngram = tuple([mask_index for i in range(self.skip_ngram_size)])
                if mask_ngram in self._raw_ngram_dictionary_:
                    mask_ngram_index = self._raw_ngram_dictionary_[mask_ngram]
                else:
                    mask_ngram_index = None
            else:
                mask_ngram_index = None
        else:
            n_gram_frequencies = self._token_frequencies_
            mask_ngram_index = mask_index

        # Set the window array
        self._window_array = []
        for i, win_fn in enumerate(self._window_functions):
            self._window_array.append(
                win_fn(
                    self._window_radii[i],
                    n_gram_frequencies,
                    mask_ngram_index,
                    *self._window_args[i],
                )
            )
        self._window_array = np.array(self._window_array)

        # Set the kernel array and adjust args
        self._em_kernel_args = []
        self._initial_kernel_args = []
        max_ker_len = np.max(self._window_array)
        self._kernel_array = np.zeros((self._n_wide, max_ker_len), dtype=np.float64)

        for i, args in enumerate(self._kernel_args):
            default_kernel_array_args = {
                "mask_index": None,
                "normalize": False,
                "offset": 0,
            }
            default_kernel_array_args.update(args)
            default_kernel_array_args["normalize"] = False
            self._kernel_array[i] = np.array(
                self._kernel_functions[i](
                    np.repeat(-1, max_ker_len),
                    *tuple(default_kernel_array_args.values()),
                )
            )
            default_initial_args = {
                "mask_index": mask_index,
                "normalize": False,
                "offset": 0,
            }
            default_initial_args.update(args)
            self._initial_kernel_args.append(tuple(default_initial_args.values()))
            self._em_kernel_args.append(
                tuple([mask_index, default_initial_args["normalize"]])
            )

        self._em_kernel_args = tuple(self._em_kernel_args)

        # Set the coo_array size
        approx_coo_size = 0
        for t in token_sequences:
            approx_coo_size += len(t)
        approx_coo_size *= (max(self.window_radii) + 1) * (20 * self._n_wide)
        if approx_coo_size < self.coo_max_bytes:
            if self.skip_ngram_size > 1:
                self._coo_sizes = np.repeat(
                    approx_coo_size // self._n_wide, self._n_wide
                ).astype(np.int64)
            else:
                self._coo_sizes = set_array_size(
                    token_sequences,
                    self._window_array,
                )
        else:
            offsets = np.array(
                [self._initial_kernel_args[i][2] for i in range(self._n_wide)]
            )
            average_window = self._window_radii - offsets
            self._coo_sizes = (self.coo_max_bytes // 20) // np.sum(average_window)
            self._coo_sizes = np.array(self._coo_sizes * average_window, dtype=np.int64)

        if np.any(self._coo_sizes == 0):
            raise ValueError(f"The coo_max_mem is too small to process the data.")

        # Build the initial matrix
        self.cooccurrences_ = multi_token_cooccurrence_matrix(
            token_sequences,
            len(self.token_label_dictionary_),
            window_size_array=self._window_array,
            window_reversals=self._window_reversals,
            kernel_array=self._kernel_array,
            kernel_args=self._em_kernel_args,
            mix_weights=self._mix_weights,
            chunk_size=self.chunk_size,
            normalize_windows=self.normalize_windows,
            array_lengths=self._coo_sizes,
            n_iter=self.n_iter,
            epsilon=self.epsilon,
            ngram_dictionary=self._raw_ngram_dictionary_,
            ngram_size=self.skip_ngram_size,
        )

        # Set attributes
        self._set_column_dicts()
        self.metric_ = distances.sparse_hellinger

        return self.cooccurrences_

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        """
        Build a token cooccurrence matrix out of an established vocabulary learned during a previous fit.
        Parameters
        ----------
        X: sequence of sequences of tokens

        Returns
        -------
        A scipy.sparse.csr_matrix
        """
        check_is_fitted(self, ["column_label_dictionary_"])

        if self.validate_data:
            validate_homogeneous_token_types(X)

        flat_sequences = flatten(X)
        # noinspection PyTupleAssignmentBalance
        (
            token_sequences,
            column_label_dictionary,
            column_index_dictionary,
            token_frequencies,
        ) = preprocess_token_sequences(
            X, flat_sequences, self.token_label_dictionary_, masking=self.mask_string
        )

        cooccurrences_ = multi_token_cooccurrence_matrix(
            token_sequences,
            len(self.token_label_dictionary_),
            window_size_array=self._window_array,
            window_reversals=self._window_reversals,
            kernel_array=self._kernel_array,
            kernel_args=self._em_kernel_args,
            mix_weights=self._mix_weights,
            chunk_size=self.chunk_size,
            normalize_windows=self.normalize_windows,
            array_lengths=self._coo_sizes,
            n_iter=self.n_iter,
            epsilon=self.epsilon,
            ngram_dictionary=self._raw_ngram_dictionary_,
            ngram_size=self.skip_ngram_size,
        )

        return cooccurrences_

    def reduce_dimension(self, dimension=150, algorithm="arpack", n_iter=10):
        check_is_fitted(self, ["column_label_dictionary_"])

        if self.n_iter <= 1:
            self.reduced_matrix_ = normalize(self.cooccurrences_, axis=1, norm="l1")
            self.reduced_matrix_ = normalize(self.reduced_matrix_, axis=1, norm="l1")
        else:
            self.reduced_matrix_ = normalize(self.cooccurrences_, axis=1, norm="l1")

        self.reduced_matrix_.data = np.power(self.reduced_matrix_.data, 0.25)

        if algorithm == "arpack":
            u, s, v = svds(self.reduced_matrix_, k=dimension)
        elif algorithm == "randomized":
            u, s, v = randomized_svd(
                self.reduced_matrix_, n_components=dimension, n_iter=n_iter
            )
        else:
            raise ValueError("algorithm should be one of 'arpack' or 'randomized'")

        u, v = svd_flip(u, v)
        self.reduced_matrix_ = u * np.power(s, 0.5)

        return self.reduced_matrix_
