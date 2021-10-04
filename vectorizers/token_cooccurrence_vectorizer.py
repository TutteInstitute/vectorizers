from .ngram_vectorizer import ngrams_of
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
    dirichlet_process_normalize,
    dp_normalize_vector,
    l1_normalize_vector,
)

from .coo_utils import (
    coo_append,
    coo_sum_duplicates,
    CooArray,
    merge_all_sum_duplicates,
    set_array_size,
)

import numpy as np
import numba
import dask
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
            np.zeros(2 * np.int64(np.ceil(np.log2(array_lengths[i]))), dtype=np.int64),
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
                            coo_data[i] = coo_append(coo_data[i], (row, col, val, key))

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
            np.zeros(2 * np.int64(np.ceil(np.log2(array_lengths[i]))), dtype=np.int64),
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
                        coo_data[i] = coo_append(coo_data[i], (row, col, val, key))

    return coo_data


@numba.njit(nogil=True)
def build_multi_sequence_grams(
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
            np.zeros(2 * np.int64(np.ceil(np.log2(array_lengths[i]))), dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        for i in range(n_windows)
    ]

    for d_i, seq in enumerate(token_sequences):
        for w_i, target_word in enumerate(seq):
            for i in range(n_windows):
                if window_reversals[i] == False:
                    doc_window = token_sequences[
                        d_i : min(
                            [len(token_sequences), d_i + window_size_array[i, 0] + 1]
                        )
                    ]
                elif window_reversals[i] == True:
                    doc_window = token_sequences[
                        max([0, d_i - window_size_array[i, 0]]) : d_i + 1
                    ]

                result_len = 0
                for window in doc_window:
                    result_len += window.shape[0]
                window_result = np.zeros(result_len).astype(np.int32)
                j = 0
                for window in doc_window:
                    for x in window:
                        window_result[j] = x
                        j += 1

                kernel_result = np.zeros(len(window_result)).astype(np.float64)
                ind = 0
                if window_reversals[i] == False:
                    for doc_index, doc in enumerate(doc_window):
                        kernel_result[ind : ind + len(doc)] = np.repeat(
                            kernel_array[i][np.abs(doc_index)], len(doc)
                        )
                        ind += len(doc)
                    kernel_result[w_i] = 0
                else:
                    for doc_index, doc in enumerate(doc_window):
                        kernel_result[ind : ind + len(doc)] = np.repeat(
                            kernel_array[i][len(doc_window) - doc_index - 1], len(doc)
                        )
                        ind += len(doc)
                    kernel_result[ind - len(doc_window[-1]) + w_i] = 0

                if kernel_masks[i] is not None:
                    for w in range(window_result.shape[0]):
                        if window_result[w] == kernel_masks[i]:
                            kernel_result[w] = 0

                if kernel_normalize[i]:
                    temp = kernel_result.sum()
                    if temp > 0:
                        kernel_result /= temp

                if i == 0:
                    windows = [window_result]
                    kernels = [mix_weights[i] * kernel_result]
                else:
                    windows.append(window_result)
                    kernels.append(mix_weights[i] * kernel_result)

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
    multi_labelled_tokens=False,
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

    multi_labelled_tokens: bool (optional, default = False)
        Indicates whether we are an iterable of iterables (False)
        or an iterable of iterables of iterables (True)
        That is it indicates whether we have a sequence of tokens
        or a sequence of bags of labels.

    Returns
    -------
    token_head, token_tail, values: numpy.array, numpy.array, numpy.array:
        Weight counts of values (kernel weighted counts) that token_head[i] cooccurred with token_tail[i]
    """
    if ngram_size > 1:
        if multi_labelled_tokens == True:
            raise ValueError(
                f"Document contexts are not supported for ngrams at this time.  "
                f"Please set multi_labelled_tokens=False."
            )
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
        if multi_labelled_tokens:
            coo_list = build_multi_sequence_grams(
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
        coo_sum_duplicates(coo, kind="quicksort")
        merge_all_sum_duplicates(coo)

    return (
        [coo.row[: coo.ind[0]] for coo in coo_list],
        [coo.col[: coo.ind[0]] for coo in coo_list],
        [coo.val[: coo.ind[0]] for coo in coo_list],
    )


def generate_chunk_boundaries(data, chunk_size=1 << 19):
    token_list_sizes = np.array([len(x) for x in data])
    cumulative_sizes = np.cumsum(token_list_sizes)
    chunks = []
    last_chunk_end = 0
    last_chunk_cumulative_size = 0
    for chunk_index, size in enumerate(cumulative_sizes):
        if size - last_chunk_cumulative_size >= chunk_size:
            chunks.append((last_chunk_end, chunk_index))
            last_chunk_end = chunk_index
            last_chunk_cumulative_size = size

    chunks.append((last_chunk_end, len(data)))

    return chunks


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
    normalizer,
    window_normalizer,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    chunk_size=1 << 19,
    multi_labelled_tokens=False,
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

    normalizer: function
        The function to perform feature normalization

    ngram_dictionary: dict (optional)
        The dictionary from tuples of token indices to an n_gram index

    ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    chunk_size: int (optional, default=1048576)
        When processing token sequences, break the list of sequences into
        chunks of this size to stream the data through, rather than storing all
        the results at once. This saves on peak memory usage.

    multi_labelled_tokens: bool (optional, default=False)
        Indicates whether your contexts are a sequence of bags of tokens with the context co-occurrence
        spanning the bags.

    Returns
    -------
    cooccurrence_matrix: scipy.sparse.csr_matrix
        A matrix of shape (n_unique_tokens, n_windows*n_unique_tokens) where the i,j entry gives
        the (weighted) count of the number of times token i cooccurs within a
        window with token (j mod n_unique_tokens) for window/kernel function (j // n_unique_tokens).
    """
    if n_unique_tokens == 0:
        raise ValueError(
            "Token dictionary is empty; try using less extreme constraints"
        )

    if n_unique_tokens == 0:
        raise ValueError(
            "Token dictionary is empty; try using less extreme constraints"
        )

    if len(ngram_dictionary) == 1 or ngram_size == 1:
        n_rows = n_unique_tokens
        array_to_tuple = pair_to_tuple  # Mock function for this case; unused
    else:
        n_rows = len(ngram_dictionary)
        array_to_tuple = make_tuple_converter(ngram_size)

    @dask.delayed()
    def process_token_sequence_chunk(chunk_start, chunk_end):
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
            multi_labelled_tokens=multi_labelled_tokens,
        )

        result = scipy.sparse.coo_matrix(
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
        result.sum_duplicates()
        return result.tocsr()

    matrix_per_chunk = [
        process_token_sequence_chunk(chunk_start, chunk_end)
        for chunk_start, chunk_end in generate_chunk_boundaries(
            token_sequences, chunk_size=chunk_size
        )
    ]
    cooccurrence_matrix = dask.delayed(sum)(matrix_per_chunk)
    cooccurrence_matrix = cooccurrence_matrix.compute()
    cooccurrence_matrix.sum_duplicates()
    cooccurrence_matrix = cooccurrence_matrix.tocsr()

    if n_iter > 0 or epsilon > 0:
        cooccurrence_matrix = normalizer(cooccurrence_matrix, axis=0, norm="l1").tocsr()
        cooccurrence_matrix.data[cooccurrence_matrix.data < epsilon] = 0
        cooccurrence_matrix.eliminate_zeros()

    # Do the EM
    for iter in range(n_iter):
        new_data_per_chunk = [
            dask.delayed(em_cooccurrence_iteration)(
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
                window_normalizer=window_normalizer,
                multi_labelled_tokens=multi_labelled_tokens,
            )
            for chunk_start, chunk_end in generate_chunk_boundaries(
                token_sequences, chunk_size=chunk_size
            )
        ]
        new_data = dask.delayed(sum)(new_data_per_chunk)
        new_data = new_data.compute()
        cooccurrence_matrix.data = new_data
        cooccurrence_matrix = normalizer(cooccurrence_matrix, axis=0, norm="l1").tocsr()
        cooccurrence_matrix.data[cooccurrence_matrix.data < epsilon] = 0
        cooccurrence_matrix.eliminate_zeros()

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
    window_normalizer,
):
    """
    Updated the csr matrix from one round of EM on the given (hstack of) n
    cooccurrence matrices provided in csr format.

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
        window_posterior = window_normalizer(window_posterior)

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
    window_normalizer,
    n_unique_tokens,
    prior_indices,
    prior_indptr,
    prior_data,
    ngram_dictionary=MOCK_DICT,
    ngram_size=1,
    array_to_tuple=pair_to_tuple,
    multi_labelled_tokens=False,
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

    multi_labelled_tokens: bool (optional, default=False)
        Indicates whether your contexts are a sequence of bags of tokens labels with the context
        co-occurrence spanning the bags.  In other words if you have sequences of
        multi-labelled tokens.

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
                        window_normalizer,
                    )

    else:
        if multi_labelled_tokens:
            for d_i, seq in enumerate(token_sequences):
                for w_i, target_word in enumerate(seq):
                    for i in range(n_windows):
                        if window_reversals[i] == False:
                            doc_window = token_sequences[
                                d_i : min(
                                    [
                                        len(token_sequences),
                                        d_i + window_size_array[i, 0] + 1,
                                    ]
                                )
                            ]
                        elif window_reversals[i] == True:
                            doc_window = token_sequences[
                                max([0, d_i - window_size_array[i, 0]]) : d_i + 1
                            ]

                        result_len = 0
                        for window in doc_window:
                            result_len += window.shape[0]
                        window_result = np.zeros(result_len).astype(np.int32)
                        j = 0
                        for window in doc_window:
                            for x in window:
                                window_result[j] = x
                                j += 1

                        kernel_result = np.zeros(len(window_result)).astype(np.float64)
                        ind = 0
                        if window_reversals[i] == False:
                            for doc_index, doc in enumerate(doc_window):
                                kernel_result[ind : ind + len(doc)] = np.repeat(
                                    kernel_array[i][np.abs(doc_index)], len(doc)
                                )
                                ind += len(doc)
                            kernel_result[w_i] = 0
                        else:
                            for doc_index, doc in enumerate(doc_window):
                                kernel_result[ind : ind + len(doc)] = np.repeat(
                                    kernel_array[i][len(doc_window) - doc_index - 1],
                                    len(doc),
                                )
                                ind += len(doc)
                            kernel_result[ind - len(doc_window[-1]) + w_i] = 0

                        if kernel_masks[i] is not None:
                            for w in range(window_result.shape[0]):
                                if window_result[w] == kernel_masks[i]:
                                    kernel_result[w] = 0

                        if kernel_normalize[i]:
                            temp = kernel_result.sum()
                            if temp > 0:
                                kernel_result /= temp

                        if i == 0:
                            windows = [window_result]
                            kernels = [mix_weights[i] * kernel_result]
                        else:
                            windows.append(window_result)
                            kernels.append(mix_weights[i] * kernel_result)

                    posterior_data = em_update_matrix(
                        posterior_data,
                        prior_indices,
                        prior_indptr,
                        prior_data,
                        n_unique_tokens,
                        target_word,
                        windows,
                        kernels,
                        window_normalizer,
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
                        window_normalizer,
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
        Argument to pass through to the window function.  Outside of boundary cases,
        this is the expected width of the (directed) windows produced by the window function.

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
        Prunes the filtered tokens when None, otherwise replaces them with the
        provided mask_string.

    skip_ngram_size: int (optional, default = 1)
        The size of ngrams to encode token cooccurences of.

    nullify_mask: bool (optional, default=False)
        Sets all cooccurrences with the mask_string equal to zero by skipping over them
        during processing.

    n_iter: int (optional, default = 0)
        Number of EM iterations to perform

    context_document_width: 2-tuple  (optional, default = (0,0) )
        The number of additional documents before and after the target to
        potentially include in the context windows

    epsilon: float32 (optional default = 0)
        Sets values in the cooccurrence matrix (after l_1 normalizing the columns)
        less than epsilon to zero

    normalization: str ("bayesian" or "frequentist")
        Sets the feature normalization to be the frequentist L_1 norm
        or the Bayesian (Dirichlet Process) normalization

    window_normalization: str ("bayesian" or "frequentist")
        Sets the window normalization to be the frequentist L_1 norm
        or the Bayesian (Dirichlet Process) normalization

    coo_max_memory: str (optional, default = "0.5 GiB")
        This value, giving a memory size in k, M, G or T, describes how much memory
        to initialize for acculumatingthe (row, col, val) triples of larger data sets.
        This should be at least 2 times the number of non-zero entries in the final
        cooccurrence matrix for near optimal speed in performance.  Optimizations to use
        significantly less memory are made for data sets with small expected numbers of
        non zeros. More memory will be allocated during processing if need be.

    multi_labelled_tokens: bool (optional, default=False)
        Indicates whether your contexts are a sequence of bags of tokens labels
        with the context co-occurrence spanning the bags.  In other words if you have
        sequences of multi-labelled tokens.
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
        normalization="frequentist",
        window_normalization="frequentist",
        coo_max_memory="0.5 GiB",
        multi_labelled_tokens=False,
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
        self.normalization = normalization
        self.window_normalization = window_normalization
        self.token_label_dictionary_ = {}
        self.token_index_dictionary_ = {}
        self._token_frequencies_ = np.array([])

        self.coo_max_bytes = str_to_bytes(self.coo_max_memory)
        self.multi_labelled_tokens = multi_labelled_tokens

        # Check the window orientations
        if not isinstance(self.window_radii, Iterable):
            self.window_radii = [self.window_radii]
        if isinstance(self.window_orientations, str) or callable(
            self.window_orientations
        ):
            self.window_orientations = [
                self.window_orientations for _ in self.window_radii
            ]

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
                    f"Unrecognized window orientations; should be callable "
                    f"or one of 'before','after', or 'directional'."
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

        if self.normalization == "bayesian":
            self._normalize = dirichlet_process_normalize
        else:
            self._normalize = normalize

        if self.window_normalization == "bayesian":
            self._window_normalize = dp_normalize_vector
        else:
            self._window_normalize = l1_normalize_vector

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
        max_ker_len = np.max(self._window_array) + 1
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
            normalizer=self._normalize,
            window_normalizer=self._window_normalize,
            ngram_dictionary=self._raw_ngram_dictionary_,
            ngram_size=self.skip_ngram_size,
            multi_labelled_tokens=self.multi_labelled_tokens,
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
            window_normalizer=self._window_normalize,
            array_lengths=self._coo_sizes,
            n_iter=self.n_iter,
            epsilon=self.epsilon,
            normalizer=self._normalize,
            ngram_dictionary=self._raw_ngram_dictionary_,
            ngram_size=self.skip_ngram_size,
            multi_labelled_tokens=self.multi_labelled_tokens,
        )

        return cooccurrences_

    def reduce_dimension(
        self,
        dimension=150,
        algorithm="arpack",
        n_iter=10,
        row_norm="frequentist",
        power=0.25,
    ):
        check_is_fitted(self, ["column_label_dictionary_"])
        if row_norm == "bayesian":
            row_normalize = dirichlet_process_normalize
        else:
            row_normalize = normalize

        if self.n_iter < 1:
            self.reduced_matrix_ = self._normalize(
                self.cooccurrences_, axis=0, norm="l1"
            )
            self.reduced_matrix_ = row_normalize(
                self.reduced_matrix_, axis=1, norm="l1"
            )
        else:
            self.reduced_matrix_ = row_normalize(self.cooccurrences_, axis=1, norm="l1")

        self.reduced_matrix_.data = np.power(self.reduced_matrix_.data, power)

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
