from collections.abc import Iterable
from .coo_utils import (
    coo_append,
    CooArray,
    em_update_matrix,
    coo_sum_duplicates,
    merge_all_sum_duplicates,
)
from .base_cooccurrence_vectorizer import BaseCooccurrenceVectorizer
import numpy as np
import numba
import scipy.sparse
from .preprocessing import preprocess_multi_token_sequences
from .utils import flatten
from ._window_kernels import (
    _MULTI_KERNEL_FUNCTIONS,
)


@numba.njit(nogil=True)
def numba_build_multi_skip_grams(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_functions,
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
        The collection of sequences of multisets of tokens to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_size_array: numpy.ndarray(float, size = (n_windows, n_unique_tokens))
        A collection of window sizes per vocabulary index per window function

    window_reversals: numpy.array(bool, size = (n_windows,))
        Array indicating whether the window is after or not.

    kernel_functions: kernel_functions: tuple
        The n-tuple of kernel functions

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
            windows = []
            kernels = []
            for i in range(n_windows):
                if not window_reversals[i]:
                    multi_window = token_sequences[
                        d_i : min(
                            [len(token_sequences), d_i + window_size_array[i, 0] + 1]
                        )
                    ]
                else:
                    multi_window = token_sequences[
                        max([0, d_i - window_size_array[i, 0]]) : d_i + 1
                    ]
                    multi_window.reverse()

                result_len = 0
                for mset in multi_window:
                    result_len += len(mset)

                this_window = np.zeros(result_len, dtype=np.int64)
                j = 0
                for mset in multi_window:
                    for x in mset:
                        this_window[j] = x
                        j += 1

                this_kernel = kernel_functions[i](multi_window, w_i, *kernel_args[i])

                windows.append(this_window)
                kernels.append(mix_weights[i] * this_kernel)

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

    for coo in coo_data:
        coo_sum_duplicates(coo)
        merge_all_sum_duplicates(coo)

    return coo_data


@numba.njit(nogil=True)
def numba_multi_em_cooccurrence_iteration(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_functions,
    kernel_args,
    mix_weights,
    n_unique_tokens,
    prior_indices,
    prior_indptr,
    prior_data,
):
    """
    Performs one round of EM on the given (hstack of) n cooccurrence matrices provided in csr format.

    Note: The algorithm assumes the matrix is an hstack of cooccurrence matrices with the same vocabulary,
    with kernel and window parameters given in the same order.

    Parameters
    ----------

    token_sequences: Iterable of Iterables of numpy.arrays
        The collection of multi set sequences to generate skip-gram data for.

    window_size_array : numpy.ndarray of shape(n, n_vocab)
        The collection of window sizes per token per directed cooccurrence

    window_reversals: numpy.array(bool)
        The collection of indicators whether or not the window is after the target token.

    kernel_functions: kernel_functions: tuple
        The n-tuple of kernel functions

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

    Returns
    -------
    posterior_data: numpy.array
        The data of the updated csr matrix after one iteration of EM.

    """

    posterior_data = np.zeros_like(prior_data)
    n_windows = window_size_array.shape[0]
    for d_i, seq in enumerate(token_sequences):
        for w_i, target_word in enumerate(seq):
            windows = []
            kernels = []
            for i in range(n_windows):
                if not window_reversals[i]:
                    multi_window = token_sequences[
                        d_i : min(
                            [len(token_sequences), d_i + window_size_array[i, 0] + 1]
                        )
                    ]
                else:
                    multi_window = token_sequences[
                        max([0, d_i - window_size_array[i, 0]]) : d_i + 1
                    ]
                    multi_window.reverse()

                this_kernel = kernel_functions[i](multi_window, w_i, *kernel_args[i])

                result_len = 0
                for mset in multi_window:
                    result_len += len(mset)

                this_window = np.zeros(result_len, dtype=np.int64)
                j = 0
                for mset in multi_window:
                    for x in mset:
                        this_window[j] = x
                        j += 1
                windows.append(this_window)
                kernels.append(mix_weights[i] * this_kernel)

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


class MultiSetCooccurrenceVectorizer(BaseCooccurrenceVectorizer):
    """Given a list of list of sequences of tokens, produce a collection of directed
    co-occurrence count matrix of tokens.  Here, each sequence of tokens is considered
    as a multiset of labels, hence this is treated as a list of lists of multisets when
    doing cooccurrence counting (with window radii and kernels applied across the multisets).

    Upon the construction of the count matrices, it will hstack them together and run
    n_iter iterations of EM to update the counts.

    Parameters
    ----------
    token_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary
        should be learned from the training data.

    max_unique_tokens: int or None (optional, default=None)
        The maximal number of elements contained in the vocabulary.  If not None, this is
        will prune the vocabulary to the top 'max_vocabulary_size' most frequent remaining tokens
        after other possible preprocessing.

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

    excluded_tokens: set or None (optional, default=None)
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

    n_threads: int (optional, default=1)
        When processing token sequences to build the matrix, break the list of sequences into
        n_threads equal sized chunks to process in parallel with Dask.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the
        provided mask_string.

    nullify_mask: bool (optional, default=False)
        Sets all cooccurrences with the mask_string equal to zero by skipping over them
        during processing.

    n_iter: int (optional, default = 0)
        Number of EM iterations to perform

    epsilon: float32 (optional default = 0)
        Sets values in the cooccurrence matrix (after l_1 normalizing the columns)
        less than epsilon to zero

    coo_initial_memory: str (optional, default = "0.5 GiB")
        This value, giving a memory size in k, M, G or T, describes how much memory
        to initialize for accumulating the (row, col, val) triples of larger data sets.
        Optimizations to use significantly less memory are made for data sets with small expected numbers of
        non zeros. More memory will be allocated during processing if need be.

    """

    def __init__(
        self,
        token_dictionary=None,
        max_unique_tokens=None,
        min_occurrences=None,
        max_occurrences=None,
        min_frequency=None,
        max_frequency=None,
        min_document_occurrences=None,
        max_document_occurrences=None,
        min_document_frequency=None,
        max_document_frequency=None,
        excluded_tokens=None,
        excluded_token_regex=None,
        window_functions="fixed",
        kernel_functions="flat",
        window_args=None,
        kernel_args=None,
        window_radii=5,
        mix_weights=None,
        window_orientations="directional",
        n_threads=1,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
        normalize_windows=True,
        n_iter=0,
        epsilon=0,
        coo_initial_memory="0.5 GiB",
    ):
        super().__init__(
            token_dictionary=token_dictionary,
            max_unique_tokens=max_unique_tokens,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_document_occurrences=min_document_occurrences,
            max_document_occurrences=max_document_occurrences,
            min_document_frequency=min_document_frequency,
            max_document_frequency=max_document_frequency,
            excluded_tokens=excluded_tokens,
            excluded_token_regex=excluded_token_regex,
            window_functions=window_functions,
            kernel_functions=kernel_functions,
            window_args=window_args,
            kernel_args=kernel_args,
            window_radii=window_radii,
            mix_weights=mix_weights,
            window_orientations=window_orientations,
            n_threads=n_threads,
            validate_data=validate_data,
            mask_string=mask_string,
            nullify_mask=nullify_mask,
            normalize_windows=normalize_windows,
            n_iter=n_iter,
            epsilon=epsilon,
            coo_initial_memory=coo_initial_memory,
        )
        self._preprocessing = preprocess_multi_token_sequences

    def _em_cooccurrence_iteration(self, token_sequences, cooccurrence_matrix):
        # call the numba function to return the new matrix.data
        result = [np.zeros_like(cooccurrence_matrix.data)]
        for seq in token_sequences:
            result.append(
                numba_multi_em_cooccurrence_iteration(
                    token_sequences=seq,
                    n_unique_tokens=len(self.token_label_dictionary_),
                    window_size_array=self._window_len_array,
                    window_reversals=self._window_reversals,
                    kernel_functions=self._kernel_functions,
                    kernel_args=self._full_kernel_args,
                    mix_weights=self._mix_weights,
                    prior_data=cooccurrence_matrix.data,
                    prior_indices=cooccurrence_matrix.indices,
                    prior_indptr=cooccurrence_matrix.indptr,
                )
            )
        return sum(result)

    def _build_skip_grams(self, token_sequences):
        # call the numba function for returning the list of CooArrays
        return numba_build_multi_skip_grams(
            token_sequences=token_sequences,
            window_size_array=self._window_len_array,
            window_reversals=self._window_reversals,
            kernel_functions=self._kernel_functions,
            kernel_args=self._full_kernel_args,
            mix_weights=self._mix_weights,
            normalize_windows=self.normalize_windows,
            n_unique_tokens=len(self.token_label_dictionary_),
            array_lengths=self._coo_sizes,
        )

    def _build_coo(self, token_sequences):
        result = []
        for seq in token_sequences:
            coo_data = self._build_skip_grams(seq)
            result.append(
                scipy.sparse.coo_matrix(
                    (
                        np.hstack([coo.val[: coo.ind[0]] for coo in coo_data]),
                        (
                            np.hstack([coo.row[: coo.ind[0]] for coo in coo_data]),
                            np.hstack([coo.col[: coo.ind[0]] for coo in coo_data]),
                        ),
                    ),
                    shape=(
                        self._n_rows,
                        len(self.token_label_dictionary_) * self._n_wide,
                    ),
                    dtype=np.float32,
                )
            )
        return sum(result)

    def _get_default_kernel_functions(self):
        return _MULTI_KERNEL_FUNCTIONS

    def _generate_chunk_boundaries(self, data, n_threads):
        token_list_sizes = np.array([sum([len(x) for x in seq]) for seq in data])
        cumulative_sizes = np.cumsum(token_list_sizes)
        chunk_size = np.ceil(cumulative_sizes[-1] / n_threads)
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
