from .preprocessing import (
    preprocess_token_sequences,
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
    dirichlet_process_normalize,
    l1_normalize_vector,
)

from .coo_utils import (
    coo_append,
    CooArray,
    set_array_size,
    em_update_matrix,
    generate_chunk_boundaries,
    coo_sum_duplicates,
    merge_all_sum_duplicates,
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
def numba_build_skip_grams(
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

    for coo in coo_data:
        coo_sum_duplicates(coo, kind="quicksort")
        merge_all_sum_duplicates(coo)

    return (
        [coo.row[: coo.ind[0]] for coo in coo_data],
        [coo.col[: coo.ind[0]] for coo in coo_data],
        [coo.val[: coo.ind[0]] for coo in coo_data],
    )


@numba.njit(nogil=True)
def numba_em_cooccurrence_iteration(
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


class BaseCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
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

    epsilon: float32 (optional default = 0)
        Sets values in the cooccurrence matrix (after l_1 normalizing the columns)
        less than epsilon to zero

    coo_initial_memory: str (optional, default = "0.5 GiB")
        This value, giving a memory size in k, M, G or T, describes how much memory
        to initialize for accumulating the (row, col, val) triples of larger data sets.
        Optimizations to use significantly less memory are made for data sets with small expected numbers of
        non zeros. More memory will be allocated during processing if need be.

    multi_labelled_tokens: bool (optional, default=False)
        Indicates whether your contexts are a sequence of bags of tokens labels
        with the context co-occurrence spanning the bags.  In other words if you have
        sequences of multi-labelled tokens.
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
        unknown_token=None,
        window_functions="fixed",
        kernel_functions="flat",
        window_args=None,
        kernel_args=None,
        window_radii=10,
        mix_weights=None,
        window_orientations="directional",
        chunk_size=1 << 20,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
        normalize_windows=True,
        n_iter=0,
        epsilon=0,
        coo_initial_memory="0.5 GiB",
    ):
        self.token_dictionary = token_dictionary
        self.max_unique_tokens = max_unique_tokens
        self.min_occurrences = min_occurrences
        self.min_frequency = min_frequency
        self.max_occurrences = max_occurrences
        self.max_frequency = max_frequency
        self.min_document_occurrences = min_document_occurrences
        self.min_document_frequency = min_document_frequency
        self.max_document_occurrences = max_document_occurrences
        self.max_document_frequency = max_document_frequency
        self.ignored_tokens = excluded_tokens
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
        self.validate_data = validate_data
        self.mask_string = mask_string
        self.nullify_mask = nullify_mask
        self.normalize_windows = normalize_windows
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.token_label_dictionary_ = {}
        self.token_index_dictionary_ = {}
        self.coo_max_bytes = str_to_bytes(coo_initial_memory)

        self._normalize = normalize
        self._window_normalize = l1_normalize_vector
        self._token_frequencies_ = np.array([])

        # Set attributes
        self.metric_ = distances.sparse_hellinger

        # Set params to be fit
        self._token_frequencies_ = None
        self.cooccurrences_ = None

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

    def _set_row_information(self, token_sequences):
        pass

    def _set_window_len_array(self):
        window_array = []
        for i, win_fn in enumerate(self._window_functions):
            window_array.append(
                win_fn(
                    self._window_radii[i],
                    self._token_frequencies_,
                    self._mask_index,
                    *self._window_args[i],
                )
            )
        self._window_len_array = np.array(window_array)

    def _set_mask_indices(self):
        if self.nullify_mask:
            self._mask_index = np.int32(len(self._token_frequencies_))
        else:
            self._mask_index = None

    def _set_full_kernel_args(self):
        # Set the kernel array and adjust args
        self._full_kernel_args = []

        for i, args in enumerate(self._kernel_args):
            default_kernel_array_args = {
                "mask_index": None,
                "normalize": False,
            }
            default_kernel_array_args.update(args)
            self._full_kernel_args.append(tuple(default_kernel_array_args.values()))

        self._full_kernel_args = tuple(self._full_kernel_args)

    def _set_coo_sizes(self, token_sequences):
        # Set the coo_array size
        approx_coo_size = 0
        for t in token_sequences:
            approx_coo_size += len(t)
        approx_coo_size *= (max(self.window_radii) + 1) * (20 * self._n_wide)
        if approx_coo_size < self.coo_max_bytes:
            self._coo_sizes = set_array_size(
                token_sequences,
                self._window_len_array_,
            )
        else:
            offsets = np.array(
                [self._initial_kernel_args[i][2] for i in range(self._n_wide)]
            )
            average_window = self._window_radii - offsets
            coo_sizes = (self.coo_max_bytes // 20) // np.sum(average_window)
            self._coo_sizes = np.array(coo_sizes * average_window, dtype=np.int64)
        if np.any(self._coo_sizes == 0):
            raise ValueError(f"The coo_initial_mem is too small to process any data.")

    def _set_additional_params(self):
        pass

    def _em_cooccurrence_iteration(self, token_sequences, cooccurrence_matrix):
        # call the numba function to return the new matrix.data
        return []

    def _build_skip_grams(self, token_sequences):
        # call the numba function for returning rows , cols, vals, for each window
        return [], [], []

    def _build_coo(self, token_sequences):
        coo_rows, coo_cols, coo_vals = self._build_skip_grams(token_sequences)
        result = scipy.sparse.coo_matrix(
            (
                np.hstack(coo_vals),
                (
                    np.hstack(coo_rows),
                    np.hstack(coo_cols),
                ),
            ),
            shape=(
                len(self._row_dict_),
                len(self.token_label_dictionary_) * self._n_wide,
            ),
            dtype=np.float32,
        )
        result.sum_duplicates()
        return result.tocsr()

    def _build_token_cooccurrence_matrix(self, token_sequences):
        """Generate a matrix of (weighted) counts of co-occurrences of tokens within
        windows in a set of sequences of tokens. Each sequence in the collection of
        sequences provides an effective boundary over which skip-grams may not pass
        (such as sentence boundaries in an NLP context). This is done for a collection
        of different window and kernel types simultaneously.

        Parameters
        ----------
        token_sequences: Iterable of Iterables
            The collection of token sequences to generate skip-gram data for.

        Returns
        -------
        cooccurrence_matrix: scipy.sparse.csr_matrix
            A matrix of shape (n_unique_tokens, n_windows*n_unique_tokens) where the i,j entry gives
            the (weighted) count of the number of times token i cooccurs within a
            window with token (j mod n_unique_tokens) for window/kernel function (j // n_unique_tokens).
        """

        matrix_per_chunk = [
            dask.delayed(
                self._build_coo, token_sequences=token_sequences[chunk_start:chunk_end]
            )
            for chunk_start, chunk_end in generate_chunk_boundaries(
                token_sequences, chunk_size=self.chunk_size
            )
        ]
        cooccurrence_matrix = dask.delayed(sum)(matrix_per_chunk)
        cooccurrence_matrix = cooccurrence_matrix.compute()
        cooccurrence_matrix.sum_duplicates()
        cooccurrence_matrix = cooccurrence_matrix.tocsr()

        if self.n_iter > 0 or self.epsilon > 0:
            cooccurrence_matrix = self.normalizer(
                cooccurrence_matrix, axis=0, norm="l1"
            ).tocsr()
            cooccurrence_matrix.data[cooccurrence_matrix.data < self.epsilon] = 0
            cooccurrence_matrix.eliminate_zeros()

        # Do the EM
        for iter in range(self.n_iter):
            new_data_per_chunk = [
                dask.delayed(self._em_cooccurrence_iteration)(
                    token_sequences=token_sequences[chunk_start:chunk_end],
                    cooccurrence_matrix = cooccurrence_matrix
                )
                for chunk_start, chunk_end in generate_chunk_boundaries(
                    token_sequences, chunk_size=self.chunk_size
                )
            ]
            new_data = dask.delayed(sum)(new_data_per_chunk)
            new_data = new_data.compute()
            cooccurrence_matrix.data = new_data
            cooccurrence_matrix = self.normalizer(
                cooccurrence_matrix, axis=0, norm="l1"
            ).tocsr()
            cooccurrence_matrix.data[cooccurrence_matrix.data < self.epsilon] = 0
            cooccurrence_matrix.eliminate_zeros()

        return cooccurrence_matrix.tocsr()

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
            max_unique_tokens=self.max_unique_tokens,
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

        if len(self.token_label_dictionary_) == 0:
            raise ValueError(
                "Token dictionary is empty; try using less extreme constraints"
            )

        # Set the row dict and frequencies
        self._set_row_information(token_sequences)

        # Set the row mask
        self._set_mask_indices()

        # Set column dicts
        self._set_column_dicts()

        # Set the window_lengths per row label
        self._set_window_len_array()

        # Update the kernel args to the tuple of default values with the added user inputs
        self._set_full_kernel_args()

        # Set the coo_array size
        self._set_coo_sizes()

        # Set any other things
        self._set_additional_params()

        # Build the matrix
        self.cooccurrences_ = self._build_token_cooccurrence_matrix(
            token_sequences,
        )

        return self.cooccurrences_

    def fit(self, X, y=None, **fit_params):
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
            max_unique_tokens=self.max_unique_tokens,
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

        if len(self.token_label_dictionary_) == 0:
            raise ValueError(
                "Token dictionary is empty; try using less extreme constraints"
            )

        # Set the row dict and frequencies
        self._set_row_information(token_sequences)

        # Set the row mask
        self._set_mask_indices()

        # Set column dicts
        self._set_column_dicts()

        # Set the window_lengths per row label
        self._set_window_len_array()

        # Update the kernel args to the tuple of default values with the added user inputs
        self._set_full_kernel_args()

        # Set the coo_array size
        self._set_coo_sizes()

        # Set any other things
        self._set_additional_params()

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

        cooccurrences_ = self._build_token_cooccurrence_matrix(
            token_sequences,
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
