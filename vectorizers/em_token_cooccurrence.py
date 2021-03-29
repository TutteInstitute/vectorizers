from ._vectorizers import (
    token_cooccurrence_matrix,
    preprocess_token_sequences,
    MOCK_DICT,
    pair_to_tuple,
    make_tuple_converter,
)

from warnings import warn
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from collections.abc import Iterable

import vectorizers.distances as distances

from .utils import (
    validate_homogeneous_token_types,
    flatten,
    coo_append,
    CooArray,
    coo_sum_duplicates,
)

import numpy as np
import numba
import scipy.sparse
from ._window_kernels import (
    _KERNEL_FUNCTIONS,
    _WINDOW_FUNCTIONS,
    window_at_index,
    update_kernel,
)


@numba.njit(nogil=True)
def build_multi_skip_ngrams(
    token_sequence,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args_array,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    ngram_dictionary,
    ngram_size=2,
    array_to_tuple=pair_to_tuple,
):
    """Given a single token sequence produce an array of weighted skip-grams
    associated to each token in the original sequence. The resulting array has
    shape (n_skip_grams, 3) where each skip_gram is a vector giving the
    head (or skip) token index, the tail token index, and the associated weight of the
    skip-gram. The weights for skip-gramming are given by the kernel_function
    that is applied to the window. Options for kernel functions include a fixed
    kernel (giving a weight of 1 to each item), a triangular kernel (giving a
    linear decaying weight depending on the distance from the skip token), and a
    harmonic kernel (giving a weight that decays inversely proportional to the
    distance from the skip_token).

    Parameters
    ----------
    token_sequence: Iterable
        The sequence of tokens to build skip-grams for.

    window_sizes: Iterable
        A collection of window sizes per vocabulary index

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    kernel_args: tuple
        Arguments to pass through to the kernel function

    reverse: bool (optional, default False)
        Whether windows follow the word (default) or are reversed and come
        before the word.

    Returns
    -------
    skip_grams: array of shape (n_skip_grams, 3)
        Each skip_gram is a vector giving the head (or skip) token index, the
        tail token index, and the associated weight of the skip-gram.
    """

    """
    coo_tuples = [(np.float32(0.0), np.float32(0.0), np.float32(0.0))]

    for i in range(ngram_size - 1, len(token_sequence)):
        if reverse:
            ngram = array_to_tuple(token_sequence[i : i + ngram_size])
        else:
            ngram = array_to_tuple(token_sequence[i - ngram_size : i])

        if ngram in ngram_dictionary:
            head_token = ngram_dictionary[ngram]
            window = window_at_index(
                token_sequence, window_sizes[head_token], i, reverse
            )
            weights = kernel_function(window, window_sizes[head_token], *kernel_args)

            coo_tuples.extend(
                [
                    (np.float32(head_token), np.float32(window[j]), weights[j])
                    for j in range(len(window))
                ]
            )
    
    return sum_coo_entries(coo_tuples)
    """
    return [(np.float32(0.0), np.float32(0.0), np.float32(0.0))]


@numba.njit(nogil=True)
def build_multi_skip_grams(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args_array,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    array_lens,
    array_merge_inds,
):
    """Given a single token sequence produce an array of weighted skip-grams
    associated to each token in the original sequence. The resulting array has
    shape (n_skip_grams, 3) where each skip_gram is a vector giving the
    head (or skip) token index, the tail token index, and the associated weight of the
    skip-gram. The weights for skip-gramming are given by the kernel_function
    that is applied to the window. Options for kernel functions include a fixed
    kernel (giving a weight of 1 to each item), a triangular kernel (giving a
    linear decaying weight depending on the distance from the skip token), and a
    harmonic kernel (giving a weight that decays inversely proportional to the
    distance from the skip_token).

    Parameters
    ----------
    token_sequence: Iterable
        The sequence of tokens to build skip-grams for.

    window_sizes: Iterable
        A collection of window sizes per vocabulary index

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    kernel_args: tuple
        Arguments to pass through to the kernel function

    reverse: bool (optional, default False)
        Whether windows follow the word (default) or are reversed and come
        before the word.

    Returns
    -------
    skip_grams: array of shape (n_skip_grams, 3)
        Each skip_gram is a vector giving the head (or skip) token index, the
        tail token index, and the associated weight of the skip-gram.
    """

    n_windows = window_size_array.shape[0]
    array_mul = n_windows * n_unique_tokens + 1
    kernel_masks = [ker[0] for ker in kernel_args_array]
    kernel_normalize = [ker[1] for ker in kernel_args_array]

    coo_data = [
        CooArray(
            np.zeros(array_lens[i], dtype=np.int32),
            np.zeros(array_lens[i], dtype=np.int32),
            np.zeros(array_lens[i], dtype=np.float32),
            np.zeros(array_lens[i], dtype=np.int64),
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
    kernel_args_array,
    mix_weights,
    normalize_windows,
    n_unique_tokens,
    array_lens,
    array_merge_inds,
):
    """Produce skip-gram data for a combined over a list of token sequences. In this
    case each token sequence represents a sequence with boundaries over which
    skip-grams may not extend (such as sentence boundaries in an NLP context).

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The token sequences to produce skip-gram data for

    window_sizes: Iterable
        A collection of window sizes per vocabulary index

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    kernel_args: tuple
        Arguments to pass through to the kernel function

    reverse: bool (optional, default False)
        Whether windows follow the word (default) or are reversed and come
        before the word.

    Returns
    -------
    skip_grams: array of shape (n_skip_grams, 3)
        The skip grams for the combined set of sequences.
    """

    coo_list = build_multi_skip_grams(
        token_sequences=token_sequences,
        window_size_array=window_size_array,
        window_reversals=window_reversals,
        kernel_array=kernel_array,
        kernel_args_array=kernel_args_array,
        mix_weights=mix_weights,
        normalize_windows=normalize_windows,
        n_unique_tokens=n_unique_tokens,
        array_lens=array_lens,
        array_merge_inds=array_merge_inds,
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
        tot_len += np.dot(window_array, counts).T
    return tot_len.astype(np.int64)


def multi_token_cooccurrence_matrix(
    token_sequences,
    n_unique_tokens,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args_array,
    mix_weights,
    normalize_windows,
    array_lens,
    array_merge_inds,
    chunk_size=1 << 20,
):
    """Generate a matrix of (weighted) counts of co-occurrences of tokens within
    windows in a set of sequences of tokens. Each sequence in the collection of
    sequences provides an effective boundary over which skip-grams may not pass
    (such as sentence boundaries in an NLP context). Options for how to generate
    windows and how to weight the counts with a window via a kernel are available.
    By default a fixed width window and a flat kernel are used, but other options
    include a variable width window based on total information within the window,
    and kernels can also be a triangular kernel (giving a
    linear decaying weight depending on the distance from the skip token), and a
    harmonic kernel (giving a weight that decays inversely proportional to the
    distance from the skip_token).

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    n_unique_tokens: int
        The number of unique tokens in the token_dictionary.

    window_sizes: Iterable
        A collection of window sizes per vocabulary index

    kernel_function: numba.jitted callable (optional, default=flat_kernel)
        A function producing weights given a window of tokens

    kernel_args: tuple (optional, default=())
        Arguments to pass through to the kernel function

    window_orientation: string (['before', 'after', 'symmetric', 'directional'])
        The orientation of the cooccurrence window.  Whether to return all the tokens that
        occurred within a window before, after on either side.
        symmetric: counts tokens occurring before and after as the same tokens
        directional: counts tokens before and after as different and returns both counts.

    chunk_size: int (optional, default=1048576)
        When processing token sequences, break the list of sequences into
        chunks of this size to stream the data through, rather than storing all
        the results at once. This saves on peak memory usage.

    Returns
    -------
    cooccurrence_matrix: scipyr.sparse.csr_matrix
        A matrix of shape (n_unique_tokens, n_unique_tokens) where the i,j entry gives
        the (weighted)  count of the number of times token i cooccurs within a
        window with token j.
    """
    if n_unique_tokens == 0:
        raise ValueError("Token dictionary is empty; try using less extreme contraints")

    # if len(ngram_dictionary) == 1 or ngram_size == 1:
    #    n_rows = n_unique_tokens
    #    array_to_tuple = pair_to_tuple  # Mock function for this case; unused
    # else:
    #    n_rows = len(ngram_dictionary)
    #   array_to_tuple = make_tuple_converter(ngram_size)

    cooccurrence_matrix = scipy.sparse.coo_matrix(
        (n_unique_tokens, window_size_array.shape[0] * n_unique_tokens)
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
            kernel_args_array=kernel_args_array,
            mix_weights=mix_weights,
            normalize_windows=normalize_windows,
            array_lens=array_lens,
            array_merge_inds=array_merge_inds,
        )

        cooccurrence_matrix += scipy.sparse.coo_matrix(
            (
                np.hstack(coo_vals),
                (
                    np.hstack(coo_rows),
                    np.hstack(coo_cols),
                ),
            ),
            shape=(n_unique_tokens, n_unique_tokens * window_size_array.shape[0]),
            dtype=np.float32,
        )

    cooccurrence_matrix.sum_duplicates()

    return cooccurrence_matrix.tocsr()


@numba.njit(nogil=True)
def em_cooccurrence_iteration(
    token_sequences,
    window_size_array,
    window_reversals,
    kernel_array,
    kernel_args_array,
    mix_weights,
    prior_indices,
    prior_indptr,
    prior_data,
):
    """
    Performs one round of EM on the given (hstack of) n directed cooccurrence matrices provided in csr format.

    Note: The algorithm assumes the matrix is an hstack of directed cooccurrence matrices with the same vocabulary,
    with the before columns first and after columns after, each corresponding to the kernel and window parameters
    given in the same order.

    Parameters
    ----------

    token_sequences: Iterable of Iterables
        The collection of token sequences to generate skip-gram data for.

    window_size_array : numpy.ndarray of shape(n, n_vocab)
        The collection of window sizes per token per directed cooccurrence

    kernel_array: numpy.ndarray of shape(n, max(window_size_array))
        The n-tuple of evaluated kernel functions of maximal length

    kernel_args_array: tuple(tuples)
        The n-tuple of update_kernel args per kernel function

    mix_weights: tuple
        The n-tuple of mix weights to apply to the kernel functions

    prior_indices:  numpy.array
        The csr indices of the hstacked directed cooccurrence matrices

    prior_indptr: numpy.array
        The csr indptr of the hstacked directed cooccurrence matrices

    prior_data: numpy.array
        The csr data of the hstacked directed cooccurrence matrices

    Returns
    -------
    posterior_data: numpy.array
        The data of the updated csr matrix after one iteration of EM.

    """

    n_vocab = prior_indptr.shape[0] - 1
    posterior_data = np.zeros_like(prior_data)

    for seq_ind in range(len(token_sequences)):

        for w_i in range(len(token_sequences[seq_ind])):
            #
            #  Partial E_step - Compute the new expectation per context for this window
            #

            target_word = token_sequences[seq_ind][w_i]
            windows = []
            kernels = []
            for i in range(len(window_size_array)):
                these_sizes = window_size_array[i]
                windows.append(
                    window_at_index(
                        token_sequences[seq_ind],
                        these_sizes[target_word],
                        w_i,
                        reverse=window_reversals[i],
                    )
                )

                kernels.append(
                    update_kernel(windows[-1], kernel_array[i], *kernel_args_array[i])
                )

            total_win_length = np.sum(np.array([len(w) for w in windows]))
            window_posterior = np.zeros(total_win_length)
            context_ind = np.zeros(total_win_length, dtype=np.int64)
            win_offset = np.append(
                np.zeros(1, dtype=np.int64),
                np.cumsum(np.array([len(w) for w in windows])),
            )[:-1]

            col_ind = prior_indices[
                prior_indptr[target_word] : prior_indptr[target_word + 1]
            ]

            for w, window in enumerate(windows):
                for i, context in enumerate(window):
                    if kernels[w][i] > 0:
                        context_ind[i + win_offset[w]] = np.searchsorted(
                            col_ind, context + w * n_vocab
                        )
                        # assert(col_ind[context_ind[i + win_offset[w]]] == context+w*n_vocab)
                        if (
                            col_ind[context_ind[i + win_offset[w]]]
                            == context + w * n_vocab
                        ):
                            window_posterior[i + win_offset[w]] = (
                                mix_weights[w]
                                * kernels[w][i]
                                * prior_data[
                                    prior_indptr[target_word]
                                    + context_ind[i + win_offset[w]]
                                ]
                            )
                        else:
                            window_posterior[i + win_offset[w]] = 0

            temp = window_posterior.sum()
            if temp > 0:
                window_posterior /= temp

            #
            # Partial M_step - Update the posteriors
            #

            for w, window in enumerate(windows):
                for i, context in enumerate(window):
                    val = window_posterior[i + win_offset[w]]
                    if val > 0:
                        posterior_data[
                            prior_indptr[target_word] + context_ind[i + win_offset[w]]
                        ] += val

    return posterior_data


class EMTokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
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

    window_functions: tuple of numba.jitted callable or str (optional, default=['fixed'])
        Functions producing a sequence of window radii given a window_radius parameter and term frequencies.
        The string options are ['fixed', 'variable'] for using pre-defined functions.

    kernel_functions: tuple of numba.jitted callable or str (optional, default=['flat'])
        Functions producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic', 'negative_binomial'] for using pre-defined functions.

    window_radii: tuple of int (optional, default=[5])
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    window_args: tuple of dicts (optional, default = None)
        Optional arguments for the window function

    kernel_args: tuple of dicts (optional, default = None)
        Optional arguments for the kernel function

    window_orientations: Iterable of strings (['before', 'after', 'directional'])
        The orientations of the cooccurrence windows.  Whether to return all the tokens that
        occurred within a window before, after, or on either side separately.

    chunk_size: int (optional, default=1048576)
        When processing token sequences, break the list of sequences into
        chunks of this size to stream the data through, rather than storing all
        the results at once. This saves on peak memory usage.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.

    nullify_mask: bool (optional, default=False)
        Sets all cooccurrences with the mask_string equal to zero by skipping over them during processing.

    n_iter: int (optional, default = 1)
        Number of EM iterations to perform

    epsilon: float32 (optional default = 1e-8)
        Sets values in the cooccurrence matrix less than epsilon to zero
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
        mix_weights=[1],
        window_orientations="directional",
        chunk_size=1 << 20,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
        normalize_windows=False,
        n_iter=1,
        epsilon=1e-11,
        coo_max_bytes=2 << 30,
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
        self.validate_data = validate_data
        self.mask_string = mask_string
        self.nullify_mask = nullify_mask
        self.normalize_windows = normalize_windows
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.coo_max_bytes = coo_max_bytes

        # Check the window orientations
        if not isinstance(self.window_radii, Iterable):
            self.window_radii = [self.window_radii]
        if isinstance(self.window_orientations, str):
            self.window_orientations = [
                self.window_orientations for i in self.window_radii
            ]

    def fit_transform(self, X, y=None, **fit_params):

        if self.validate_data:
            validate_homogeneous_token_types(X)

        flat_sequences = flatten(X)
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

        self._window_reversals = []
        self._window_orientations = []
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
            mask_index = len(self._token_frequencies_)
        else:
            mask_index = None

        # Set window args
        self._window_args = []
        if isinstance(self.window_args, dict):
            self._window_args = tuple(
                [tuple(self.window_args.values()) for i in range(self._n_wide)]
            )
        elif self.window_args is None:
            self._window_args = tuple([tuple([]) for i in range(self._n_wide)])
        else:
            for i, args in enumerate(self.window_args):
                self._window_args.append(tuple(args.values()))
                if self.window_orientations[i] == "directional":
                    self._window_args.append(tuple(args.values()))
            self._window_args = tuple(self._window_args)

        # Set initial kernel args
        if isinstance(self.kernel_args, dict):
            self._kernel_args = [self.kernel_args for i in range(self._n_wide)]
        elif self.kernel_args is None:
            self._kernel_args = [dict([]) for i in range(self._n_wide)]
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

        # Set the window array
        self._window_array = []
        for i, win_fn in enumerate(self._window_functions):
            self._window_array.append(
                win_fn(
                    self._window_radii[i],
                    self._token_frequencies_,
                    mask_index,
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

        ## Set the coo_array size

        approx_coo_size = 0
        for t in token_sequences:
            approx_coo_size += len(t)
        approx_coo_size *= (max(self.window_radii) + 1) * (20 * (self._n_wide))
        if approx_coo_size < self.coo_max_bytes:
            self._coo_sizes = set_array_size(
                token_sequences,
                self._window_array,
            )
            self._merge_inds = self._coo_sizes + 1
        else:
            offsets = np.array(
                [self._initial_kernel_args[i][2] for i in range(self._n_wide)]
            )
            average_window = self._window_radii - offsets
            self._coo_sizes = (self.coo_max_bytes // 20) // np.sum(average_window)
            self._coo_sizes = np.array(self._coo_sizes * average_window, dtype=np.int64)
            self._merge_inds = np.repeat(2 ** 16, self._n_wide)

        self.cooccurrences_ = multi_token_cooccurrence_matrix(
            token_sequences,
            len(self.token_label_dictionary_),
            window_size_array=self._window_array,
            window_reversals=self._window_reversals,
            kernel_array=self._kernel_array,
            kernel_args_array=self._em_kernel_args,
            mix_weights=self._mix_weights,
            chunk_size=self.chunk_size,
            normalize_windows=self.normalize_windows,
            array_lens=self._coo_sizes,
            array_merge_inds=self._merge_inds,
        )

        self.cooccurrences_ = normalize(self.cooccurrences_, axis=0, norm="l1").tocsr()
        self.cooccurrences_.data[self.cooccurrences_.data < self.epsilon] = 0
        self.cooccurrences_.eliminate_zeros()
        self.cooccurrences_ = normalize(self.cooccurrences_, axis=0, norm="l1").tocsr()

        # Do the EM
        n_chunks = (len(token_sequences) // self.chunk_size) + 1
        for iter in range(self.n_iter):
            new_data = np.zeros_like(self.cooccurrences_.data)
            for chunk_index in range(n_chunks):
                chunk_start = chunk_index * self.chunk_size
                chunk_end = min(len(token_sequences), chunk_start + self.chunk_size)
                new_data += em_cooccurrence_iteration(
                    token_sequences=token_sequences[chunk_start:chunk_end],
                    window_size_array=self._window_array,
                    window_reversals=self._window_reversals,
                    kernel_array=self._kernel_array,
                    kernel_args_array=self._em_kernel_args,
                    mix_weights=self._mix_weights,
                    prior_data=self.cooccurrences_.data,
                    prior_indices=self.cooccurrences_.indices,
                    prior_indptr=self.cooccurrences_.indptr,
                )
            self.cooccurrences_.data = new_data
            self.cooccurrences_ = normalize(
                self.cooccurrences_, axis=0, norm="l1"
            ).tocsr()
            self.cooccurrences_.data[self.cooccurrences_.data < self.epsilon] = 0
            self.cooccurrences_.eliminate_zeros()
            self.cooccurrences_ = normalize(
                self.cooccurrences_, axis=0, norm="l1"
            ).tocsr()

        self.column_label_dictionary_ = {}
        for i in range(self._window_array.shape[0]):

            self.column_label_dictionary_.update(
                {
                    "pre_"
                    + str(i)
                    + "_"
                    + str(token): index
                    + 2 * i * len(self.token_label_dictionary_)
                    for token, index in self.token_label_dictionary_.items()
                }
            )
            self.column_label_dictionary_.update(
                {
                    "post_"
                    + str(i)
                    + "_"
                    + str(token): index
                    + (2 * i + 1) * len(self.token_label_dictionary_)
                    for token, index in self.token_label_dictionary_.items()
                }
            )

        self.column_index_dictionary_ = {
            item[1]: item[0] for item in self.column_label_dictionary_.items()
        }
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
            kernel_args_array=self._em_kernel_args,
            mix_weights=self._mix_weights,
            chunk_size=self.chunk_size,
            normalize_windows=self.normalize_windows,
            array_lens=self._coo_sizes,
            array_merge_inds=self._merge_inds,
        )

        cooccurrences_ = normalize(cooccurrences_, axis=0, norm="l1").tocsr()
        cooccurrences_.data[cooccurrences_.data < self.epsilon] = 0
        cooccurrences_.eliminate_zeros()
        cooccurrences_ = normalize(cooccurrences_, axis=0, norm="l1").tocsr()

        # Do the EM
        n_chunks = (len(token_sequences) // self.chunk_size) + 1

        for iter in range(self.n_iter):
            new_data = np.zeros_like(cooccurrences_.data)
            for chunk_index in range(n_chunks):
                chunk_start = chunk_index * self.chunk_size
                chunk_end = min(len(token_sequences), chunk_start + self.chunk_size)
                new_data += em_cooccurrence_iteration(
                    token_sequences=token_sequences[chunk_start:chunk_end],
                    window_size_array=self._window_array,
                    window_reversals=self._window_reversals,
                    kernel_array=self._kernel_array,
                    kernel_args_array=self._em_kernel_args,
                    mix_weights=self._mix_weights,
                    prior_data=cooccurrences_.data,
                    prior_indices=cooccurrences_.indices,
                    prior_indptr=cooccurrences_.indptr,
                )
            cooccurrences_.data = new_data
            cooccurrences_ = normalize(cooccurrences_, axis=0, norm="l1").tocsr()
            cooccurrences_.data[cooccurrences_.data < self.epsilon] = 0
            cooccurrences_.eliminate_zeros()
            cooccurrences_ = normalize(cooccurrences_, axis=0, norm="l1").tocsr()

        return cooccurrences_
