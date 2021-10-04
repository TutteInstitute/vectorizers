import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_is_fitted

import scipy.linalg
import scipy.stats
import scipy.sparse

from .utils import (
    flatten,
    validate_homogeneous_token_types,
)

from .coo_utils import sum_coo_entries

from .preprocessing import preprocess_token_sequences
import vectorizers.distances as distances

from ._window_kernels import (
    _KERNEL_FUNCTIONS,
    _WINDOW_FUNCTIONS,
    window_at_index,
)


@numba.njit(nogil=True)
def build_skip_grams(
    token_sequence,
    window_sizes,
    kernel_function,
    kernel_args,
    reverse=False,
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

    coo_tuples = [(np.float32(0.0), np.float32(0.0), np.float32(0.0))]
    for i, head_token in enumerate(token_sequence):
        window = window_at_index(token_sequence, window_sizes[head_token], i, reverse)
        weights = kernel_function(window, *kernel_args)

        coo_tuples.extend(
            [
                (np.float32(head_token), np.float32(window[j]), weights[j])
                for j in range(len(window))
            ]
        )

    return sum_coo_entries(coo_tuples)


def skip_grams_matrix_coo_data(
    list_of_token_sequences,
    window_sizes,
    kernel_function,
    kernel_args,
):
    """Given a list of token sequences construct the relevant data for a sparse
    matrix representation with a row for each token sequence and a column for each
    skip-gram.

    Parameters
    ----------
    list_of_token_sequences: Iterable of Iterables
        The token sequences to construct skip-gram based representations of.

    window_sizes: Iterable
        A collection of window sizes per vocabulary index

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    kernel_args: tuple
        Arguments to pass through to the kernel function


    Returns
    -------
    row: array
        Row data for a COO format sparse matrix representation

    col: array
        Col data for a COO format sparse matrix representation

    data: array
        Value data for a COO format sparse matrix representation
    """
    result_row = []
    result_col = []
    result_data = []

    n_unique_tokens = len(window_sizes) - 1

    for row_idx in range(len(list_of_token_sequences)):
        skip_gram_data = build_skip_grams(
            list_of_token_sequences[row_idx],
            window_sizes,
            kernel_function,
            kernel_args,
        )
        for i, skip_gram in enumerate(skip_gram_data):
            result_row.append(row_idx)
            result_col.append(
                np.int32(skip_gram[0]) * n_unique_tokens + np.int32(skip_gram[1])
            )
            result_data.append(skip_gram[2])

    return np.asarray(result_row), np.asarray(result_col), np.asarray(result_data)


@numba.njit(nogil=True)
def sequence_skip_grams(
    token_sequences,
    window_sizes,
    kernel_function,
    kernel_args,
    reverse=False,
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
    skip_grams = [(np.float32(0), np.float32(0), np.float32(0))]
    for i, token_sequence in enumerate(token_sequences):
        skip_grams.extend(
            build_skip_grams(
                token_sequence,
                window_sizes,
                kernel_function,
                kernel_args,
                reverse,
            )
        )

    return np.array(sum_coo_entries(skip_grams))


class SkipgramVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a
    kernel weighted count matrix of ordered pairs of tokens occurring in a window.
    If passed a single sequence of tokens it  will use windows to determine cooccurrence.
    If passed a list of sequences of tokens it will use windows within each sequence in the list
    -- with windows not extending beyond the boundaries imposed by the individual sequences in the list.

    Parameters
    ----------
    token_dictionary: dictionary or None (optional, default=None)
        A fixed ditionary mapping tokens to indices, or None if the dictionary
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

    excluded_regex: str or None (optional, default=None)
        The regular expression by which tokens are ignored if re.fullmatch returns True.

    window_function: numba.jitted callable or str (optional, default='fixed')
        A function producing a sequence of window radii given a window_radius parameter and term frequencies.
        The string options are ['fixed', 'variable'] for using pre-defined functions.

    kernel_function: numba.jitted callable or str (optional, default='flat')
        A function producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic'] for using pre-defined functions.

    window_args: dict (optional, default = None)
        Optional arguments for the window function

    kernel_args: dict (optional, default = None)
        Optional arguments for the kernel function

    window_radius: int (optional, default=5)
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.

    nullify_mask: bool (optional, default=False)
        Sets all cooccurrences with the mask_string equal to zero by skipping over them during processing.
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
        window_function="fixed",
        kernel_function="flat",
        window_args=dict(),
        kernel_args=dict(),
        window_radius=5,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
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

        self.window_function = window_function
        self.kernel_function = kernel_function
        self.kernel_args = kernel_args
        self.window_args = window_args
        self.window_radius = window_radius
        self.validate_data = validate_data
        self.mask_string = (mask_string,)
        self.nullify_mask = (nullify_mask,)

    def fit(self, X, y=None, **fit_params):

        if self.validate_data:
            validate_homogeneous_token_types(X)

        flat_sequence = flatten(X)
        (
            token_sequences,
            self._token_dictionary_,
            self._inverse_token_dictionary_,
            self._token_frequencies_,
        ) = preprocess_token_sequences(
            X,
            flat_sequence,
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
        )

        if callable(self.kernel_function):
            self._kernel_function = self.kernel_function
        elif self.kernel_function in _KERNEL_FUNCTIONS:
            self._kernel_function = _KERNEL_FUNCTIONS[self.kernel_function]
        else:
            raise ValueError(
                f"Unrecognized kernel_function; should be callable or one of {_KERNEL_FUNCTIONS.keys()}"
            )

        if callable(self.window_function):
            self._window_function = self.window_function
        elif self.window_function in _WINDOW_FUNCTIONS:
            self._window_function = _WINDOW_FUNCTIONS[self.window_function]
        else:
            raise ValueError(
                f"Unrecognized window_function; should be callable or one of {_WINDOW_FUNCTIONS.keys()}"
            )

            # Update kernel and window args
        if self.nullify_mask:
            if self.mask_string is None:
                raise ValueError(f"Cannot suppress mask with mask_string = None")
            mask_index = len(self._token_frequencies_)
        else:
            mask_index = None

        self._kernel_args = {"mask_index": mask_index, "normalize": False, "offset": 0}
        self._kernel_args.update(self.kernel_args)
        self._kernel_args = tuple(self._kernel_args.values())

        self._window_sizes = self._window_function(
            self.window_radius,
            self._token_frequencies_,
            mask_index,
            *self.window_args.values(),
        )

        # Build the matrix
        row, col, data = skip_grams_matrix_coo_data(
            token_sequences,
            self._window_sizes,
            self._kernel_function,
            tuple(*self.kernel_args.values()),
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        column_sums = np.array(base_matrix.sum(axis=0))[0]
        self._column_is_kept = column_sums > 0
        self._kept_columns = np.where(self._column_is_kept)[0]

        self.column_label_dictionary_ = {}
        for i in range(self._kept_columns.shape[0]):
            raw_val = self._kept_columns[i]
            first_token = self._inverse_token_dictionary_[
                raw_val // len(self._token_dictionary_)
            ]
            second_token = self._inverse_token_dictionary_[
                raw_val % len(self._token_dictionary_)
            ]
            self.column_label_dictionary_[(first_token, second_token)] = i

        self.column_index_dictionary_ = {
            index: token for token, index in self.column_label_dictionary_.items()
        }
        self._train_matrix = base_matrix.tocsc()[:, self._column_is_kept].tocsr()
        self._train_matrix.eliminate_zeros()
        self.metric_ = distances.sparse_hellinger

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._train_matrix

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "_token_dictionary_",
                "_column_is_kept",
            ],
        )
        flat_sequence = flatten(X)
        (token_sequences, _, _, _) = preprocess_token_sequences(
            X,
            flat_sequence,
            self._token_dictionary_,
        )

        n_unique_tokens = len(self._token_dictionary_)

        row, col, data = skip_grams_matrix_coo_data(
            token_sequences,
            self._window_sizes,
            self._kernel_function,
            tuple(*self.kernel_args.values()),
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        result = base_matrix.tocsc()[:, self._column_is_kept].tocsr()

        return result
