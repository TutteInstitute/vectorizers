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
)

from .coo_utils import CooArray, COO_QUICKSORT_LIMIT

import numpy as np
import numba
from numba.typed import List
import dask
import scipy.sparse
from ._window_kernels import (
    _KERNEL_FUNCTIONS,
    _WINDOW_FUNCTIONS,
)


class BaseCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a horizontal join of a
    collection of directed co-occurrence count matrices of tokens. If passed a single
    sequence of tokens it will use windows to determine co-occurrence. If passed a list
    of sequences of tokens it will use windows within each sequence in the list -- with
    windows not extending beyond the boundaries imposed by the individual sequences in the list.

    Upon the construction of the individual count matrices, it will hstack them together and run
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
        coo_initial_memory="1 GiB",
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
        self.excluded_tokens = excluded_tokens
        self.excluded_token_regex = excluded_token_regex
        self.window_orientations = window_orientations
        self.window_functions = window_functions
        self.kernel_functions = kernel_functions
        self.window_args = window_args
        self.kernel_args = kernel_args
        self.mix_weights = mix_weights
        self.window_radii = window_radii
        self.n_threads = n_threads
        self.validate_data = validate_data
        self.mask_string = mask_string
        self.nullify_mask = nullify_mask
        self.normalize_windows = normalize_windows
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.token_label_dictionary_ = {}
        self.token_index_dictionary_ = {}
        self.coo_initial_bytes = str_to_bytes(coo_initial_memory)

        # Set attributes
        self.metric_ = distances.sparse_hellinger
        self._preprocessing = preprocess_token_sequences

        # Set params to be fit
        self._token_frequencies_ = np.array([])
        self.cooccurrences_ = None
        self.reduced_matrix_ = None

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

        built_in_kernels = self._get_default_kernel_functions()
        self._kernel_functions = []
        for i, ker in enumerate(self.kernel_functions):
            if callable(ker):
                self._kernel_functions.append(ker)
            elif ker in built_in_kernels:
                self._kernel_functions.append(built_in_kernels[ker])
            else:
                raise ValueError(
                    f"Unrecognized kernel_function; should be callable or one of {built_in_kernels.keys()}"
                )
            if self.window_orientations[i] == "directional":
                self._kernel_functions.append(self._kernel_functions[-1])
        if len(set(self._kernel_functions)) > 1:
            raise NotImplementedError(f"All kernel functions must be the same.")
        self._kernel_functions = tuple(self._kernel_functions)

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

        # Check a few other inputs
        assert self.n_threads > 0
        assert self.n_iter >= 0
        assert self.epsilon >= 0

    def _get_default_kernel_functions(self):
        return _KERNEL_FUNCTIONS

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
        self._n_rows = len(self.token_label_dictionary_)

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
        # Set the full kernel args
        self._full_kernel_args = numba.typed.List([])

        for i, args in enumerate(self._kernel_args):
            default_kernel_array_args = {
                "mask_index": self._mask_index,
                "normalize": False,
                "offset": 0,
            }
            default_kernel_array_args.update(args)
            self._full_kernel_args.append(tuple(default_kernel_array_args.values()))

    def _set_coo_sizes(self, token_sequences):
        # Set the coo_array size
        approx_coo_size = 0
        for t in token_sequences:
            approx_coo_size += len(t)
        approx_coo_size *= (max(self.window_radii) + 1) * (20 * self._n_wide)
        if approx_coo_size < self.coo_initial_bytes:
            self._coo_sizes = np.repeat(
                approx_coo_size // self._n_wide, self._n_wide
            ).astype(np.int64)
        else:
            offsets = np.array(
                [self._full_kernel_args[i][2] for i in range(self._n_wide)]
            )
            average_window = self._window_radii - offsets
            coo_sizes = (self.coo_initial_bytes // 20) // np.sum(average_window)
            self._coo_sizes = np.array(coo_sizes * average_window, dtype=np.int64)

        self._coo_sizes = np.divmod(self._coo_sizes, self.n_threads)[0]

    def _generate_chunk_boundaries(self, data, n_threads):
        token_list_sizes = np.array([len(x) for x in data])
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

    def _set_additional_params(self, token_sequences):
        pass

    def _em_cooccurrence_iteration(self, token_sequences, cooccurrence_matrix):
        # call the numba function to return the new matrix.data
        return np.array([])

    def _build_skip_grams(self, token_sequences):
        # call the numba function for returning the list of CooArrays
        return CooArray()

    def _build_coo(self, token_sequences):
        coo_data = self._build_skip_grams(token_sequences)
        result = scipy.sparse.coo_matrix(
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
        return result

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
        if self.n_threads > 1:
            matrix_per_chunk = [
                dask.delayed(self._build_coo)(
                    token_sequences=token_sequences[chunk_start:chunk_end],
                )
                for chunk_start, chunk_end in self._generate_chunk_boundaries(
                    token_sequences, self.n_threads
                )
            ]
            cooccurrence_matrix = dask.delayed(sum)(matrix_per_chunk)
            cooccurrence_matrix = cooccurrence_matrix.compute()
        else:
            cooccurrence_matrix = self._build_coo(token_sequences=token_sequences)

        cooccurrence_matrix.sum_duplicates()
        cooccurrence_matrix = cooccurrence_matrix.tocsr()

        if self.n_iter > 0 or self.epsilon > 0:
            cooccurrence_matrix = normalize(
                cooccurrence_matrix, axis=0, norm="l1"
            ).tocsr()
            cooccurrence_matrix.data[cooccurrence_matrix.data < self.epsilon] = 0
            cooccurrence_matrix.eliminate_zeros()

        # Do the EM
        for i in range(self.n_iter):
            if self.n_threads > 1:
                new_data_per_chunk = [
                    dask.delayed(self._em_cooccurrence_iteration)(
                        token_sequences=token_sequences[chunk_start:chunk_end],
                        cooccurrence_matrix=cooccurrence_matrix,
                    )
                    for chunk_start, chunk_end in self._generate_chunk_boundaries(
                        token_sequences, self.n_threads
                    )
                ]
                new_data = dask.delayed(sum)(new_data_per_chunk)
                new_data = new_data.compute()
            else:
                new_data = self._em_cooccurrence_iteration(
                    token_sequences=token_sequences,
                    cooccurrence_matrix=cooccurrence_matrix,
                )
            cooccurrence_matrix.data = new_data
            cooccurrence_matrix = normalize(
                cooccurrence_matrix, axis=0, norm="l1"
            ).tocsr()
            cooccurrence_matrix.data[cooccurrence_matrix.data < self.epsilon] = 0
            cooccurrence_matrix.eliminate_zeros()

        return cooccurrence_matrix.tocsr()

    def fit_transform(self, X, y=None, **fit_params):

        if self.validate_data:
            validate_homogeneous_token_types(X)

        # noinspection PyTupleAssignmentBalance
        (
            token_sequences,
            self.token_label_dictionary_,
            self.token_index_dictionary_,
            self._token_frequencies_,
        ) = self._preprocessing(
            X,
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
            ignored_tokens=self.excluded_tokens,
            excluded_token_regex=self.excluded_token_regex,
            masking=self.mask_string,
        )

        if len(self.token_label_dictionary_) == 0:
            raise ValueError(
                "Token dictionary is empty; try using less extreme constraints"
            )

        # Set the row dict and frequencies
        self._set_row_information(token_sequences)

        # Set any other things
        self._set_additional_params(token_sequences)

        # Set the row mask
        self._set_mask_indices()

        # Set column dicts
        self._set_column_dicts()

        # Set the window_lengths per row label
        self._set_window_len_array()

        # Update the kernel args to the tuple of default values with the added user inputs
        self._set_full_kernel_args()

        # Set the coo_array size
        self._set_coo_sizes(token_sequences)

        # Build the matrix
        self.cooccurrences_ = self._build_token_cooccurrence_matrix(
            token_sequences,
        )

        return self.cooccurrences_

    def fit(self, X, y=None, **fit_params):
        if self.validate_data:
            validate_homogeneous_token_types(X)

        # noinspection PyTupleAssignmentBalance
        (
            token_sequences,
            self.token_label_dictionary_,
            self.token_index_dictionary_,
            self._token_frequencies_,
        ) = self._preprocessing(
            X,
            token_dictionary=self.token_dictionary,
            max_unique_tokens=self.max_unique_tokens,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            min_document_occurrences=self.min_document_occurrences,
            max_document_occurrences=self.max_document_occurrences,
            min_document_frequency=self.min_document_frequency,
            max_document_frequency=self.max_document_frequency,
            ignored_tokens=self.excluded_tokens,
            excluded_token_regex=self.excluded_token_regex,
            masking=self.mask_string,
        )

        if len(self.token_label_dictionary_) == 0:
            raise ValueError(
                "Token dictionary is empty; try using less extreme constraints"
            )

        # Set the row dict and frequencies
        self._set_row_information(token_sequences)

        # Set any other things
        self._set_additional_params(token_sequences)

        # Set the row mask
        self._set_mask_indices()

        # Set column dicts
        self._set_column_dicts()

        # Set the window_lengths per row label
        self._set_window_len_array()

        # Update the kernel args to the tuple of default values with the added user inputs
        self._set_full_kernel_args()

        # Set the coo_array size
        self._set_coo_sizes(token_sequences)

        # Build the matrix
        self.cooccurrences_ = self._build_token_cooccurrence_matrix(
            token_sequences,
        )

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

        # noinspection PyTupleAssignmentBalance
        (
            token_sequences,
            column_label_dictionary,
            column_index_dictionary,
            token_frequencies,
        ) = self._preprocessing(
            X, token_dictionary=self.token_label_dictionary_, masking=self.mask_string
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
        power=0.25,
    ):
        check_is_fitted(self, ["column_label_dictionary_"])

        if self.n_iter < 1:
            self.reduced_matrix_ = normalize(self.cooccurrences_, axis=0, norm="l1")
            self.reduced_matrix_ = normalize(self.reduced_matrix_, axis=1, norm="l1")
        else:
            self.reduced_matrix_ = normalize(self.cooccurrences_, axis=1, norm="l1")

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
