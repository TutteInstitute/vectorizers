from ._vectorizers import token_cooccurrence_matrix, preprocess_token_sequences

from warnings import warn
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

import vectorizers.distances as distances

from .utils import validate_homogeneous_token_types, flatten

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
def em_cooccurrence_iteration(
    token_sequences,
    window_size_array,
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
    full_mix_weights = np.repeat(mix_weights, 2)
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
                windows.extend(
                    [
                        window_at_index(
                            token_sequences[seq_ind],
                            these_sizes[target_word],
                            w_i,
                            reverse=True,
                        ),
                        window_at_index(
                            token_sequences[seq_ind], these_sizes[target_word], w_i
                        ),
                    ]
                )

                kernels.extend(
                    [
                        update_kernel(
                            windows[-2], kernel_array[i], *kernel_args_array[i]
                        ),
                        update_kernel(
                            windows[-1], kernel_array[i], *kernel_args_array[i]
                        ),
                    ]
                )

            total_win_length = np.sum(np.array([len(w) for w in windows]))
            window_posterior = np.zeros(total_win_length, dtype=np.float32)
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
                                full_mix_weights[w]
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

    window_orientation: string (['before', 'after', 'symmetric', 'directional'])
        The orientation of the cooccurrence window.  Whether to return all the tokens that
        occurred within a window before, after on either side.
        symmetric: counts tokens occurring before and after as the same tokens
        directional: counts tokens before and after as different and returns both counts.

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
        window_functions=tuple(["fixed"]),
        kernel_functions=tuple(["flat"]),
        window_args=None,
        kernel_args=None,
        window_radii=tuple([5],),
        mix_weights=tuple([1],),
        chunk_size=1 << 20,
        validate_data=True,
        mask_string=None,
        nullify_mask=False,
        n_iter=1,
        epsilon=1e-8,
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
        self.n_iter = n_iter
        self.epsilon = epsilon

        assert len(self.window_radii) == len(self.kernel_functions)
        assert len(self.mix_weights) == len(self.window_functions)
        assert len(self.window_radii) == len(self.window_functions)
        self._n_wide = len(self.window_radii)

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

        # Set kernel functions
        self._kernel_functions = []
        for ker in self.kernel_functions:
            if callable(ker):
                self._kernel_functions.append(ker)
            elif ker in _KERNEL_FUNCTIONS:
                self._kernel_functions.append(_KERNEL_FUNCTIONS[ker])
            else:
                raise ValueError(
                    f"Unrecognized kernel_function; should be callable or one of {_KERNEL_FUNCTIONS.keys()}"
                )

        # Set window functions
        self._window_functions = []
        for win in self.window_functions:
            if callable(win):
                self._window_functions.append(win)
            elif win in _WINDOW_FUNCTIONS:
                self._window_functions.append(_WINDOW_FUNCTIONS[win])
            else:
                raise ValueError(
                    f"Unrecognized window_function; should be callable or one of {_WINDOW_FUNCTIONS.keys()}"
                )

        # Set mask nullity
        if self.nullify_mask:
            if self.mask_string is None:
                raise ValueError(f"Cannot nullify mask with mask_string = None")
            mask_index = len(self._token_frequencies_)
        else:
            mask_index = None

        # Set window args
        self._window_args = []
        if self.window_args is not None:
            for args in self.window_args:
                self._window_args.append(tuple(args.values()))
            self._window_args = tuple(self._window_args)
        else:
            self._window_args = tuple([tuple([]) for i in range(self._n_wide)])

        # Create the window size array
        self._window_sizes = []
        for i, win_fn in enumerate(self._window_functions):
            self._window_sizes.append(
                win_fn(
                    self.window_radii[i],
                    self._token_frequencies_,
                    mask_index,
                    *self._window_args[i],
                )
            )
        self._window_sizes = np.array(self._window_sizes)

        # Set kernel args and size array
        self._kernel_args = []
        self._kernel_array = []
        max_ker_len = np.max(self._window_sizes)
        if self.kernel_args is None:
            self.kernel_args = [dict() for i in range(self._n_wide)]
        for i, args in enumerate(self.kernel_args):
            default_global_args = {
                "mask_index": None,
                "normalize": False,
                "offset": 0,
            }
            default_global_args.update(args)
            default_global_args["normalize"] = False
            self._kernel_array.append(
                self._kernel_functions[i](
                    np.repeat(-1, max_ker_len),
                    self.window_radii[i],
                    *default_global_args.values(),
                )
            )
            default_global_args.update(args)
            self._kernel_args.append(
                tuple([mask_index, default_global_args["normalize"]])
            )
        self._kernel_args = tuple(self._kernel_args)
        self._kernel_array = np.array(self._kernel_array)

        self.cooccurrences_ = scipy.sparse.hstack(
            [
                token_cooccurrence_matrix(
                    token_sequences,
                    len(self.token_label_dictionary_),
                    kernel_function=self._kernel_functions[i],
                    kernel_args=self._kernel_args[i],
                    window_sizes=self._window_sizes[i],
                    window_orientation="directional",
                    chunk_size=self.chunk_size,
                )
                for i in range(self._n_wide)
            ]
        ).tocsr()

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
                    window_size_array=self._window_sizes,
                    kernel_array=self._kernel_array,
                    kernel_args_array=self._kernel_args,
                    mix_weights=np.array(self.mix_weights),
                    prior_data=self.cooccurrences_.data,
                    prior_indices=self.cooccurrences_.indices,
                    prior_indptr=self.cooccurrences_.indptr,
                )
            self.cooccurrences_.data = new_data
            self.cooccurrences_ = normalize(
                self.cooccurrences_, axis=0, norm="l1"
            ).tocsr()
            # self.cooccurrences_.eliminate_zeros()

        self.column_label_dictionary_ = {}
        for i in range(self._window_sizes.shape[0]):

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

        cooccurrences_ = scipy.sparse.hstack(
            [
                token_cooccurrence_matrix(
                    token_sequences,
                    len(self.token_label_dictionary_),
                    kernel_function=self._kernel_functions[i],
                    kernel_args=self._kernel_args[i],
                    window_sizes=self._window_sizes[i],
                    window_orientation="directional",
                    chunk_size=self.chunk_size,
                )
                for i in range(self._n_wide)
            ]
        ).tocsr()

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
                    window_size_array=self._window_sizes,
                    kernel_array=self._kernel_array,
                    kernel_args_array=self._kernel_args,
                    mix_weights=np.array(self.mix_weights),
                    prior_data=cooccurrences_.data,
                    prior_indices=cooccurrences_.indices,
                    prior_indptr=cooccurrences_.indptr,
                )
            cooccurrences_.data = new_data
            cooccurrences_ = normalize(cooccurrences_, axis=0, norm="l1").tocsr()
        return cooccurrences_
