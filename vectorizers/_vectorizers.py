"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from collections import defaultdict
import scipy.linalg
import scipy.stats
import scipy.sparse
from typing import Union, Sequence, AnyStr

from vectorizers.utils import (
    flatten,
    vectorize_diagram,
    pairwise_gaussian_ground_distance,
)
import vectorizers.distances as distances


@numba.njit(nogil=True)
def construct_token_dictionary_and_frequency(token_sequence, token_dictionary=None):
    n_tokens = len(token_sequence)
    if token_dictionary is None:
        unique_tokens = sorted(list(set(token_sequence)))
        token_dictionary = dict(zip(unique_tokens, range(len(unique_tokens))))

    index_list = [
        token_dictionary[token] for token in token_sequence if token in token_dictionary
    ]
    token_counts = np.bincount(index_list).astype(np.float32)

    token_frequency = token_counts / n_tokens

    return token_dictionary, token_frequency, n_tokens


@numba.njit(nogil=True)
def prune_token_dictionary(
    token_dictionary,
    token_frequencies,
    ignored_tokens=None,
    min_frequency=0.0,
    max_frequency=1.0,
):

    if ignored_tokens is not None:
        tokens_to_prune = set(ignored_tokens)
    else:
        tokens_to_prune = set([])

    reverse_vocabulary = {index: word for word, index in token_dictionary.items()}

    infrequent_tokens = np.where(token_frequencies <= min_frequency)[0]
    frequent_tokens = np.where(token_frequencies >= max_frequency)[0]

    tokens_to_prune.update({reverse_vocabulary[i] for i in infrequent_tokens})
    tokens_to_prune.update({reverse_vocabulary[i] for i in frequent_tokens})

    vocab_tokens = [token for token in token_dictionary if token not in tokens_to_prune]
    new_vocabulary = dict(zip(vocab_tokens, range(len(vocab_tokens))))
    new_token_frequency = np.array(
        [token_frequencies[token_dictionary[token]] for token in new_vocabulary]
    )

    return new_vocabulary, new_token_frequency


@numba.njit(nogil=True)
def preprocess_token_sequences(
    token_sequences,
    token_dictionary=None,
    min_occurrences=None,
    max_occurrences=None,
    min_frequency=None,
    max_frequency=None,
    ignored_tokens=None,
):
    flat_sequence = flatten(token_sequences)

    # Get vocabulary and word frequencies
    (
        token_dictonary,
        token_frequencies,
        total_tokens,
    ) = construct_token_dictionary_and_frequency(flat_sequence, token_dictionary)

    if min_occurrences is None:
        if min_frequency is None:
            min_frequency = 0.0
    else:
        if min_frequency is not None:
            assert min_occurrences / total_tokens == min_frequency
        else:
            min_frequency = min_occurrences / total_tokens

    if max_occurrences is None:
        if max_frequency is None:
            max_frequency = 1.0
    else:
        if max_frequency is not None:
            assert max_occurrences / total_tokens == max_frequency
        else:
            max_frequency = min(1.0, max_occurrences / total_tokens)

    token_dictionary, token_frequencies = prune_token_dictionary(
        token_dictionary,
        token_frequencies,
        ignored_tokens=ignored_tokens,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    result_sequences = [
        np.array(
            [token_dictionary[token] for token in sequence if token in token_dictionary]
        )
        for sequence in token_sequences
    ]

    return result_sequences, token_dictionary, token_frequencies


@numba.njit(nogil=True)
def information_window(token_sequence, desired_entropy, token_frequency):
    result = []

    for i in range(len(token_sequence)):
        counter = 0
        current_entropy = 0.0

        for j in range(i + 1, len(token_sequence)):
            current_entropy -= np.log(token_frequency[int(token_sequence[j])])
            counter += 1
            if current_entropy >= desired_entropy:
                break

        result.append(token_sequence[i + 1 : i + 1 + counter])

    return result


@numba.njit(nogil=True)
def fixed_window(token_sequence, window_size):
    result = []

    for i in range(len(token_sequence)):
        result.append(token_sequence[i + 1 : i + window_size + 1])

    return result


@numba.njit(nogil=True)
def flat_kernel(window):
    return np.ones(len(window), dtype=np.float32)


@numba.njit(nogil=True)
def triangle_kernel(window, window_size):
    start = max(window_size, len(window))
    stop = window_size - len(window)
    return np.arange(start, stop, -1).astype(np.float32)


@numba.njit(nogil=True)
def harmonic_kernel(window):
    result = np.arange(1, len(window) + 1).astype(np.float32)
    return 1.0 / result


@numba.njit(nogil=True)
def build_skip_grams(
    token_sequence, window_function, kernel_function, window_args, kernel_args
):
    original_tokens = token_sequence
    n_original_tokens = len(original_tokens)

    if n_original_tokens < 2:
        return np.zeros((1, 3), dtype=np.float32)

    windows = window_function(token_sequence, *window_args)

    new_tokens = np.empty(
        (np.sum(np.array([len(w) for w in windows])), 3), dtype=np.float32
    )
    new_token_count = 0

    for i in range(n_original_tokens):
        head_token = original_tokens[i]
        window = windows[i]
        weights = kernel_function(window, *kernel_args)

        for j in range(len(window)):
            new_tokens[new_token_count, 0] = numba.types.float32(head_token)
            new_tokens[new_token_count, 1] = numba.types.float32(window[j])
            new_tokens[new_token_count, 2] = weights[j]
            new_token_count += 1

    return new_tokens


@numba.njit(nogil=True)
def skip_grams_matrix_coo_data(
    list_of_token_sequences,
    window_function,
    kernel_function,
    window_args,
    kernel_args,
    n_tokens,
):
    result_row = [numba.types.int32(0) for i in range(0)]
    result_col = [numba.types.int32(0) for i in range(0)]
    result_data = [numba.types.float32(0) for i in range(0)]

    for row_idx in range(len(list_of_token_sequences)):
        skip_gram_data = build_skip_grams(
            list_of_token_sequences[row_idx],
            window_function,
            kernel_function,
            window_args,
            kernel_args,
        )
        for skip_gram in skip_gram_data:
            result_row.append(row_idx)
            result_col.append(
                numba.types.int32(skip_gram[0]) * n_tokens
                + numba.types.int32(skip_gram[1])
            )
            result_data.append(skip_gram[2])

    return np.array(result_row), np.array(result_col), np.array(result_data)


@numba.njit(nogil=True, parallel=True)
def sequence_skip_grams(
    token_sequences, window_function, kernel_function, window_args, kernel_args
):
    skip_grams_per_sequence = [
        build_skip_grams(
            token_sequence, window_function, kernel_function, window_args, kernel_args
        )
        for token_sequence in token_sequences
    ]
    return np.vstack(skip_grams_per_sequence)


def token_cooccurence_matrix(
    token_sequences,
    window_function=fixed_window,
    kernel_function=flat_kernel,
    window_args=(5,),
    kernel_args=(),
    token_dictionary=None,
    symmetrize=False,
):

    raw_coo_data = sequence_skip_grams(
        token_sequences, window_function, kernel_function, window_args, kernel_args
    )
    cooccurrence_matrix = scipy.sparse.coo_matrix(
        (
            raw_coo_data.T[2],
            (raw_coo_data.T[0].astype(np.int64), raw_coo_data.T[1].astype(np.int64)),
        ),
        dtype=np.float32,
    )
    if symmetrize:
        cooccurrence_matrix = cooccurrence_matrix + cooccurrence_matrix.transpose()

    index_dictionary = {index: token for token, index in token_dictionary.items()}

    return cooccurrence_matrix.tocsr(), token_dictionary, index_dictionary


@numba.njit(nogil=True)
def ngrams_of(sequence, ngram_size, ngram_behaviour="exact"):
    result = []
    for i in range(len(sequence)):
        if ngram_behaviour == "exact":
            if i + ngram_size <= len(sequence):
                result.append(sequence[i : i + ngram_size])
        elif ngram_behaviour == "subgrams":
            for j in range(1, ngram_size + 1):
                if i + j <= len(sequence):
                    result.append(sequence[i : i + j])
        else:
            raise ValueError("Unrecognized ngram_behaviour!")
    return result


class TokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a
    co-occurrence count matrix of tokens. If passed a single sequence of tokens it
    will use windows to determine co-occurrence. If passed a list of sequences of
    tokens it will use windows within each sequence in the list -- with windows not
    extending beyond the boundaries imposed by the individual sequences in the list."""

    def __init__(
        self,
        token_dictionary=None,
        min_occurrences=None,
        max_occurrences=None,
        min_frequency=None,
        max_frequency=None,
        ignored_tokens=None,
        window_function=fixed_window,
        kernel_function=flat_kernel,
        window_args=(5,),
        kernel_args=(),
        symmetrize=False,
    ):
        self.token_dictionary = token_dictionary
        self.min_occurrences = min_occurrences
        self.min_frequency = min_frequency
        self.max_occurrences = max_occurrences
        self.max_frequency = max_frequency
        self.ignored_tokens = ignored_tokens

        self.window_function = window_function
        self.kernel_function = kernel_function
        self.window_args = window_args
        self.kernel_args = kernel_args

        self.symmetrize = symmetrize

    def fit_transform(self, X, y=None, **fit_params):

        (
            token_sequences,
            self.token_dictionary_,
            self.token_frequencies_,
        ) = preprocess_token_sequences(
            X,
            self.token_dictionary,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            ignored_tokens=self.ignored_tokens,
        )

        if self.window_function is information_window:
            window_args = (*self.window_args, self.token_frequencies_)
        else:
            window_args = self.window_args

        (
            self.cooccurrences_,
            self.token_dictionary_,
            self.index_dictionary_,
        ) = token_cooccurence_matrix(
            X,
            window_function=self.window_function,
            kernel_function=self.kernel_function,
            window_args=window_args,
            kernel_args=self.kernel_args,
            symmetrize=self.symmetrize,
        )

        self.metric_ = distances.sparse_hellinger

        return self.cooccurrences_

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y)
        return self


class DistributionVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_components=20, random_state=None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def _validate_params(self):
        if (
            not np.issubdtype(type(self.n_components), np.integer)
            or self.n_components < 2
        ):
            raise ValueError(
                "n_components must be and integer greater than or equal " "to 2."
            )

    def fit(self, X, y=None, **fit_params):
        random_state = check_random_state(self.random_state)
        self._validate_params()

        combined_data = np.vstack(X)
        combined_data = check_array(combined_data)

        self.mixture_model_ = GaussianMixture(
            n_components=self.n_components, random_state=random_state
        )
        self.mixture_model_.fit(combined_data)
        self.ground_distance_ = pairwise_gaussian_ground_distance(
            self.mixture_model_.means_, self.mixture_model_.covariances_,
        )
        self.metric_ = distances.hellinger

    def transform(self, X):
        check_is_fitted(self, ["mixture_model_", "ground_distance_"])
        result = np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )


class HistogramVectorizer(BaseEstimator, TransformerMixin):
    """Convert a time series of binary events into a histogram of
    event occurrences over a time frame. If the data has explicit time stamps
    it can be aggregated over hour of day, day of week, day of month, day of year
    , week of year or month of year."""

    pass


class SkipgramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        token_dictionary=None,
        min_occurrences=None,
        max_occurrences=None,
        min_frequency=None,
        max_frequency=None,
        ignored_tokens=None,
        window_function=fixed_window,
        kernel_function=flat_kernel,
        window_args=(5,),
        kernel_args=(),
    ):
        self.token_dictionary = token_dictionary
        self.min_occurrences = min_occurrences
        self.min_frequency = min_frequency
        self.max_occurrences = max_occurrences
        self.max_frequency = max_frequency
        self.ignored_tokens = ignored_tokens

        self.window_function = window_function
        self.kernel_function = kernel_function
        self.window_args = window_args
        self.kernel_args = kernel_args

    def fit(self, X, y=None, **fit_params):
        (
            token_sequences,
            self.token_dictionary_,
            self.token_frequencies_,
        ) = preprocess_token_sequences(
            X,
            self.token_dictionary,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            ignored_tokens=self.ignored_tokens,
        )

        n_tokens = len(self.token_dictionary_)
        index_dictionary = {
            index: token for token, index in self.token_dictionary_.items()
        }

        row, col, data = skip_grams_matrix_coo_data(
            X,
            self.window_function,
            self.kernel_function,
            self.window_args,
            self.kernel_args,
            n_tokens,
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        column_sums = np.array(base_matrix.sum(axis=0))[0]
        self._column_is_kept = column_sums > 0
        self._kept_columns = np.where(self._column_is_kept)[0]

        self.skipgram_dictionary_ = {}
        for i in range(self._kept_columns.shape[0]):
            raw_val = self._kept_columns[i]
            first_token = index_dictionary[raw_val // n_tokens]
            second_token = index_dictionary[raw_val % n_tokens]
            self.skipgram_dictionary_[(first_token, second_token)] = i

        self._train_matrix = base_matrix[:, self._column_is_kept]
        self.metric_ = distances.sparse_hellinger
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(self, X, y, **fit_params)
        return self._train_matrix

    def transform(self, X):
        (token_sequences, _, _,) = preprocess_token_sequences(
            X, self.token_dictionary_, ignored_tokens=self.ignored_tokens,
        )

        n_tokens = len(self.token_dictionary_)

        row, col, data = skip_grams_matrix_coo_data(
            X,
            self.window_function,
            self.kernel_function,
            self.window_args,
            self.kernel_args,
            n_tokens,
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        result = base_matrix[:, self._column_is_kept]

        return result


class NgramVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, ngram_size=2, ngram_behaviour="exact", ngram_dictionary=None,
    ):
        self.ngram_size = ngram_size
        self.ngram_behaviour = ngram_behaviour
        self.ngram_dictionary = ngram_dictionary

    def fit(self, X, y=None, **fit_params):
        if self.ngram_dictionary is not None:
            self.ngram_dictionary_ = self.ngram_dictionary
        else:
            self.ngram_dictionary_ = defaultdict
            self.ngram_dictionary_.default_factory = self.ngram_dictionary_.__len__

        indptr = [0]
        indices = []
        data = []
        for sequence in X:
            counter = {}
            for gram in ngrams_of(sequence, self.ngram_size, self.ngram_behaviour):
                try:
                    col_index = self.ngram_dictionary_[gram]
                    if col_index in counter:
                        counter[col_index] += 1
                    else:
                        counter[col_index] = 1
                except KeyError:
                    # Out of predefined ngrams; drop
                    continue

            indptr.append(indptr[-1] + len(counter))
            indices.extend(counter.keys())
            data.extend(counter.values())

        # Remove defaultdict behavior
        self.ngram_dictionary_ = dict(self.ngram_dictionary_)

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32
        indices = np.asarray(indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        data = np.frombuffer(data, dtype=np.intc)

        self._train_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.ngram_dictionary_)),
            dtype=self.dtype,
        )
        self._train_matrix.sort_indices()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._train_matrix

    def transform(self, X):
        indptr = [0]
        indices = []
        data = []
        for sequence in X:
            counter = {}
            for gram in ngrams_of(sequence, self.ngram_size, self.ngram_behaviour):
                try:
                    col_index = self.ngram_dictionary_[gram]
                    if col_index in counter:
                        counter[col_index] += 1
                    else:
                        counter[col_index] = 1
                except KeyError:
                    # Out of predefined ngrams; drop
                    continue

            indptr.append(indptr[-1] + len(counter))
            indices.extend(counter.keys())
            data.extend(counter.values())

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32
        indices = np.asarray(indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        data = np.frombuffer(data, dtype=np.intc)

        result = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.ngram_dictionary_)),
            dtype=self.dtype,
        )
        result.sort_indices()

        return result


class KDEVectorizer(BaseEstimator, TransformerMixin):
    pass


class ProductDistributionVectorizer(BaseEstimator, TransformerMixin):
    pass


class Wasserstein1DHistogramTransformer(BaseEstimator, TransformerMixin):
    pass


class SequentialDifferenceTransformer(BaseEstimator, TransformerMixin):
    pass
