"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections import defaultdict
import scipy.sparse


def construct_token_dictionary_and_frequency(
    token_sequence, token_dictionary=None
):
    n_tokens = len(token_sequence)
    if token_dictionary is None:
        unique_tokens = sorted(list(set(token_sequence)))
        token_dictionary = dict(zip(unique_tokens, range(len(unique_tokens))))

    index_list = [token_dictionary[token] for token in token_sequence if token in token_dictionary]
    token_counts = np.bincount(index_list).astype(np.float32)

    token_frequency = token_counts / n_tokens

    return token_dictionary, token_frequency, n_tokens


@numba.njit(nogil=True)
def information_window(token_sequence, token_frequency, desired_entropy):

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


def word_word_cooccurence_matrix(
    token_sequences,
    window_function=fixed_window,
    kernel_function=flat_kernel,
    window_args=(5,),
    kernel_args=(),
    token_dictionary=None,
    min_df=5,
    max_df=1.0,
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


class TokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a
    co-occurrence count matrix of tokens. If passed a single sequence of tokens it
    will use windows to determine co-occurence. If passed a list of sequences of
    tokens it will use windows within each sequence in the list -- with windows not
    extending beyond the boundaries imposed by the individual sequences in the list."""

    pass


class DistributionVectorizer(BaseEstimator, TransformerMixin):


    pass


class HistogramVectorizer(BaseEstimator, TransformerMixin):
    """Convert a time series of binary events into a histogram of
    event occurrences over a time frame. If the data has explicit time stamps
    it can be aggregated over hour of day, day of week, day of month, day of year
    , week of year or month of year."""

    pass


class SkipgramVectorizer(BaseEstimator, TransformerMixin):

    pass


class NgramVectorizer(BaseEstimator, TransformerMixin):

    pass


class KDEVectorizer(BaseEstimator, TransformerMixin):

    pass


class ProductDistributionVectorizer(BaseEstimator, TransformerMixin):

    pass


class Wasserstein1DHistogramTransformer(BaseEstimator, TransformerMixin):

    pass


class SequentialDifferenceTransformer(BaseEstimator, TransformerMixin):

    pass
