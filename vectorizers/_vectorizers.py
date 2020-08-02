"""
This is a module to be used as a reference for building other modules
"""
from warnings import warn

import numpy as np
import numba
from numba.typed import List

from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import pandas as pd
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize

from collections import defaultdict
import scipy.linalg
import scipy.stats
import scipy.sparse
from typing import Union, Sequence, AnyStr

import re

from .utils import (
    flatten,
    vectorize_diagram,
    pairwise_gaussian_ground_distance,
    validate_homogeneous_token_types,
    sparse_collapse,
)
import vectorizers.distances as distances

from ._window_kernels import _KERNEL_FUNCTIONS, _WINDOW_FUNCTIONS

def construct_document_frequency(token_by_doc_sequence, token_dictionary):
    """Returns the frequency of documents that each token appears in.

    Parameters
    ----------
    token_by_doc_sequence: Iterable
        A sequence of sequences of tokens

    token_dictionary: dictionary
        A fixed dictionary providing the mapping of tokens to indices

    Returns
    -------
    document_frequency: np.array
        The document frequency of tokens ordered by token_dictionary
    """

    n_tokens = len(token_dictionary)
    doc_freq = np.zeros(n_tokens)
    for doc in token_by_doc_sequence:
        doc_freq += np.bincount(
            [token_dictionary[token] for token in set(doc)], minlength=n_tokens
        )

    return doc_freq / len(token_by_doc_sequence)


def construct_token_dictionary_and_frequency(token_sequence, token_dictionary=None):
    """Construct a dictionary mapping tokens to indices and a table of token
    frequencies (where the frequency of token 'x' is given by token_frequencies[
    token_dictionary['x']]).

    Parameters
    ----------
    token_sequence: Iterable
        A single long sequence of tokens

    token_dictionary: dictionary or None (optional, default=None)
        Optionally a fixed dictionary providing the mapping of tokens to indices

    Returns
    -------
    token_dictionary: dictionary
        The dictionary mapping tokens to indices

    token_frequency: array of shape (len(token_dictionary),)
        The frequency of occurrence of tokens (with index from the token dictionary)

    n_tokens: int
        The total number of tokens in the sequence
    """
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


def select_tokens_by_regex(tokens, regex):
    if not isinstance(regex, re.Pattern):
        regex = re.compile(regex)

    result = set([])
    for token in tokens:
        if regex.fullmatch(token) is not None:
            result.add(token)

    return result


def prune_token_dictionary(
    token_dictionary,
    token_frequencies,
    token_doc_frequencies=np.array([]),
    ignored_tokens=None,
    excluded_token_regex=None,
    min_frequency=0.0,
    max_frequency=1.0,
    min_occurrences=None,
    max_occurrences=None,
    min_document_frequency=0.0,
    max_document_frequency=1.0,
    min_document_occurrences=None,
    max_document_occurrences=None,
    total_tokens=None,
    total_documents=None,
):
    """Prune the token dictionary based on constraints of tokens to ignore and
    min and max allowable token frequencies. This will remove any tokens that should
    be ignored and any tokens that occur less often than the minimum frequency or
    more often than the maximum frequency.

    Parameters
    ----------
    token_dictionary: dictionary
        The token dictionary mapping tokens to indices for pruning

    token_frequencies: array of shape (len(token_dictionary),)
        The frequency of occurrence of the tokens in the dictionary

    token_doc_frequencies: array of shape (len(token_dictionary),)
        The frequency of documents with occurrences of the tokens in the dictionary

    ignored_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored, and thus removed from the
        dictionary. This could be, for example, top words in an NLP context.

    min_frequency: float (optional, default=0.0)
        The minimum frequency of occurrence allowed for tokens. Tokens that occur
        less frequently than this will be pruned.

    max_frequency float (optional, default=1.0)
        The maximum frequency of occurrence allowed for tokens. Tokens that occur
        more frequently than this will be pruned.

    min_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.
    min_document_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of documents with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_document_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of documents with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    min_document_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of documents with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    max_document_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of documents with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    total_tokens: int or None (optional, default=None)
        Must be set if you pass in min_occurrence and max_occurrence.

    total_documents: int or None (optional, default=None)
        Must be set if you pass in min_document_occurrence and max_document_occurrence.

    Returns
    -------
    new_token_dictionary: dictionary
        The pruned dictionary of token to index mapping

    new_token_frequencies: array of shape (len(new_token_dictionary),)
        The token frequencies remapped to the new token indexing given by
        new_token_dictionary.
    """

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

    ## Prune by document frequency

    if min_document_occurrences is None:
        if min_document_frequency is None:
            min_document_frequency = 0.0
    else:
        if min_document_frequency is not None:
            assert min_document_occurrences / total_documents == min_document_frequency
        else:
            min_document_frequency = min_document_occurrences / total_documents

    if max_document_occurrences is None:
        if max_document_frequency is None:
            max_document_frequency = 1.0
    else:
        if max_document_frequency is not None:
            assert max_document_occurrences / total_documents == max_document_frequency
        else:
            max_document_frequency = min(
                1.0, max_document_occurrences / total_documents
            )

    if ignored_tokens is not None:
        tokens_to_prune = set(ignored_tokens)
    else:
        tokens_to_prune = set([])

    reverse_token_dictionary = {index: word for word, index in token_dictionary.items()}

    infrequent_tokens = np.where(token_frequencies < min_frequency)[0]
    frequent_tokens = np.where(token_frequencies > max_frequency)[0]

    infrequent_doc_tokens = np.where(token_doc_frequencies < min_document_frequency)[0]
    frequent_doc_tokens = np.where(token_doc_frequencies > max_document_frequency)[0]

    tokens_to_prune.update({reverse_token_dictionary[i] for i in infrequent_tokens})
    tokens_to_prune.update({reverse_token_dictionary[i] for i in frequent_tokens})
    tokens_to_prune.update({reverse_token_dictionary[i] for i in infrequent_doc_tokens})
    tokens_to_prune.update({reverse_token_dictionary[i] for i in frequent_doc_tokens})

    if excluded_token_regex is not None:
        tokens_to_prune.update(
            select_tokens_by_regex(token_dictionary.keys(), excluded_token_regex)
        )

    vocab_tokens = [token for token in token_dictionary if token not in tokens_to_prune]
    new_vocabulary = dict(zip(vocab_tokens, range(len(vocab_tokens))))
    new_token_frequency = np.array(
        [token_frequencies[token_dictionary[token]] for token in new_vocabulary]
    )

    return new_vocabulary, new_token_frequency


def preprocess_tree_sequences(
    tree_sequences,
    flat_sequence,
    token_dictionary=None,
    min_occurrences=None,
    max_occurrences=None,
    min_frequency=None,
    max_frequency=None,
    min_tree_occurrences=None,
    max_tree_occurrences=None,
    min_tree_frequency=None,
    max_tree_frequency=None,
    ignored_tokens=None,
    excluded_token_regex=None,
    masking=None,
):
    """Perform a standard set of preprocessing for token sequences. This includes
    constructing a token dictionary and token frequencies, pruning the dictionary
    according to frequency and ignored token constraints, and editing the token
    sequences to only include tokens in the pruned dictionary. Note that either
    min_occurrences or min_frequency can be provided (respectively
    max_occurences or max_frequency). If both are provided they must agree.

    Parameters
    ----------
    tree_sequences: sequence of tuples (sparse matrix of size (n,n), array of size (n))
        Each tuple in this sequence represents a labelled tree.
        The first element is a sparse adjacency matrix
        The second element is an array of node labels

    flat_sequence: tuple
        A tuple tokens for processing.

    token_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, constraining the tokens
        that are allowed. If None then the allowed tokens and a mapping will
        be learned from the data and returned.

    min_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    min_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of occurrence for a token to be
        considered valid. If None then no constraint will be applied.

    max_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of occurrence for a token to be
        considered valid. If None then no constraint will be applied.

    min_tree_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of trees with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_tree_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of trees with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    min_tree_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of trees with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    max_tree_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of trees with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    ignored_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored. If None then no tokens will
        be ignored.

    masking: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.

    Returns
    -------
    result_sequences: list of np.ndarray
        The sequences, pruned of tokens not meeting constraints.

    token_dictionary: dictionary
        The token dictionary mapping tokens to indices.

    token_frequencies: array of shape (len(token_dictionary),)
        The frequency of occurrence of the tokens in the token_dictionary.
    """
    (
        token_dictionary_,
        token_frequencies,
        total_tokens,
    ) = construct_token_dictionary_and_frequency(flat_sequence, token_dictionary)

    if token_dictionary is None:
        if {
            min_tree_frequency,
            min_tree_occurrences,
            max_tree_frequency,
            max_tree_occurrences,
        } != {None}:
            token_doc_frequencies = construct_document_frequency(
                [tree[1] for tree in tree_sequences], token_dictionary_
            )
        else:
            token_doc_frequencies = np.array([])

        token_dictionary, token_frequencies = prune_token_dictionary(
            token_dictionary_,
            token_frequencies,
            token_doc_frequencies=token_doc_frequencies,
            ignored_tokens=ignored_tokens,
            excluded_token_regex=excluded_token_regex,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
            min_document_frequency=min_tree_frequency,
            max_document_frequency=max_tree_frequency,
            min_document_occurrences=min_tree_occurrences,
            max_document_occurrences=max_tree_occurrences,
            total_tokens=total_tokens,
            total_documents=len(tree_sequences),
        )

    # We will prune the edges from any nodes who's labels are to be filtered and reconnect their parents with their children.
    # This will remove them from our computation without having to alter the matrix size or label_sequence.
    if masking is None:
        result_sequence = []
        for adj_matrix, label_sequence in tree_sequences:
            node_index_to_remove = [
                i for i, x in enumerate(label_sequence) if x not in token_dictionary
            ]
            result_matrix = adj_matrix.tolil().copy()
            for node_index in node_index_to_remove:
                remove_node(result_matrix, node_index)

            #  If we want to eliminate the zero row/columns and trim the label_sequence:
            #
            #  label_in_dictionary = np.array([x in token_dictionary for x in label_sequence])
            #  result_matrix = result_matrix.tocsr()[label_in_dictionary, :]
            #  result_matrix = result_matrix.T[label_in_dictionary, :].T.tocoo()
            #  result_labels = label_sequence[label_in_dictionary]
            #  result_sequence.append((result_matrix, result_labels))

            result_sequence.append((result_matrix, label_sequence))
    else:
        result_sequence = []
        if masking in token_dictionary:
            del token_dictionary[masking]

        for adj_matrix, label_sequence in tree_sequences:
            new_labels = [
                label if label in token_dictionary else masking
                for label in label_sequence
            ]
            result_sequence.append((adj_matrix, new_labels))
        token_dictionary[masking] = len(token_dictionary)

    inverse_token_dictionary = {
        index: token for token, index in token_dictionary.items()
    }

    return (
        result_sequence,
        token_dictionary,
        inverse_token_dictionary,
        token_frequencies,
    )


def preprocess_token_sequences(
    token_sequences,
    flat_sequence,
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
    masking=None,
):
    """Perform a standard set of preprocessing for token sequences. This includes
    constructing a token dictionary and token frequencies, pruning the dictionary
    according to frequency and ignored token constraints, and editing the token
    sequences to only include tokens in the pruned dictionary. Note that either
    min_occurrences or min_frequency can be provided (respectively
    max_occurences or max_frequency). If both are provided they must agree.

    Parameters
    ----------
    token_sequences: Iterable of (tuple | list | np.ndarray)
        A list of token sequences. Each sequence should be tuple, list or
        numpy array of tokens.

    flat_sequence: tuple
        A tuple tokens for processing.

    token_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, constraining the tokens
        that are allowed. If None then the allowed tokens and a mapping will
        be learned from the data and returned.

    min_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    min_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of occurrence for a token to be
        considered valid. If None then no constraint will be applied.

    max_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of occurrence for a token to be
        considered valid. If None then no constraint will be applied.

    min_document_occurrences: int or None (optional, default=None)
        A constraint on the minimum number of documents with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    max_document_occurrences: int or None (optional, default=None)
        A constraint on the maximum number of documents with occurrences for a token to be considered
        valid. If None then no constraint will be applied.

    min_document_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of documents with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    max_document_frequency: float or None (optional, default=None)
        A constraint on the minimum frequency of documents with occurrences for a token to be
        considered valid. If None then no constraint will be applied.

    ignored_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored. If None then no tokens will
        be ignored.

    masking: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.

    Returns
    -------
    result_sequences: list of np.ndarray
        The sequences, pruned of tokens not meeting constraints.

    token_dictionary: dictionary
        The token dictionary mapping tokens to indices.

    token_frequencies: array of shape (len(token_dictionary),)
        The frequency of occurrence of the tokens in the token_dictionary.
    """

    # Get vocabulary and word frequencies

    (
        token_dictionary_,
        token_frequencies,
        total_tokens,
    ) = construct_token_dictionary_and_frequency(flat_sequence, token_dictionary)

    if token_dictionary is None:
        if {
            min_document_frequency,
            min_document_occurrences,
            max_document_frequency,
            max_document_occurrences,
        } != {None}:
            token_doc_frequencies = construct_document_frequency(
                token_sequences, token_dictionary_
            )
        else:
            token_doc_frequencies = np.array([])

        token_dictionary, token_frequencies = prune_token_dictionary(
            token_dictionary_,
            token_frequencies,
            token_doc_frequencies=token_doc_frequencies,
            ignored_tokens=ignored_tokens,
            excluded_token_regex=excluded_token_regex,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
            min_document_frequency=min_document_frequency,
            max_document_frequency=max_document_frequency,
            min_document_occurrences=min_document_occurrences,
            max_document_occurrences=max_document_occurrences,
            total_tokens=total_tokens,
            total_documents=len(token_sequences),
        )

    if masking is None:
        result_sequences = List()
        for sequence in token_sequences:
            result_sequences.append(
                np.array(
                    [
                        token_dictionary[token]
                        for token in sequence
                        if token in token_dictionary
                    ],
                    dtype=np.int64,
                )
            )
    else:
        result_sequences = List()
        if masking in token_dictionary:
            del token_dictionary[masking]

        for sequence in token_sequences:
            result_sequences.append(
                np.array(
                    [
                        len(token_dictionary)
                        if not (token in token_dictionary)
                        else token_dictionary[token]
                        for token in sequence
                    ],
                    dtype=np.int64,
                )
            )
        token_dictionary[masking] = len(token_dictionary)

    inverse_token_dictionary = {
        index: token for token, index in token_dictionary.items()
    }

    return (
        result_sequences,
        token_dictionary,
        inverse_token_dictionary,
        token_frequencies,
    )


def remove_node(adjacency_matrix, node, inplace=True):
    if not inplace:
        if scipy.sparse.isspmatrix_lil(adjacency_matrix):
            adj = adjacency_matrix.copy()
        else:
            adj = adjacency_matrix.tolil()
    elif not scipy.sparse.isspmatrix_lil(adjacency_matrix):
        raise ValueError("Can only remove node in place from LIL matrices")
    else:
        adj = adjacency_matrix
    # Copy the row we want to kill
    row_to_remove = adj.rows[node].copy()
    data_to_remove = adj.data[node].copy()
    # Ensure we ignore any self-loops in the row
    try:
        index_to_remove = row_to_remove.index(node)
        row_to_remove = (
            row_to_remove[:index_to_remove] + row_to_remove[index_to_remove + 1 :]
        )
        data_to_remove = (
            data_to_remove[:index_to_remove] + data_to_remove[index_to_remove + 1 :]
        )
    except ValueError:
        pass
    # Process all the rows making changes as required
    for i in range(adj.rows.shape[0]):
        if i == node:
            adj.rows[i] = []
            adj.data[i] = []
        else:
            try:
                # Find out if this node has selected node as a successor
                index_to_modify = adj.rows[i].index(node)
                # If so replace the entry for that node with successor entries
                # from the selected node
                adj.rows[i][index_to_modify : index_to_modify + 1] = row_to_remove
                adj.data[i][index_to_modify : index_to_modify + 1] = data_to_remove
            except ValueError:
                # We didn't have the selected node in the data; nothing to do
                pass

    if not inplace:
        # Clean up the result
        result = adj.tocsr()
        result.eliminate_zeros()
        result.sort_indices()
        return result
    else:
        return adj


@numba.njit(nogil=True)
def build_skip_grams(
    token_sequence, window_function, kernel_function, window_args, kernel_args
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

    window_function: numba.jitted callable
        A function producing a sequence of windows given a source sequence

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    window_args: tuple
        Arguments to pass through to the window function

    kernel_args: tuple
        Arguments to pass through to the kernel function

    Returns
    -------
    skip_grams: array of shape (n_skip_grams, 3)
        Each skip_gram is a vector giving the head (or skip) token index, the
        tail token index, and the associated weight of the skip-gram.
    """
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
            new_tokens[new_token_count, 0] = np.float32(head_token)
            new_tokens[new_token_count, 1] = np.float32(window[j])
            new_tokens[new_token_count, 2] = weights[j]
            new_token_count += 1

    return new_tokens


def build_tree_skip_grams(
    token_sequence, adjacency_matrix, kernel_function, window_size
):
    """
    Takes and adjacency matrix counts the co-occurrence of each token within a window_size
    number of hops from each vertex.  These counts are weighted by the kernel function.
    Parameters
    ----------
    token_sequence: array
        This should be a sequence of tokens that represent the labels of the adjacency matrix's vertices
        len(token_sequence) == adjacency_matrix.shape[0]
    adjacency_matrix: matrix of ndarray
        This should be an adjacency matrix of a graph.
    kernel_function: a function that takes a window and a window_size parameter
        and returns a vector of weights.
    window_size:
        The size of the window to apply the kernel over.

    Returns
    -------
    matrix: sparse matrix of shape (unique_labels, unique_labels)
        This is the matrix of the summation of values between unique labels
    labels: array of length (unique_labels)
        This is the array of the labels of the rows and columns of our matrix.
    """
    weights = kernel_function(np.arange(window_size), window_size)
    count_matrix = adjacency_matrix * weights[0]
    walk = adjacency_matrix
    for i in range(1, window_size):
        walk = walk @ adjacency_matrix
        count_matrix += walk * weights[i]

    # Now collapse the rows and columns with the same label
    (grouped_matrix, new_labels) = sparse_collapse(count_matrix, token_sequence)

    return grouped_matrix, new_labels


def skip_grams_matrix_coo_data(
    list_of_token_sequences,
    window_function,
    kernel_function,
    window_args,
    kernel_args,
    token_dictionary,
):
    """Given a list of token sequences construct the relevant data for a sparse
    matrix representation with a row for each token sequence and a column for each
    skip-gram.

    Parameters
    ----------
    list_of_token_sequences: Iterable of Iterables
        The token sequences to construct skip-gram based representations of.

    window_function: numba.jitted callable
        A function producing a sequence of windows given a source sequence

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    window_args: tuple
        Arguments to pass through to the window function

    kernel_args: tuple
        Arguments to pass through to the kernel function

    n_unique_tokens: int
        The total number of unique tokens across all the sequences

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

    n_unique_tokens = len(token_dictionary)

    for row_idx in range(len(list_of_token_sequences)):
        skip_gram_data = build_skip_grams(
            list_of_token_sequences[row_idx],
            window_function,
            kernel_function,
            window_args,
            kernel_args,
        )
        for i in range(skip_gram_data.shape[0]):
            skip_gram = skip_gram_data[i]
            result_row.append(row_idx)
            result_col.append(
                np.int32(skip_gram[0]) * n_unique_tokens + np.int32(skip_gram[1])
            )
            result_data.append(skip_gram[2])

    return np.asarray(result_row), np.asarray(result_col), np.asarray(result_data)


@numba.njit(nogil=True)
def sequence_skip_grams(
    token_sequences, window_function, kernel_function, window_args, kernel_args
):
    """Produce skip-gram data for a combined over a list of token sequences. In this
    case each token sequence represents a sequence with boundaries over which
    skip-grams may not extend (such as sentence boundaries in an NLP context).

    Parameters
    ----------
    token_sequences: Iterable of Iterables
        The token sequences to produce skip-gram data for

    window_function: numba.jitted callable
        A function producing a sequence of windows given a source sequence

    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens

    window_args: tuple
        Arguments to pass through to the window function

    kernel_args: tuple
        Arguments to pass through to the kernel function

    Returns
    -------
    skip_grams: array of shape (n_skip_grams, 3)
        The skip grams for the combined set of sequences.
    """
    skip_grams_per_sequence = [
        build_skip_grams(
            token_sequence, window_function, kernel_function, window_args, kernel_args
        )
        for token_sequence in token_sequences
    ]
    total_n_skip_grams = 0
    for arr in skip_grams_per_sequence:
        total_n_skip_grams += arr.shape[0]
    result = np.empty((total_n_skip_grams, 3), dtype=np.float32)
    count = 0
    for arr in skip_grams_per_sequence:
        result[count : count + arr.shape[0]] = arr
        count += arr.shape[0]
    return result


def sequence_tree_skip_grams(
    tree_sequences, kernel_function, window_size, label_dictionary, window_orientation,
):
    """
    Takes a sequence of labelled trees and counts the weighted skip grams of their labels.
    For our purposes we take a labelled tree to be a tuple containing an adjacency matrix and an
    array of it's vertex labels.
    Parameters
    ----------
    tree_sequences: sequence of tuples (sparse matrix of size (n,n), array of size (n))
        Each tuple in this sequence represents a labelled tree.
        The first element is a sparse adjacency matrix
        The second element is an array of node labels
    kernel_function: numba.jitted callable
        A function producing weights given a window of tokens
        Currently this needs to take a window_size parameter.
    window_size: int
        The number of steps out to look in your tree skip gram
    label_dictionary: dict
        A dictionary mapping from your valid label set to indices.  This is used as the global
        alignment for your new label space.  All tokens still present in your tree sequences must
        exist within your label_dictionary.
    window_orientation: string (['before', 'after', 'symmetric', 'directional'])
        The orientation of the cooccurrence window.  Whether to return all the tokens that
        occurred within a window before, after on either side.
        symmetric: counts tokens occurring before and after as the same tokens
        directional: counts tokens before and after as different and returns both counts.

    Returns
    -------
    skip_gram_matrix: sparse.csr_matrix
        A matrix of weighted graph label co-occurence counts.
    """
    n_tokens = len(label_dictionary)
    global_counts = scipy.sparse.coo_matrix((n_tokens, n_tokens))
    for adj_matrix, token_sequence in tree_sequences:
        (count_matrix, unique_labels,) = build_tree_skip_grams(
            token_sequence=token_sequence,
            adjacency_matrix=adj_matrix,
            kernel_function=kernel_function,
            window_size=window_size,
        )
        # Reorder these based on the label_dictionary
        count_matrix = count_matrix.tocoo()
        rows = [label_dictionary[unique_labels[x]] for x in count_matrix.row]
        cols = [label_dictionary[unique_labels[x]] for x in count_matrix.col]
        data = count_matrix.data
        reordered_matrix = scipy.sparse.coo_matrix(
            (data, (rows, cols)), shape=(n_tokens, n_tokens)
        )
        global_counts += reordered_matrix
    global_counts = global_counts.tocsr()

    if window_orientation == "before":
        global_counts = global_counts.T
    elif window_orientation == "symmetric":
        global_counts += global_counts.T
    elif window_orientation == "after":
        pass
    elif window_orientation == "directional":
        global_counts = scipy.sparse.hstack([global_counts.T, global_counts])
    else:
        raise ValueError(
            f"Sorry your window_orientation must be one of ['before', 'after', 'symmetric', 'directional']. "
            f"You passed us {window_orientation}"
        )

    return global_counts


def token_cooccurrence_matrix(
    token_sequences,
    n_unique_tokens,
    window_function,
    kernel_function,
    window_args,
    kernel_args,
    window_orientation="symmetric",
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

    window_function: numba.jitted callable (optional, default=fixed_window)
        A function producing a sequence of windows given a source sequence

    kernel_function: numba.jitted callable (optional, default=flat_kernel)
        A function producing weights given a window of tokens

    window_args: tuple (optional, default=(5,)
        Arguments to pass through to the window function

    kernel_args: tuple (optional, default=())
        Arguments to pass through to the kernel function

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

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

    cooccurrence_matrix = scipy.sparse.coo_matrix(
        (n_unique_tokens, n_unique_tokens), dtype=np.float32
    )
    n_chunks = (len(token_sequences) // chunk_size) + 1

    for chunk_index in range(n_chunks):
        chunk_start = chunk_index * chunk_size
        chunk_end = min(len(token_sequences), chunk_start + chunk_size)

        raw_coo_data = sequence_skip_grams(
            token_sequences[chunk_start:chunk_end],
            window_function,
            kernel_function,
            window_args,
            kernel_args,
        )
        cooccurrence_matrix += scipy.sparse.coo_matrix(
            (
                raw_coo_data.T[2],
                (
                    raw_coo_data.T[0].astype(np.int64),
                    raw_coo_data.T[1].astype(np.int64),
                ),
            ),
            shape=(n_unique_tokens, n_unique_tokens),
            dtype=np.float32,
        )
        cooccurrence_matrix.sum_duplicates()

    if window_orientation == "before":
        cooccurrence_matrix = cooccurrence_matrix.transpose()
    elif window_orientation == "after":
        cooccurrence_matrix = cooccurrence_matrix
    elif window_orientation == "symmetric":
        cooccurrence_matrix = cooccurrence_matrix + cooccurrence_matrix.transpose()
    elif window_orientation == "directional":
        cooccurrence_matrix = scipy.sparse.hstack(
            [cooccurrence_matrix.transpose(), cooccurrence_matrix]
        )
    else:
        raise ValueError(
            f'window_orientation must be one of the strings ["before", "after", "symmetric","directional"]'
        )

    return cooccurrence_matrix.tocsr()


@numba.njit(nogil=True)
def ngrams_of(sequence, ngram_size, ngram_behaviour="exact"):
    """Produce n-grams of a sequence of tokens. The n-gram behaviour can either
    be "exact", meaning that only n-grams of exactly size n are produced,
    or "subgrams" meaning that all n-grams of size less than or equal to n are
    produced.

    Parameters
    ----------
    sequence: Iterable
        The sequence of tokens to produce n-grams of.

    ngram_size: int
        The size of n-grams to use.

    ngram_behaviour: string (optional, default="exact")
        The n-gram behaviour. Should be one of:
            * "exact"
            * "subgrams"

    Returns
    -------
    ngrams: list
        A list of the n-grams of the sequence.
    """
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


@numba.njit(nogil=True)
def min_non_zero_difference(data):
    """Find the minimum non-zero sequential difference in a single dimensional
    array of values. This is useful for determining the minimal reasonable kernel
    bandwidth for a 1-dimensional KDE over a dataset.

    Parameters
    ----------
    data: array
        One dimensional array of values

    Returns
    -------
    min_difference: float
        The minimal difference between sequential values.
    """
    sorted_data = np.sort(data)
    differences = sorted_data[1:] - sorted_data[:-1]
    return np.min(differences[differences > 0])


def jackknife_bandwidths(data, bandwidths, kernel="gaussian"):
    """Perform jack-knife sampling over different bandwidths for KDEs for each
    time-series in the dataset.

    Parameters
    ----------
    data: list of arrays
        A list of (variable length) arrays of values. The values should represent
        "times" of "events".

    bandwidths: array
        The possible bandwidths to try

    kernel: string (optional, default="gaussian")
        The kernel to use for the KDE. Should be accepted by sklearn's KernelDensity
        class.

    Returns
    -------
    result: array of shape (n_bandwidths,)
        The total likelihood of unobserved data over all jackknife samplings and all
        time series in the dataset for each bandwidth.
    """
    result = np.zeros(bandwidths.shape[0])
    for j in range(bandwidths.shape[0]):
        kde = KernelDensity(bandwidth=bandwidths[j], kernel=kernel)
        for i in range(len(data)):
            likelihood = 0.0
            for k in range(len(data[i])):
                if k < len(data[i]) - 1:
                    jackknife_sample = np.hstack([data[i][:k], data[i][k + 1 :]])
                else:
                    jackknife_sample = data[i][:k]
                kde.fit(jackknife_sample[:, None])
                likelihood += np.exp(kde.score(np.array([[data[i][k]]])))

            result[j] += likelihood

    return result


class TokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a
    co-occurrence count matrix of tokens. If passed a single sequence of tokens it
    will use windows to determine co-occurrence. If passed a list of sequences of
    tokens it will use windows within each sequence in the list -- with windows not
    extending beyond the boundaries imposed by the individual sequences in the list.

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

    window_function: numba.jitted callable or str (optional, default='fixed')
        A function producing a sequence of windows given a source sequence and a window_radius and term frequencies.
        The string options are ['fixed', 'information'] for using pre-defined functions.

    kernel_function: numba.jitted callable or str (optional, default='flat')
        A function producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic'] for using pre-defined functions.

    window_radius: int (optional, default=5)
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

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
        window_radius=5,
        window_orientation="directional",
        chunk_size=1 << 20,
        validate_data=True,
        mask_string=None,
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
        self.window_radius = window_radius

        self.window_orientation = window_orientation
        self.chunk_size = chunk_size
        self.validate_data = validate_data
        self.mask_string = mask_string

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

            ## Adjust the window size for the info window
        if self.window_function == "information":
            entropy = np.dot(
                self._token_frequencies_, np.log2(self._token_frequencies_)
            )
            self._window_size = self.window_radius * entropy
        else:
            self._window_size = self.window_radius

        self.cooccurrences_ = token_cooccurrence_matrix(
            token_sequences,
            len(self.token_label_dictionary_),
            window_function=self._window_function,
            kernel_function=self._kernel_function,
            window_args=(self._window_size, self._token_frequencies_),
            kernel_args=(self.window_radius,),
            window_orientation=self.window_orientation,
            chunk_size=self.chunk_size,
        )
        self.cooccurrences_.eliminate_zeros()

        if self.window_orientation in ["before", "after", "symmetric"]:
            self.column_label_dictionary_ = self.token_label_dictionary_
            self.column_index_dictionary_ = self.token_index_dictionary_
        elif self.window_orientation == "directional":
            self.column_label_dictionary_ = {
                "pre_" + str(token): index
                for token, index in self.token_label_dictionary_.items()
            }
            self.column_label_dictionary_.update(
                {
                    "post_" + str(token): index + len(self.token_label_dictionary_)
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

        cooccurrences = token_cooccurrence_matrix(
            token_sequences,
            len(self.token_label_dictionary_),
            window_function=self._window_function,
            kernel_function=self._kernel_function,
            window_args=(self._window_size, self._token_frequencies_),
            kernel_args=(self.window_radius,),
            window_orientation=self.window_orientation,
        )
        cooccurrences.eliminate_zeros()

        return cooccurrences


class LabelledTreeCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a sequence of labelled trees and counts and produces a cooccurrence count matrix
    of their labels.
    For our purposes we take a labelled tree to be a tuple containing an adjacency matrix and an
    array of it's vertex labels.
    Parameters
    ----------
    tree_sequences: sequence of tuples (sparse matrix of size (n,n), array of size (n))
        Each tuple in this sequence represents a labelled tree.
        The first element is a sparse adjacency matrix
        The second element is an array of node labels

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

    min_tree_occurrences: int or None (optional, default=None)
        The minimal number of trees with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_tree_frequency.

    max_tree_occurrences int or None (optional, default=None)
        The maximal number of trees with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_tree_frequency.

    min_tree_frequency: float or None (optional, default=None)
        The minimal frequency of trees with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by min_tree_occurrences.

    max_tree_frequency: float or None (optional, default=None)
        The maximal frequency trees with an occurrences of a token for the token to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_tree_occurrences.

    ignored_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored entirely. If None then no tokens will
        be ignored in this fashion.

    excluded_regex: str or None (optional, default=None)
        The regular expression by which tokens are ignored if re.fullmatch returns True.

    kernel_function: numba.jitted callable or str (optional, default='flat')
        A function producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic'] for using pre-defined functions.

    window_radius: int (optional, default=5)
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

    window_orientation: string (['before', 'after', 'symmetric', 'directional'])
        The orientation of the cooccurrence window.  Whether to return all the tokens that
        occurred within a window before, after on either side.
        symmetric: counts tokens occurring before and after as the same tokens
        directional: counts tokens before and after as different and returns both counts.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, otherwise replaces them with the provided mask_string.
    """

    def __init__(
        self,
        min_occurrences=None,
        max_occurrences=None,
        min_frequency=None,
        max_frequency=None,
        min_tree_occurrences=None,
        max_tree_occurrences=None,
        min_tree_frequency=None,
        max_tree_frequency=None,
        ignored_tokens=None,
        excluded_token_regex=None,
        # window_function="fixed",
        kernel_function="flat",
        window_radius=5,
        token_dictionary=None,
        window_orientation="directional",
        validate_data=True,
        mask_string=None,
    ):
        self.token_dictionary = token_dictionary
        self.min_occurrences = min_occurrences
        self.min_frequency = min_frequency
        self.max_occurrences = max_occurrences
        self.max_frequency = max_frequency
        self.min_tree_occurrences = min_tree_occurrences
        self.min_tree_frequency = min_tree_frequency
        self.max_tree_occurrences = max_tree_occurrences
        self.max_tree_frequency = max_tree_frequency
        self.ignored_tokens = ignored_tokens
        self.excluded_token_regex = excluded_token_regex

        # self.window_function = window_function
        self.kernel_function = kernel_function
        self.window_radius = window_radius

        self.window_orientation = window_orientation
        self.validate_data = validate_data
        self.mask_string = mask_string

    def fit(self, X, y=None, **fit_params):
        """
        Takes a sequence of labelled trees and learns the weighted cooccurrence counts of the labels.
        Parameters
        ----------
        X: Iterable of tuples (nd.array | scipy.sparse.csr_matrix, label_sequence)
        An iterable of tuples the with
        first = adjacency matrices of the tree's that make up your corpus.
        second = a sequence of labels of length first.shape[0] representing the labels of each vertex.

        Returns
        -------
        self
        """
        # Filter and process the raw token counts and build the token dictionaries.
        # WARNING: We count isolated vertices as an occurrence of a label.  It's a feature.
        raw_token_sequences = [label_sequence for adjacency, label_sequence in X]
        flat_sequences = flatten(raw_token_sequences)
        (
            clean_tree_sequence,
            self.token_label_dictionary_,
            self.token_index_dictionary_,
            self._token_frequencies_,
        ) = preprocess_tree_sequences(
            X,
            flat_sequences,
            self.token_dictionary,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            min_tree_occurrences=self.min_tree_occurrences,
            max_tree_occurrences=self.max_tree_occurrences,
            min_tree_frequency=self.min_tree_frequency,
            max_tree_frequency=self.max_tree_frequency,
            ignored_tokens=self.ignored_tokens,
            excluded_token_regex=self.excluded_token_regex,
            masking=self.mask_string,
        )

        if callable(self.kernel_function):
            self._kernel_function = self.kernel_function
        elif self.kernel_function in _KERNEL_FUNCTIONS:
            self._kernel_function = _KERNEL_FUNCTIONS[self.kernel_function]
        else:
            raise ValueError(
                f"Unrecognized kernel_function; should be callable or one of {_KERNEL_FUNCTIONS.keys()}"
            )

        self._window_size = self.window_radius

        # build token_cooccurrence_matrix()
        self.cooccurrences_ = sequence_tree_skip_grams(
            clean_tree_sequence,
            kernel_function=self._kernel_function,
            window_size=self._window_size,
            label_dictionary=self.token_label_dictionary_,
            window_orientation=self.window_orientation,
        )

        if self.window_orientation in ["before", "after", "symmetric"]:
            self.column_label_dictionary_ = self.token_label_dictionary_
            self.column_index_dictionary_ = self.token_index_dictionary_
        elif self.window_orientation == "directional":
            self.column_label_dictionary_ = {
                "pre_" + token: index
                for token, index in self.token_label_dictionary_.items()
            }
            self.column_label_dictionary_.update(
                {
                    "post_" + token: index + len(self.token_label_dictionary_)
                    for token, index in self.token_label_dictionary_.items()
                }
            )
            self.column_index_dictionary_ = {
                item[1]: item[0] for item in self.column_label_dictionary_.items()
            }
        self.metric_ = distances.sparse_hellinger

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Takes a sequence of labelled trees and learns the weighted cooccurrence counts of the labels.
        Parameters
        ----------
        X: Iterable of tuples (nd.array | scipy.sparse.csr_matrix, label_sequence)
        An iterable of tuples the with
        first = adjacency matrices of the tree's that make up your corpus.
        second = a sequence of labels of length first.shape[0] representing the labels of each vertex.

        Returns
        -------
        cooccurrence matrix: scipy.sparse.csr_matrix
            A weighted label cooccurrence count matrix
        """
        self.fit(X)
        return self.cooccurrences_

    def transform(self, X):
        """
        Takes a sequence of labelled trees and returns the weighted cooccurrence counts of the labels
        given the label vocabulary learned in the fit.

        Parameters
        ----------
        X: Iterable of tuples (nd.array | scipy.sparse.csr_matrix, label_sequence)
        An iterable of tuples the with
        first = adjacency matrices of the tree's that make up your corpus.
        second = a sequence of labels of length first.shape[0] representing the labels of each vertex.

        Returns
        -------
        cooccurrence matrix: scipy.sparse.csr_matrix
            A weighted label cooccurrence count matrix
        """

        raw_token_sequences = [label_sequence for adjacency, label_sequence in X]
        flat_sequences = flatten(raw_token_sequences)
        (
            clean_tree_sequence,
            self.token_label_dictionary_,
            self.token_index_dictionary_,
            self._token_frequencies_,
        ) = preprocess_tree_sequences(
            X, flat_sequences, self.token_label_dictionary_, masking=self.mask_string
        )

        if callable(self.kernel_function):
            self._kernel_function = self.kernel_function
        elif self.kernel_function in _KERNEL_FUNCTIONS:
            self._kernel_function = _KERNEL_FUNCTIONS[self.kernel_function]
        else:
            raise ValueError(
                f"Unrecognized kernel_function; should be callable or one of {_KERNEL_FUNCTIONS.keys()}"
            )

        self._window_size = self.window_radius

        # build token_cooccurrence_matrix()
        cooccurrences = sequence_tree_skip_grams(
            clean_tree_sequence,
            kernel_function=self._kernel_function,
            window_size=self._window_size,
            label_dictionary=self.token_label_dictionary_,
            window_orientation=self.window_orientation,
        )
        cooccurrences.eliminate_zeros()

        return cooccurrences


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

    def _validate_data(self, X):
        try:
            assert np.isscalar(X[0][0][0])
        except:
            raise ValueError("Input must be a collection of collections of points")

        try:
            dims = [np.array(x).shape[1] for x in X]
        except:
            raise ValueError(
                "Elements of each point collection must be of the same dimension."
            )

        if not hasattr(self, "data_dimension_"):
            self.data_dimension_ = np.mean(dims)

        if not (
            np.max(dims) == self.data_dimension_ or np.min(dims) == self.data_dimension_
        ):
            raise ValueError("Each point collection must be of equal dimension.")

    def fit(self, X, y=None, **fit_params):
        random_state = check_random_state(self.random_state)
        self._validate_params()
        self._validate_data(X)

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
        self._validate_data(X)
        result = np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )


def find_bin_boundaries(flat, n_bins):
    """
    Only uniform distribution is currently implemented.
    TODO: Implement Normal
    :param flat: an iterable.
    :param n_bins:
    :return:
    """
    flat.sort()
    flat_csum = np.cumsum(flat)
    bin_range = flat_csum[-1] / n_bins
    bin_indices = [0]
    for i in range(1, len(flat_csum)):
        if (flat_csum[i] >= bin_range * len(bin_indices)) & (
            flat[i] > flat[bin_indices[-1]]
        ):
            bin_indices.append(i)
    bin_values = np.array(flat, dtype=float)[bin_indices]

    if bin_values.shape[0] < n_bins:
        warn(
            f"Could not generate n_bins={n_bins} bins as there are not enough "
            f"distinct values. Please check your data."
        )

    return bin_values


def expand_boundaries(my_interval_index, absolute_range):
    """
    Expands the outer bind on a pandas IntervalIndex to encompase the range specified by the 2-tuple absolute_range.

    Parameters
    ----------
    my_interval_index: pandas IntervalIndex object (right closed)
    absolute_range: 2-tuple.
        (min_value, max_value)

    Returns
    -------
    index: a pandas IntervalIndex
        A pandas IntervalIndex with the boundaries potentially expanded to encompas the absolute range.
    """
    """
    expands the outer bind on a pandas IntervalIndex to encompase the range specified by the 2-tuple absolute_range
    :param my_interval_index:
    :param absolute_range: 2tuple 
    :return: a pandas IntervalIndex
    """
    interval_list = my_interval_index.to_list()
    # Check if the left boundary needs expanding
    if interval_list[0].left > absolute_range[0]:
        interval_list[0] = pd.Interval(
            left=absolute_range[0], right=interval_list[0].right
        )
    # Check if the right boundary needs expanding
    last = len(interval_list) - 1
    if interval_list[last].right < absolute_range[1]:
        interval_list[last] = pd.Interval(
            left=interval_list[last].left, right=absolute_range[1]
        )
    return pd.IntervalIndex(interval_list)


def add_outier_bins(my_interval_index, absolute_range):
    """
    Appends extra bins to either side our our interval index if appropriate.
    That only occurs if the absolute_range is wider than the observed range in your training data.
    :param my_interval_index:
    :param absolute_range:
    :return:
    """
    interval_list = my_interval_index.to_list()
    # Check if the left boundary needs expanding
    if interval_list[0].left > absolute_range[0]:
        left_outlier = pd.Interval(left=absolute_range[0], right=interval_list[0].left)
        interval_list.insert(0, left_outlier)

    last = len(interval_list) - 1
    if interval_list[last].right < absolute_range[1]:
        right_outlier = pd.Interval(
            left=interval_list[last].right, right=absolute_range[1]
        )
        interval_list.append(right_outlier)
    return pd.IntervalIndex(interval_list)


class HistogramVectorizer(BaseEstimator, TransformerMixin):
    """Convert a time series of binary events into a histogram of
    event occurrences over a time frame. If the data has explicit time stamps
    it can be aggregated over hour of day, day of week, day of month, day of year
    , week of year or month of year.

    Parameters
    ----------
    n_components: int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if n_bins < 2.

    strategy: {‘uniform’, ‘quantile’, 'gmm'}, (default=’quantile’)
        The method to use for bin selection in the histogram. In general the
        quantile option, which will select variable width bins based on the
        distribution of the training data, is suggested, but uniformly spaced
        identically sized bins, or soft gns learned from a Gaussian mixture model
        are also available.

    ground_distance: {'euclidean'}
        The distance to induce between bins.

    absolute_range: (minimum_value_possible, maximum_value_possible) (default=(-np.inf, np.inf))
        By default values outside of training data range are included in the extremal bins.
        You can specify these values if you know something about your values (e.g. (0, np.inf) )

    append_outlier_bins: bool (default=False)
        Whether to add extra bins to catch values outside of your training
        data where appropriate? These bins will increase the total number of
        components (to ``n_components + 2`` and will be the first bin (for
        outlying small data) and the last bin (for outlying large data).
    """

    # TODO: time stamps, generic groupby
    def __init__(
        self,
        n_components=20,
        strategy="uniform",
        ground_distance="euclidean",
        absolute_range=(-np.inf, np.inf),
        append_outlier_bins=False,
    ):
        self.n_components = n_components
        self.strategy = strategy
        self.ground_distance = ground_distance  # Not currently making use of this.
        self.absolute_range = absolute_range
        self.append_outlier_bins = append_outlier_bins

    def _validate_params(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Learns the histogram bins.
        Still need to check switch.
        :param X:
        :return:
        """
        flat = flatten(X)
        flat = list(
            filter(
                lambda n: n > self.absolute_range[0] and n < self.absolute_range[1],
                flat,
            )
        )
        if self.strategy == "uniform":
            self.bin_intervals_ = pd.interval_range(
                start=np.min(flat), end=np.max(flat), periods=self.n_components
            )
        if self.strategy == "quantile":
            self.bin_intervals_ = pd.IntervalIndex.from_breaks(
                find_bin_boundaries(flat, self.n_components)
            )
        if self.append_outlier_bins == True:
            self.bin_intervals_ = add_outier_bins(
                self.bin_intervals_, self.absolute_range
            )
        else:
            self.bin_intervals_ = expand_boundaries(
                self.bin_intervals_, self.absolute_range
            )
        self.metric_ = distances.hellinger
        return self

    def _vector_transform(self, vector):
        """
        Applies the transform to a single row of the data.
        """
        return pd.cut(vector, self.bin_intervals_).value_counts()

    def transform(self, X):
        """
        Apply binning to a full data set returning an nparray.
        """
        check_is_fitted(self, ["bin_intervals_"])
        result = np.ndarray((len(X), len(self.bin_intervals_)))
        for i, seq in enumerate(X):
            result[i, :] = self._vector_transform(seq).values
        return result


def temporal_cyclic_transform(datetime_series, periodicity=None):
    """
    TODO: VERY UNFINISHED
    Replaces all time resolutions above the resolution specified with a fixed value.
    This creates a cycle within a datetime series.
    Parameters
    ----------
    datetime_series: a pandas series of datetime objects
    periodicity: string ['year', 'month' , 'week', 'day', 'hour']
        What time period to create cycles.

    Returns
    -------
    cyclic_series: pandas series of datetime objects

    """
    collapse_times = {}
    if periodicity in ["year", "month", "day", "hour"]:
        collapse_times["year"] = 1970
        if periodicity in ["month", "day", "hour"]:
            collapse_times["month"] = 1
            if periodicity in ["day", "hour"]:
                collapse_times["day"] = 1
                if periodicity in ["hour"]:
                    collapse_times["hour"] = 0
        cyclic_series = datetime_series.apply(lambda x: x.replace(**collapse_times))
    elif periodicity == "week":
        raise NotImplementedError("we have not implemented week cycles yet")
    else:
        raise ValueError(
            f"Sorry resolution={periodicity} is not a valid option.  "
            + f"Please select from ['year', 'month', 'week', 'day', 'hour']"
        )
    return cyclic_series


class CyclicHistogramVectorizer(BaseEstimator, TransformerMixin):
    """

    """

    def __init__(
        self, periodicity="week", resolution="day",
    ):
        self.periodicity = periodicity
        self.resolution = resolution

    def _validate_params(self):
        pass

    def fit(self, X, y=None, **fit_params):
        cyclic_data = temporal_cyclic_transform(
            pd.to_datetime(X), periodicity=self.periodicity
        )
        resampled = (
            pd.Series(index=cyclic_data, data=1).resample(self.resolution).count()
        )
        self.temporal_bins_ = resampled.index
        return self


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
        A function producing a sequence of windows given a source sequence and a window_radius and term frequencies.
        The string options are ['fixed', 'information'] for using pre-defined functions.

    kernel_function: numba.jitted callable or str (optional, default='flat')
        A function producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic'] for using pre-defined functions.

    window_radius: int (optional, default=5)
        Argument to pass through to the window function.  Outside of boundary cases, this is the expected width
        of the (directed) windows produced by the window function.

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

    window_orientation: string (['before', 'after', 'symmetric'])
        The orientation of the cooccurrence window.  Whether to return all the tokens that
        occurred within a window before, after or on either side.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).
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
        window_radius=5,
        validate_data=True,
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
        self.window_radius = window_radius
        self.validate_data = validate_data

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

        n_unique_tokens = len(self._token_dictionary_)

        # Set the kernel and window functions

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

        #  Adjust the window size for the info window
        if self.window_function == "information":
            entropy = np.dot(
                self._token_frequencies_, np.log2(self._token_frequencies_)
            )
            self._window_size = self.window_radius * entropy
        else:
            self._window_size = self.window_radius

        # Build the matrix
        row, col, data = skip_grams_matrix_coo_data(
            token_sequences,
            self._window_function,
            self._kernel_function,
            (self._window_size, self._token_frequencies_),
            tuple([self.window_radius]),
            self._token_dictionary_,
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        column_sums = np.array(base_matrix.sum(axis=0))[0]
        self._column_is_kept = column_sums > 0
        self._kept_columns = np.where(self._column_is_kept)[0]

        self.column_label_dictionary_ = {}
        for i in range(self._kept_columns.shape[0]):
            raw_val = self._kept_columns[i]
            first_token = self._inverse_token_dictionary_[raw_val // n_unique_tokens]
            second_token = self._inverse_token_dictionary_[raw_val % n_unique_tokens]
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
        check_is_fitted(self, ["_token_dictionary_", "_column_is_kept",])
        flat_sequence = flatten(X)
        (token_sequences, _, _, _) = preprocess_token_sequences(
            X, flat_sequence, self._token_dictionary_,
        )

        n_unique_tokens = len(self._token_dictionary_)

        row, col, data = skip_grams_matrix_coo_data(
            token_sequences,
            self._window_function,
            self._kernel_function,
            (self._window_size, self._token_frequencies_),
            tuple([self.window_radius]),
            self._token_dictionary_,
        )

        base_matrix = scipy.sparse.coo_matrix((data, (row, col)))
        result = base_matrix.tocsc()[:, self._column_is_kept].tocsr()

        return result


class NgramVectorizer(BaseEstimator, TransformerMixin):
    """Given a sequence, or list of sequences of tokens, produce a
    count matrix of n-grams of successive tokens.  This either produces n-grams for a fixed size or all n-grams
    up to a fixed size.

    Parameters
    ----------
    ngram_size: int (default = 1)
        The size of the ngrams to count.

    ngram_behaviour: string (optional, default="exact")
        The n-gram behaviour. Should be one of ["exact", "subgrams"] to produce either fixed size ngram_size
        or all ngrams of size upto (and including) ngram_size.

    ngram_dictionary: dictionary or None (optional, default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary
        should be learned from the training data.

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
        determined by min_occurences.

    max_frequency: float or None (optional, default=None)
        The maximal frequency of occurrence of a token for it to be considered and
        counted. If None then there is no constraint, or the constraint is
        determined by max_occurences.

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

    token_dictionary: dictionary or None (optional, default=None)
        A dictionary mapping tokens to indices

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).
    """

    def __init__(
        self,
        ngram_size=1,
        ngram_behaviour="exact",
        ngram_dictionary=None,
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
        validate_data=True,
    ):
        self.ngram_size = ngram_size
        self.ngram_behaviour = ngram_behaviour
        self.ngram_dictionary = ngram_dictionary
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
        self.validate_data = validate_data

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

        if self.ngram_dictionary is not None:
            self.column_label_dictionary_ = self.ngram_dictionary
        else:
            if self.ngram_size == 1:
                self.column_label_dictionary_ = self._token_dictionary_
            else:
                self.column_label_dictionary_ = defaultdict()
                self.column_label_dictionary_.default_factory = (
                    self.column_label_dictionary_.__len__
                )

        indptr = [0]
        indices = []
        data = []
        for sequence in token_sequences:
            counter = {}
            numba_sequence = np.array(sequence)
            for index_gram in ngrams_of(
                numba_sequence, self.ngram_size, self.ngram_behaviour
            ):
                try:
                    if len(index_gram) == 1:
                        token_gram = self._inverse_token_dictionary_[index_gram[0]]
                    else:
                        token_gram = tuple(
                            self._inverse_token_dictionary_[index]
                            for index in index_gram
                        )
                    col_index = self.column_label_dictionary_[token_gram]
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
        self.column_label_dictionary_ = dict(self.column_label_dictionary_)
        self.column_index_dictionary_ = {
            index: token for token, index in self.column_label_dictionary_.items()
        }

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32
        indices = np.asarray(indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        data = np.asarray(data, dtype=np.intc)

        self._train_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.column_label_dictionary_)),
            dtype=np.float32,
        )
        self._train_matrix.sort_indices()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self._train_matrix

    def transform(self, X):
        check_is_fitted(
            self,
            [
                "_token_dictionary_",
                "_inverse_token_dictionary_",
                "column_label_dictionary_",
            ],
        )
        flat_sequence = flatten(X)
        (token_sequences, _, _, _) = preprocess_token_sequences(
            X, flat_sequence, self._token_dictionary_,
        )

        indptr = [0]
        indices = []
        data = []

        for sequence in token_sequences:
            counter = {}
            numba_sequence = np.array(sequence)
            for index_gram in ngrams_of(
                numba_sequence, self.ngram_size, self.ngram_behaviour
            ):
                try:
                    if len(index_gram) == 1:
                        token_gram = self._inverse_token_dictionary_[index_gram[0]]
                    else:
                        token_gram = tuple(
                            self._inverse_token_dictionary_[index]
                            for index in index_gram
                        )
                    col_index = self.column_label_dictionary_[token_gram]
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
        data = np.asarray(data, dtype=np.intc)

        result = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.column_label_dictionary_)),
            dtype=np.float32,
        )
        result.sort_indices()

        return result


class KDEVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bandwidth=None,
        n_components=50,
        kernel="gaussian",
        evaluation_grid_strategy="uniform",
    ):
        self.n_components = n_components
        self.evaluation_grid_strategy = evaluation_grid_strategy
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y=None, **fit_params):

        combined_data = np.array(flatten(X))

        if self.bandwidth is None:
            # Estimate the bandwidth by looking at training data
            # We do a jack-knife across each time series and
            # find the bandwidth choice that works best over all
            # time series
            min, max = np.min(combined_data), np.max(combined_data)
            avg_n_events = np.mean([len(x) for x in X])
            max_bandwidth = (max - min) / avg_n_events
            min_bandwidth = min_non_zero_difference(combined_data)
            bandwidths = 10.0 ** np.linspace(
                np.log10(min_bandwidth), np.log10(max_bandwidth), 50
            )
            jackknifed_total_likelihoods = jackknife_bandwidths(X, bandwidths)
            self.bandwidth_ = bandwidths[np.argmax(jackknifed_total_likelihoods)]
        else:
            self.bandwidth_ = self.bandwidth

        if self.evaluation_grid_strategy == "uniform":
            min, max = np.min(combined_data), np.max(combined_data)
            self.evaluation_grid_ = np.linspace(min, max, self.n_components)
        elif self.evaluation_grid_strategy == "density":
            uniform_quantile_grid = np.linspace(0, 1.0, self.n_components)
            self.evaluation_grid_ = np.quantile(combined_data, uniform_quantile_grid)
        else:
            raise ValueError(
                "Unrecognized evaluation_grid_strategy; should be one "
                'of: "uniform" or "density"'
            )

        return self

    def transform(self, X):
        check_is_fitted(self, ["bandwidth_", "evaluation_grid_"])

        result = np.empty((len(X), self.n_components), dtype=np.float64)

        for i, sample in enumerate(X):
            kde = KernelDensity(bandwidth=self.bandwidth_, kernel=self.kernel)
            kde.fit(sample[:, None])
            log_probability = kde.score_samples(self.evaluation_grid_[:, None])
            result[i] = np.exp(log_probability)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class ProductDistributionVectorizer(BaseEstimator, TransformerMixin):
    pass


class Wasserstein1DHistogramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        X = check_array(X)
        normalized_X = normalize(X, norm="l1")
        result = np.cumsum(normalized_X, axis=1)
        self.metric_ = "l1"
        return result


class SequentialDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        result = []

        for sequence in X:
            seq = np.array(sequence)
            result.append(seq[self.offset :] - seq[: -self.offset])

        return result
