"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from numba.typed import List
import scipy.linalg
import scipy.stats
import scipy.sparse
import re


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

    max_frequency: float (optional, default=1.0)
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

    excluded_token_regex: str (optional, default=None)
        A regular expression which constrains the vocabulary to exclude tokens that match the expression.

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

    # Prune by document frequency
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

    excluded_token_regex: str (optional, default=None)
        A regular expression which constrains the vocabulary to exclude tokens that match the expression.

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

    # We will prune the edges from any nodes who's labels are to be filtered and
    # reconnect their parents with their children.
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
    token_sequences: Iterable of (tuple | list | numpy.array)
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

    excluded_token_regex: str (optional, default=None)
        A regular expression which constrains the vocabulary to exclude tokens that match the expression.

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
                    dtype=np.int32,
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
                    dtype=np.int32,
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
