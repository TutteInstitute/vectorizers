from warnings import warn
import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
import scipy.linalg
import scipy.stats
import scipy.sparse
from .preprocessing import preprocess_tree_sequences
from .utils import (
    flatten,
    sparse_collapse,
)
import vectorizers.distances as distances

from ._window_kernels import _KERNEL_FUNCTIONS

MOCK_DICT = numba.typed.Dict()
MOCK_DICT[(-1, -1)] = -1


def build_tree_skip_grams(
    token_sequence,
    adjacency_matrix,
    kernel_function,
    kernel_args,
    window_size,
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
    weights = kernel_function(np.arange(window_size), window_size, *kernel_args)
    count_matrix = adjacency_matrix * weights[0]
    walk = adjacency_matrix
    for i in range(1, window_size):
        walk = walk @ adjacency_matrix
        count_matrix += walk * weights[i]

    # Now collapse the rows and columns with the same label
    (grouped_matrix, new_labels) = sparse_collapse(count_matrix, token_sequence)

    return grouped_matrix, new_labels


def sequence_tree_skip_grams(
    tree_sequences,
    kernel_function,
    kernel_args,
    window_size,
    label_dictionary,
    window_orientation,
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

    kernel_args: tuple
        Arguments to pass through to the kernel function

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
            kernel_args=kernel_args,
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


class LabelledTreeCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a sequence of labelled trees and counts and produces a cooccurrence count matrix
    of their labels.
    For our purposes we take a labelled tree to be a tuple containing an adjacency matrix and an
    array of it's vertex labels.

    Parameters
    ----------

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

    excluded_token_regex: str or None (optional, default=None)
        The regular expression by which tokens are ignored if re.fullmatch returns True.

    kernel_function: numba.jitted callable or str (optional, default='flat')
        A function producing weights given a window of tokens and a window_radius.
        The string options are ['flat', 'triangular', 'harmonic'] for using pre-defined functions.

    kernel_args: dict (optional, default = None)
        Optional arguments to pass the kernel function

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
        kernel_args={},
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

        self.kernel_args = kernel_args
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
            Each tuple in this sequence represents a labelled tree.
            The first element is a (sparse) adjacency matrix
            The second element is an array of node labels

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

        self._kernel_args = {"mask_index": None, "normalize": False, "offset": 0}
        self._kernel_args.update(self.kernel_args)
        self._kernel_args = tuple(self._kernel_args.values())

        self._window_size = self.window_radius

        # build token_cooccurrence_matrix()
        self.cooccurrences_ = sequence_tree_skip_grams(
            clean_tree_sequence,
            kernel_function=self._kernel_function,
            kernel_args=self.kernel_args,
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
            Each tuple in this sequence represents a labelled tree.
            The first element is a (sparse) adjacency matrix
            The second element is an array of node labels

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
            Each tuple in this sequence represents a labelled tree.
            The first element is a (sparse) adjacency matrix
            The second element is an array of node labels


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
            kernel_args=self.kernel_args,
            window_size=self._window_size,
            label_dictionary=self.token_label_dictionary_,
            window_orientation=self.window_orientation,
        )
        cooccurrences.eliminate_zeros()

        return cooccurrences
