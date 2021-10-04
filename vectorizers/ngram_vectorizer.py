import numpy as np
import numba

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from collections import defaultdict
import scipy.linalg
import scipy.stats
import scipy.sparse

from .utils import (
    flatten,
    validate_homogeneous_token_types,
)

from .preprocessing import preprocess_token_sequences


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
            X,
            flat_sequence,
            self._token_dictionary_,
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
