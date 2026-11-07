import itertools

import numpy as np
import numba
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

import scipy.linalg
import scipy.stats
import scipy.sparse

from .utils import validate_homogeneous_token_types
from .preprocessing import prune_token_dictionary, token_pruning_mask


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


def _factorize_documents(X, validate=False):
    """Encode all tokens of a list of token sequences in a single pass.

    Returns
    -------
    codes: array of shape (n_tokens,)
        The index of each token in ``unique_tokens``.

    unique_tokens: list
        The distinct tokens, in sorted order.

    offsets: array of shape (len(X) + 1,)
        Document ``i`` is ``codes[offsets[i]:offsets[i + 1]]``.
    """
    offsets = np.zeros(len(X) + 1, dtype=np.int64)
    np.cumsum(np.fromiter(map(len, X), dtype=np.int64, count=len(X)), out=offsets[1:])
    flat = np.fromiter(
        itertools.chain.from_iterable(X), dtype=object, count=offsets[-1]
    )
    codes, uniques = pd.factorize(flat)
    if codes.shape[0] and codes.min() < 0:
        # None/NaN tokens are rare; keeping them as tokens is a slower path
        codes, uniques = pd.factorize(flat, use_na_sentinel=False)
    if validate:
        validate_homogeneous_token_types([list(uniques)])
    order = sorted(range(len(uniques)), key=uniques.__getitem__)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[codes], [uniques[i] for i in order], offsets


def _compress(values, keep, offsets):
    """Remove the entries of a flat segmented array where ``keep`` is False,
    adjusting the segment offsets to match."""
    if keep.all():
        return values, offsets
    return values[keep], np.concatenate(([0], np.cumsum(keep)))[offsets]


def _encode_tokens(token_codes, unique_tokens, offsets, token_dictionary, unknown=-1):
    """Index factorized tokens by ``token_dictionary``. Tokens not in it become
    ``unknown``, and are dropped if that is -1."""
    index = np.array(
        [token_dictionary.get(token, unknown) for token in unique_tokens],
        dtype=np.int64,
    )[token_codes]
    return _compress(index, index >= 0, offsets)


@numba.njit(nogil=True, parallel=True)
def _ngram_index_rows(tokens, offsets, ngram_size, subgrams):
    """All n-grams of each segment as rows of token indices, padded with -1."""
    min_size = 1 if subgrams else ngram_size
    n_docs = offsets.shape[0] - 1
    gram_offsets = np.zeros(n_docs + 1, dtype=np.int64)
    for d in range(n_docs):
        for size in range(min_size, ngram_size + 1):
            gram_offsets[d + 1] += max(offsets[d + 1] - offsets[d] - size + 1, 0)
    gram_offsets = np.cumsum(gram_offsets)

    grams = np.full((gram_offsets[-1], ngram_size), -1, dtype=np.int32)
    for d in numba.prange(n_docs):
        pos = gram_offsets[d]
        end = offsets[d + 1]
        for i in range(offsets[d], end):
            for size in range(min_size, min(ngram_size, end - i) + 1):
                grams[pos, :size] = tokens[i : i + size]
                pos += 1
    return grams, gram_offsets


@numba.njit(nogil=True, parallel=True)
def _segment_count_arrays(columns, offsets):
    """CSR (indptr, indices, data) counting the non-negative values of each
    segment."""
    n_rows = offsets.shape[0] - 1
    columns = columns.copy()
    row_nnz = np.zeros(n_rows + 1, dtype=np.int64)
    for r in numba.prange(n_rows):
        segment = columns[offsets[r] : offsets[r + 1]]
        segment.sort()
        nnz = 0
        for i in range(segment.shape[0]):
            if segment[i] >= 0 and (i == 0 or segment[i] != segment[i - 1]):
                nnz += 1
        row_nnz[r + 1] = nnz
    indptr = np.cumsum(row_nnz)

    indices = np.empty(indptr[-1], dtype=columns.dtype)
    data = np.zeros(indptr[-1], dtype=np.int64)
    for r in numba.prange(n_rows):
        segment = columns[offsets[r] : offsets[r + 1]]
        pos = indptr[r] - 1
        for i in range(segment.shape[0]):
            if segment[i] >= 0:
                if i == 0 or segment[i] != segment[i - 1]:
                    pos += 1
                    indices[pos] = segment[i]
                data[pos] += 1
    return indptr, indices, data


def _row_keys(rows, base):
    """Integer keys for the rows of a -1 padded array of indices less than
    ``base - 1``, equal exactly for equal rows and ordered as the rows are
    lexicographically (so shorter rows precede their extensions)."""
    keys = np.zeros(rows.shape[0], dtype=np.int64)
    n_keys = 1
    for k in range(rows.shape[1]):
        if n_keys * base > np.iinfo(np.int64).max:
            keys, uniques = pd.factorize(keys, sort=True)
            n_keys = len(uniques)
        keys = keys * base + rows[:, k] + 1
        n_keys *= base
    return keys


def _row_ids(rows, base):
    """Consecutive ids in the order of _row_keys, and the index of a
    representative row for each id."""
    ids, uniques = pd.factorize(_row_keys(rows, base), sort=True)
    representative = np.empty(len(uniques), dtype=np.int64)
    representative[ids] = np.arange(ids.shape[0])
    return ids, representative


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

    max_unique_tokens: int or None (optional, default=None)
        The maximal number of elements contained in the vocabulary.  If not None, this
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

    excluded_tokens: set or None (optional, default=None)
        A set of tokens that should be ignored entirely. If None then no tokens will
        be ignored in this fashion.

    excluded_token_regex: str or None (optional, default=None)
        The regular expression by which tokens are ignored if re.fullmatch returns True.

    mask_string: str (optional, default=None)
        Prunes the filtered tokens when None, so that n-grams span the gaps they
        leave; otherwise replaces them with the provided mask_string.

    nullify_mask: bool (optional, default=False)
        Drop every n-gram containing the mask_string, so that only n-grams made
        entirely of retained tokens, without spanning gaps, are counted.
        Requires mask_string.

    validate_data: bool (optional, default=True)
        Check whether the data is valid (e.g. of homogeneous token type).
    """

    ATTRIBUTES_EXPECTED_IDENTICAL = [
        "ngram_size",
        "ngram_behaviour",
        "ngram_dictionary",
        "token_dictionary",
        "max_unique_tokens",
        "min_occurrences",
        "max_occurrences",
        "min_frequency",
        "max_frequency",
        "min_document_occurrences",
        "max_document_occurrences",
        "min_document_frequency",
        "max_document_frequency",
        "excluded_tokens",
        "excluded_token_regex",
        "mask_string",
        "nullify_mask",
        "validate_data",
    ]

    def __init__(
        self,
        ngram_size=1,
        ngram_behaviour="exact",
        ngram_dictionary=None,
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
        mask_string=None,
        nullify_mask=False,
        validate_data=True,
    ):
        self.ngram_size = ngram_size
        self.ngram_behaviour = ngram_behaviour
        self.ngram_dictionary = ngram_dictionary
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
        self.mask_string = mask_string
        self.nullify_mask = nullify_mask
        self.validate_data = validate_data
        self.column_label_dictionary_ = {}
        self.column_index_dictionary_ = {}
        self._mask_index = None

    def _pruning_parameters(self):
        return dict(
            max_unique_tokens=self.max_unique_tokens,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            min_occurrences=self.min_occurrences,
            max_occurrences=self.max_occurrences,
            min_document_frequency=self.min_document_frequency,
            max_document_frequency=self.max_document_frequency,
            min_document_occurrences=self.min_document_occurrences,
            max_document_occurrences=self.max_document_occurrences,
        )

    def _document_frequencies(self, codes, offsets, n_codes):
        if {
            self.min_document_frequency,
            self.min_document_occurrences,
            self.max_document_frequency,
            self.max_document_occurrences,
        } == {None}:
            return np.array([])
        _, indices, _ = _segment_count_arrays(codes, offsets)
        return np.bincount(indices, minlength=n_codes) / (len(offsets) - 1)

    def _columns_by_label(self, rows):
        # A user supplied ngram_dictionary is keyed by bare tokens for unigrams
        bare_unigrams = self.ngram_dictionary is not None
        columns = np.empty(rows.shape[0], dtype=np.int64)
        for i, row in enumerate(rows.tolist()):
            gram = tuple(self._inverse_token_dictionary_[t] for t in row if t >= 0)
            if bare_unigrams and len(gram) == 1:
                gram = gram[0]
            columns[i] = self.column_label_dictionary_.get(gram, -1)
        return columns

    def _ngrams(self, tokens, offsets):
        """The n-grams of the encoded documents as -1 padded rows of token indices,
        without those containing a nullified mask, and their offsets."""
        grams, offsets = _ngram_index_rows(
            tokens.astype(np.int32),
            offsets,
            self.ngram_size,
            self.ngram_behaviour == "subgrams",
        )
        if self._mask_index is not None:
            grams, offsets = _compress(
                grams, ~(grams == self._mask_index).any(axis=1), offsets
            )
        return grams, offsets

    def _ngram_columns(self, tokens, offsets):
        """The column of every n-gram of the encoded documents (-1 if it is not a
        column), and the offsets delimiting each document's n-grams."""
        if self.ngram_size == 1 and self.ngram_dictionary is None:
            if self._mask_index is not None:
                tokens = np.where(tokens == self._mask_index, -1, tokens)
            return tokens, offsets
        grams, offsets = self._ngrams(tokens, offsets)
        base = len(self._token_dictionary_) + 1
        ngram_rows = getattr(self, "_ngram_rows_", None)
        if ngram_rows is None:
            ids, representative = _row_ids(grams, base)
            return self._columns_by_label(grams[representative])[ids], offsets
        # Key the learned n-grams together with the new ones, so keys are shared
        keys = _row_keys(np.vstack([ngram_rows, grams]), base)
        n_learned = len(ngram_rows)
        return pd.Index(keys[:n_learned]).get_indexer(keys[n_learned:]), offsets

    def _count_matrix(self, columns, offsets):
        indptr, indices, data = _segment_count_arrays(columns, offsets)
        result = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(offsets) - 1, len(self.column_label_dictionary_)),
            dtype=np.float32,
        )
        result.sort_indices()
        return result

    def _fit_tokens(self, X):
        """Learn the token dictionary; return the encoded documents."""
        token_codes, unique_tokens, offsets = _factorize_documents(
            X, validate=self.validate_data
        )
        if self.token_dictionary is None:
            token_dictionary, token_frequencies = prune_token_dictionary(
                dict(zip(unique_tokens, range(len(unique_tokens)))),
                np.bincount(token_codes).astype(np.float32) / len(token_codes),
                token_doc_frequencies=self._document_frequencies(
                    token_codes, offsets, len(unique_tokens)
                ),
                ignored_tokens=self.excluded_tokens,
                excluded_token_regex=self.excluded_token_regex,
                total_tokens=len(token_codes),
                total_documents=len(offsets) - 1,
                **self._pruning_parameters(),
            )
        else:
            token_dictionary = self.token_dictionary
            known, _ = _encode_tokens(
                token_codes, unique_tokens, offsets, token_dictionary
            )
            token_frequencies = np.bincount(known).astype(np.float32) / len(
                token_codes
            )

        unknown_token = -1
        if self.mask_string is not None:
            token_dictionary.pop(self.mask_string, None)
            unknown_token = len(token_dictionary)
        tokens, offsets = _encode_tokens(
            token_codes, unique_tokens, offsets, token_dictionary, unknown_token
        )
        if self.mask_string is not None:
            token_dictionary[self.mask_string] = unknown_token

        self._token_dictionary_ = token_dictionary
        self._inverse_token_dictionary_ = {
            index: token for token, index in token_dictionary.items()
        }
        self._token_frequencies_ = token_frequencies
        self._mask_index = np.int32(unknown_token) if self.nullify_mask else None
        return tokens, offsets

    def _fit_ngram_columns(self, tokens, offsets):
        """Learn the n-gram dictionary; return the column of every n-gram."""
        n_documents = len(offsets) - 1
        grams, offsets = self._ngrams(tokens, offsets)
        ids, representative = _row_ids(grams, len(self._token_dictionary_) + 1)
        kept = np.flatnonzero(
            token_pruning_mask(
                np.bincount(ids).astype(np.float32) / len(ids),
                token_doc_frequencies=self._document_frequencies(
                    ids, offsets, len(representative)
                ),
                total_tokens=len(ids),
                total_documents=n_documents,
                **self._pruning_parameters(),
            )
        )
        # Sorted, so the position of an n-gram's row is its column index
        self._ngram_rows_ = grams[representative[kept]]
        self.column_label_dictionary_ = {
            tuple(self._inverse_token_dictionary_[t] for t in row if t >= 0): i
            for i, row in enumerate(self._ngram_rows_.tolist())
        }
        column_of_id = np.full(len(representative), -1, dtype=np.int64)
        column_of_id[kept] = np.arange(len(kept))
        return column_of_id[ids], offsets

    def fit(self, X, y=None, **fit_params):
        if self.ngram_behaviour not in ("exact", "subgrams"):
            raise ValueError("Unrecognized ngram_behaviour!")
        if self.nullify_mask and self.mask_string is None:
            raise ValueError("Cannot nullify mask with mask_string = None")

        tokens, offsets = self._fit_tokens(X)
        self._ngram_rows_ = None
        if self.ngram_dictionary is None and self.ngram_size > 1:
            columns, offsets = self._fit_ngram_columns(tokens, offsets)
        else:
            self.column_label_dictionary_ = (
                self._token_dictionary_
                if self.ngram_dictionary is None
                else self.ngram_dictionary
            )
            columns, offsets = self._ngram_columns(tokens, offsets)
        self.column_index_dictionary_ = {
            index: token for token, index in self.column_label_dictionary_.items()
        }

        self._train_matrix = self._count_matrix(columns, offsets)
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
        token_codes, unique_tokens, offsets = _factorize_documents(X)
        # Mask unseen tokens as fit does, so n-grams never bridge them
        unknown_token = (
            -1
            if self.mask_string is None
            else self._token_dictionary_.get(self.mask_string, -1)
        )
        tokens, offsets = _encode_tokens(
            token_codes, unique_tokens, offsets, self._token_dictionary_, unknown_token
        )
        columns, offsets = self._ngram_columns(tokens, offsets)
        return self._count_matrix(columns, offsets)

    def __add__(self, other):
        try:
            check_is_fitted(
                self,
                [
                    "_token_dictionary_",
                    "_inverse_token_dictionary_",
                    "column_label_dictionary_",
                ],
            )
        except NotFittedError:
            return other

        try:
            check_is_fitted(
                other,
                [
                    "_token_dictionary_",
                    "_inverse_token_dictionary_",
                    "column_label_dictionary_",
                ],
            )
        except NotFittedError:
            return self

        if any(
            getattr(self, a) != getattr(other, a)
            for a in self.ATTRIBUTES_EXPECTED_IDENTICAL
        ):
            raise NotImplementedError(
                "Unimplemented merging between these two vectorizer instances, given their respective parameter sets."
            )
        if self.ngram_size > 1:
            raise NotImplementedError(
                "Unimplemented merging for ngram sizes greater than one"
            )

        disjoint_vocab = set(other.column_index_dictionary_.values()).difference(
            self.column_index_dictionary_.values()
        )
        joint_column_index_dictionary = self.column_index_dictionary_.copy()
        left_vocab_size = len(self.column_index_dictionary_)
        for i, x in enumerate(disjoint_vocab, start=left_vocab_size):
            joint_column_index_dictionary[i] = x
        joint_column_label_dictionary = {
            y: x for x, y in joint_column_index_dictionary.items()
        }
        right_to_joint_index_map = {
            other.column_label_dictionary_[word]: joint_column_label_dictionary[word]
            for word in other.column_label_dictionary_.keys()
        }

        new_cols = len(joint_column_index_dictionary)
        top_train_matrix = self._train_matrix.copy()
        top_train_matrix.resize((self._train_matrix.shape[0], new_cols))

        bottom_joint_matrix = other._train_matrix.tocoo().copy()
        bottom_joint_matrix.resize((other._train_matrix.shape[0], new_cols))
        bottom_joint_matrix.col = np.array(
            [right_to_joint_index_map[x] for x in other._train_matrix.tocoo().col]
        )

        joint_train_matrix = scipy.sparse.vstack(
            [top_train_matrix, bottom_joint_matrix]
        )
        joint_vectorizer = NgramVectorizer(
            **{a: getattr(self, a) for a in self.ATTRIBUTES_EXPECTED_IDENTICAL}
        )
        joint_vectorizer.column_index_dictionary_ = joint_column_index_dictionary
        joint_vectorizer.column_label_dictionary_ = joint_column_label_dictionary
        joint_vectorizer._train_matrix = joint_train_matrix.tocsr()
        joint_vectorizer._token_dictionary_ = joint_vectorizer.column_label_dictionary_
        # Since we don't store the number of tokens we merge the token_frequencies as their unweighted average
        # It would be better if we stored the token counts to do the right thing of using a weighted average
        top_freq = np.concatenate(
            (self._token_frequencies_, np.zeros((len(disjoint_vocab),)))
        )
        bottom_freq = np.zeros(top_freq.shape)
        for x in range(len(other.column_index_dictionary_)):
            bottom_freq[right_to_joint_index_map[x]] = other._token_frequencies_[x]
        joint_vectorizer._token_frequencies_ = np.mean([top_freq, bottom_freq], axis=0)
        joint_vectorizer._inverse_token_dictionary_ = (
            joint_vectorizer.column_label_dictionary_
        )
        return joint_vectorizer
