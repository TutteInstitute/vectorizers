"""Byte-pair encoding for labelled directed trees.

Inputs use the same public tree convention as ``LabelledTreeCooccurrenceVectorizer``:
each tree is ``(adjacency_matrix, label_sequence)``.  The adjacency matrix is a
square dense or sparse matrix with parent-to-child directed edges, and the label
sequence contains one string label per node.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

EdgeKey = tuple[int, int]
Tree = tuple[Any, Sequence[str]]
EncodedTree = tuple[scipy.sparse.csr_matrix, np.ndarray]

UNKNOWN_LABEL = "<UNK>"
UNKNOWN_CODE = 0
_VALID_RETURN_TYPES = ("matrix", "trees", "tokens")


def tree_bpe_token(rank: int) -> str:
    """Return the stable string label for the ``rank``-th learned tree-BPE code."""

    if rank < 0:
        raise ValueError("tree-BPE rank must be nonnegative.")
    return f"__tree_bpe_{rank}__"


@dataclass(frozen=True, slots=True)
class TreeBPERule:
    """One learned parent/child label-pair contraction rule."""

    rank: int
    token: str
    parent_label: str
    child_label: str
    parent_code: int
    child_code: int
    code: int
    count: int
    actual_events: int


@dataclass(frozen=True, slots=True)
class _PairSelection:
    key: EdgeKey
    count: int


@dataclass(slots=True)
class _TokenCodec:
    token_to_id: dict[str, int] = field(default_factory=lambda: {UNKNOWN_LABEL: UNKNOWN_CODE})
    id_to_token: list[str] = field(default_factory=lambda: [UNKNOWN_LABEL])
    sort_key_by_id: list[str] = field(default_factory=lambda: [repr(UNKNOWN_LABEL)])

    def intern(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        if token_id is not None:
            return token_id
        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)
        self.sort_key_by_id.append(repr(token))
        return token_id

    def decode(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def sort_key(self, token_id: int) -> str:
        return self.sort_key_by_id[token_id]


def _as_tree_list(X: Iterable[Tree]) -> list[Tree]:
    trees = list(X)
    if not trees:
        raise ValueError("X must contain at least one labelled tree.")
    return trees


def _adjacency_edges(adjacency: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if scipy.sparse.issparse(adjacency):
        coo = adjacency.tocoo(copy=True)
        coo.sum_duplicates()
        coo.eliminate_zeros()
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        shape = coo.shape
    else:
        array = np.asarray(adjacency)
        if array.ndim != 2:
            raise ValueError("adjacency_matrix must be two-dimensional.")
        rows, cols = np.nonzero(array)
        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
        shape = array.shape
    if rows.shape[0] > 1:
        order = np.lexsort((cols, rows))
        rows = rows[order]
        cols = cols[order]
    return rows, cols, shape


def _normalize_tree(tree: Tree, *, validate_data: bool) -> tuple[list[int], list[list[int]], list[str]]:
    try:
        adjacency, labels = tree
    except (TypeError, ValueError) as exc:
        raise TypeError("each tree must be a pair (adjacency_matrix, label_sequence).") from exc

    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("label_sequence must be one-dimensional.")
    if label_array.shape[0] == 0:
        raise ValueError("a labelled tree must contain at least one node.")
    label_list = label_array.tolist()
    if validate_data:
        bad_labels = [label for label in label_list if not isinstance(label, str)]
        if bad_labels:
            raise TypeError("all tree labels must be strings.")

    n_nodes = len(label_list)
    rows, cols, shape = _adjacency_edges(adjacency)
    if shape != (n_nodes, n_nodes):
        raise ValueError("adjacency_matrix shape must match the label_sequence length.")

    if validate_data:
        if rows.shape[0] != n_nodes - 1:
            raise ValueError("a directed tree with n nodes must contain exactly n - 1 edges.")
        if np.any(rows >= cols):
            bad = np.flatnonzero(rows >= cols)[0]
            edge = (int(rows[bad]), int(cols[bad]))
            raise ValueError(
                "tree edges must be directed from lower to higher node index; "
                f"found edge {edge!r}."
            )

    parent = [-1] * n_nodes
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    indegree = np.zeros(n_nodes, dtype=np.int64)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if parent[col] != -1 and validate_data:
            raise ValueError(f"node {col} has more than one parent.")
        parent[col] = row
        children[row].append(col)
        indegree[col] += 1

    if validate_data:
        roots = np.flatnonzero(indegree == 0)
        if roots.shape[0] != 1:
            raise ValueError(f"expected exactly one directed root; found {roots.shape[0]}.")
        non_roots = np.flatnonzero(indegree != 0)
        if np.any(indegree[non_roots] != 1):
            raise ValueError("every non-root node must have exactly one parent.")

    for row in children:
        row.sort()
    return parent, children, label_list


@dataclass(slots=True)
class _CompactEdgeTree:
    parent: list[int]
    children: list[list[int]]
    label: list[int]
    alive: list[bool]
    edge_index: dict[EdgeKey, set[int]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def from_normalized(
        cls,
        parent: Sequence[int],
        children: Sequence[Sequence[int]],
        labels: Sequence[int],
        *,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> "_CompactEdgeTree":
        state = cls(
            parent=list(parent),
            children=[list(row) for row in children],
            label=list(labels),
            alive=[True] * len(labels),
        )
        state.rebuild_edge_index(pair_counts=pair_counts)
        return state

    def rebuild_edge_index(self, *, pair_counts: Counter[EdgeKey] | None = None) -> None:
        self.edge_index = defaultdict(set)
        for child in range(len(self.parent)):
            if self._edge_is_live(child):
                self._add_edge(child, pair_counts=pair_counts)

    def _edge_is_live(self, child: int) -> bool:
        if child < 0 or child >= len(self.alive) or not self.alive[child]:
            return False
        parent = self.parent[child]
        return 0 <= parent < len(self.alive) and self.alive[parent]

    def _edge_key_unchecked(self, child: int) -> EdgeKey:
        parent = self.parent[child]
        return (self.label[parent], self.label[child])

    def _edge_key(self, child: int) -> EdgeKey:
        if not self._edge_is_live(child):
            raise ValueError(f"node {child!r} does not have a live incoming edge.")
        return self._edge_key_unchecked(child)

    def _add_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        self.edge_index[key].add(child)
        if pair_counts is not None:
            pair_counts[key] += 1

    def _remove_edge(self, child: int, *, pair_counts: Counter[EdgeKey] | None) -> None:
        key = self._edge_key_unchecked(child)
        bucket = self.edge_index.get(key)
        if bucket is None or child not in bucket:
            raise RuntimeError(f"live edge for child {child!r} is missing from its bucket.")
        bucket.remove(child)
        if not bucket:
            self.edge_index.pop(key, None)
        if pair_counts is not None:
            pair_counts[key] -= 1
            if pair_counts[key] < 0:
                raise RuntimeError(f"edge-pair count for {key!r} became negative.")
            if pair_counts[key] == 0:
                pair_counts.pop(key, None)

    def _edge_sort_key(self, child: int) -> tuple[int, int]:
        return (child, self.parent[child])

    def contract_pair(
        self,
        key: EdgeKey,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None = None,
    ) -> int:
        bucket = self.edge_index.get(key)
        if not bucket:
            return 0
        candidates = sorted(bucket, key=self._edge_sort_key)
        used: set[int] = set()
        events = 0
        for child in candidates:
            if not self._edge_is_live(child):
                continue
            parent = self.parent[child]
            if parent in used or child in used or self._edge_key(child) != key:
                continue
            self._contract_edge(parent, child, new_label=new_label, pair_counts=pair_counts)
            used.add(parent)
            used.add(child)
            events += 1
        return events

    def _contract_edge(
        self,
        parent_node: int,
        child_node: int,
        *,
        new_label: int,
        pair_counts: Counter[EdgeKey] | None,
    ) -> None:
        if not self._edge_is_live(child_node) or self.parent[child_node] != parent_node:
            raise ValueError("attempted to contract a non-live edge occurrence.")

        grandparent = self.parent[parent_node]
        old_parent_children = self.children[parent_node]
        child_children = self.children[child_node]

        if grandparent != -1:
            self._remove_edge(parent_node, pair_counts=pair_counts)

        remaining: list[int] = []
        found = False
        for current in old_parent_children:
            self._remove_edge(current, pair_counts=pair_counts)
            if current == child_node:
                found = True
            else:
                remaining.append(current)
        if not found:
            raise RuntimeError("contracted child is missing from parent child list.")

        for current in child_children:
            self._remove_edge(current, pair_counts=pair_counts)

        self.label[parent_node] = new_label
        for current in child_children:
            self.parent[current] = parent_node
        remaining.extend(child_children)
        self.children[parent_node] = remaining

        self.alive[child_node] = False
        self.parent[child_node] = -1
        self.children[child_node] = []

        if grandparent != -1:
            self._add_edge(parent_node, pair_counts=pair_counts)
        for current in remaining:
            self._add_edge(current, pair_counts=pair_counts)

    def to_labeled_tree(self, decode=None) -> EncodedTree:
        live = [node for node, keep in enumerate(self.alive) if keep]
        mapping = {old: new for new, old in enumerate(live)}
        rows: list[int] = []
        cols: list[int] = []
        labels: list[Any] = []
        for old in live:
            labels.append(decode(self.label[old]) if decode is not None else self.label[old])
            old_parent = self.parent[old]
            if old_parent != -1:
                rows.append(mapping[old_parent])
                cols.append(mapping[old])
        adjacency = scipy.sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(live), len(live)),
            dtype=np.float32,
        )
        dtype = object if decode is not None else np.int64
        return adjacency, np.asarray(labels, dtype=dtype)


def _select_best_pair(
    counts: Counter[EdgeKey],
    codec: _TokenCodec,
    *,
    min_pair_count: int,
) -> _PairSelection | None:
    best_key: EdgeKey | None = None
    best_priority: tuple[int, str, str] | None = None
    best_count = 0
    for key, count in counts.items():
        if count < min_pair_count:
            continue
        parent_id, child_id = key
        priority = (int(count), codec.sort_key(parent_id), codec.sort_key(child_id))
        if best_priority is None or priority > best_priority:
            best_priority = priority
            best_key = key
            best_count = int(count)
    if best_key is None:
        return None
    return _PairSelection(best_key, best_count)


def _encode_base_labels(labels: Sequence[str], label_dictionary: dict[str, int]) -> list[int]:
    return [label_dictionary.get(label, UNKNOWN_CODE) for label in labels]


def tree_bpe_train(
    tree_sequence: Iterable[Tree],
    *,
    max_vocab_size: int = 10000,
    min_pair_count: int = 2,
    validate_data: bool = True,
) -> tuple[
    list[str],
    list[EdgeKey],
    list[EncodedTree],
    int,
    dict[str, int],
    dict[int, str],
    list[TreeBPERule],
]:
    """Train tree-BPE rules from labelled directed trees."""

    if not isinstance(max_vocab_size, int) or max_vocab_size <= 0:
        raise ValueError("max_vocab_size must be a non-zero positive integer.")
    if not isinstance(min_pair_count, int) or min_pair_count <= 0:
        raise ValueError("min_pair_count must be a non-zero positive integer.")

    normalized = [_normalize_tree(tree, validate_data=validate_data) for tree in _as_tree_list(tree_sequence)]
    base_labels = sorted({label for _parent, _children, labels in normalized for label in labels})
    reserved_collision = [tree_bpe_token(rank) for rank in range(max_vocab_size) if tree_bpe_token(rank) in base_labels]
    if reserved_collision:
        raise ValueError(
            "input labels collide with reserved Tree BPE token labels: "
            + ", ".join(reserved_collision)
        )

    codec = _TokenCodec()
    for label in base_labels:
        codec.intern(label)
    label_dictionary = {label: codec.token_to_id[label] for label in base_labels}
    max_label_code = len(base_labels)
    label_index_dictionary = {code: label for label, code in label_dictionary.items()}
    label_index_dictionary[UNKNOWN_CODE] = UNKNOWN_LABEL

    counts: Counter[EdgeKey] = Counter()
    states = [
        _CompactEdgeTree.from_normalized(
            parent,
            children,
            _encode_base_labels(labels, label_dictionary),
            pair_counts=counts,
        )
        for parent, children, labels in normalized
    ]

    tokens: list[str] = []
    code_list: list[EdgeKey] = []
    rules: list[TreeBPERule] = []

    while len(tokens) < max_vocab_size:
        best = _select_best_pair(counts, codec, min_pair_count=min_pair_count)
        if best is None:
            break
        parent_id, child_id = best.key
        token = tree_bpe_token(len(tokens))
        if token in codec.token_to_id:
            raise ValueError(f"learned tree-BPE token {token!r} collides with an input label.")
        new_id = codec.intern(token)
        actual_events = sum(
            state.contract_pair(best.key, new_label=new_id, pair_counts=counts)
            for state in states
        )
        if actual_events == 0:
            counts.pop(best.key, None)
            continue
        rank = len(tokens)
        tokens.append(token)
        code_list.append(best.key)
        rules.append(
            TreeBPERule(
                rank=rank,
                token=token,
                parent_label=codec.decode(parent_id),
                child_label=codec.decode(child_id),
                parent_code=parent_id,
                child_code=child_id,
                code=new_id,
                count=best.count,
                actual_events=actual_events,
            )
        )

    encoded_trees = [state.to_labeled_tree() for state in states]
    return (
        tokens,
        code_list,
        encoded_trees,
        max_label_code,
        label_dictionary,
        label_index_dictionary,
        rules,
    )


@dataclass(slots=True)
class TreeBPEEncoder:
    """Reusable fitted tree-BPE encoder for new labelled trees."""

    label_dictionary: dict[str, int]
    label_index_dictionary: dict[int, str]
    tokens: list[str]
    code_list: list[EdgeKey]
    max_label_code: int
    validate_data: bool = True

    def decode_code(self, code: int) -> str:
        if code <= self.max_label_code:
            return self.label_index_dictionary.get(code, UNKNOWN_LABEL)
        return self.tokens[code - self.max_label_code - 1]

    def encode(self, tree: Tree, *, return_type: str = "trees") -> EncodedTree:
        if return_type not in ("trees", "tokens"):
            raise ValueError("return_type for TreeBPEEncoder.encode must be 'trees' or 'tokens'.")
        parent, children, labels = _normalize_tree(tree, validate_data=self.validate_data)
        state = _CompactEdgeTree.from_normalized(
            parent,
            children,
            _encode_base_labels(labels, self.label_dictionary),
        )
        new_code = self.max_label_code + 1
        for code_pair in self.code_list:
            state.contract_pair(code_pair, new_label=new_code)
            new_code += 1
        if return_type == "tokens":
            return state.to_labeled_tree(decode=self.decode_code)
        return state.to_labeled_tree()

    def encode_all(self, tree_sequence: Iterable[Tree], *, return_type: str = "trees") -> list[EncodedTree]:
        return [self.encode(tree, return_type=return_type) for tree in tree_sequence]


def tree_bpe_encode_all(
    tree_sequence: Iterable[Tree],
    code_list: list[EdgeKey],
    label_dictionary: dict[str, int],
    label_index_dictionary: dict[int, str],
    tokens: list[str],
    max_label_code: int,
    *,
    return_type: str = "trees",
    validate_data: bool = True,
) -> list[EncodedTree]:
    """Encode labelled trees with a previously learned tree-BPE code list."""

    encoder = TreeBPEEncoder(
        label_dictionary=label_dictionary,
        label_index_dictionary=label_index_dictionary,
        tokens=tokens,
        code_list=code_list,
        max_label_code=max_label_code,
        validate_data=validate_data,
    )
    return encoder.encode_all(tree_sequence, return_type=return_type)


def _build_column_dictionary(encoded_trees: Sequence[EncodedTree]) -> dict[int, int]:
    if not encoded_trees:
        return {}
    unique_codes = np.unique(np.hstack([labels for _adjacency, labels in encoded_trees]))
    return {int(code): int(index) for index, code in enumerate(unique_codes)}


def _matrix_from_encoded_trees(
    encoded_trees: Sequence[EncodedTree],
    column_label_dictionary: dict[int, int],
) -> scipy.sparse.csr_matrix:
    indices: list[int] = []
    data: list[int] = []
    indptr = [0]
    for _adjacency, labels in encoded_trees:
        for code in labels:
            column = column_label_dictionary.get(int(code))
            if column is not None:
                indices.append(column)
                data.append(1)
        indptr.append(len(indices))
    result = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(len(encoded_trees), len(column_label_dictionary)),
        dtype=np.float32,
    )
    result.sum_duplicates()
    return result


class TreeBytePairEncodingVectorizer(BaseEstimator, TransformerMixin):
    """Create vector representations of labelled trees using tree BPE.

    Parameters
    ----------
    max_vocab_size: int, optional, default=10000
        Maximum number of parent/child contraction rules to learn.
    min_pair_count: int, optional, default=2
        Minimum raw parent/child label-pair count required for a pair to be
        contracted into a learned token.
    return_type: {"matrix", "trees", "tokens"}, optional, default="matrix"
        Type of transformed data to return.
    validate_data: bool, optional, default=True
        Validate that each input is a directed labelled tree with edges ordered
        by increasing node index.

    Attributes
    ----------
    tokens_: list of str
        String labels for the learned tree-BPE tokens. The ``i``th token has
        integer code ``i + max_label_code_ + 1``.
    code_list_: list of pairs of int
        Parent/child code pairs contracted to create the learned tokens.
    max_label_code_: int
        Largest integer code assigned to an original input label.
    label_dictionary_: dict
        Mapping from original string labels to integer codes.
    label_index_dictionary_: dict
        Inverse mapping from original integer codes to string labels.
    encoder_: TreeBPEEncoder
        Reusable fitted encoder for new trees.
    rules_: list of TreeBPERule
        Human-readable learned rules.
    """

    def __init__(
        self,
        max_vocab_size: int = 10000,
        min_pair_count: int = 2,
        return_type: str = "matrix",
        validate_data: bool = True,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_pair_count = min_pair_count
        self.return_type = return_type
        self.validate_data = validate_data

    def _validate_parameters(self) -> None:
        if self.return_type not in _VALID_RETURN_TYPES:
            raise ValueError(
                "return_type must be one of 'matrix', 'trees', or 'tokens', "
                f"not {self.return_type!r}."
            )
        if not isinstance(self.max_vocab_size, int) or self.max_vocab_size <= 0:
            raise ValueError("max_vocab_size must be a non-zero positive integer.")
        if not isinstance(self.min_pair_count, int) or self.min_pair_count <= 0:
            raise ValueError("min_pair_count must be a non-zero positive integer.")
        if not isinstance(self.validate_data, bool):
            raise ValueError("validate_data must be a bool.")

    def _format_output(self, encoded_trees: list[EncodedTree]) -> Any:
        if self.return_type == "trees":
            return encoded_trees
        if self.return_type == "tokens":
            return [
                (
                    adjacency,
                    np.asarray([self.encoder_.decode_code(int(code)) for code in labels], dtype=object),
                )
                for adjacency, labels in encoded_trees
            ]
        if self.return_type == "matrix":
            return _matrix_from_encoded_trees(encoded_trees, self.column_label_dictionary_)
        raise ValueError(
            "return_type must be one of 'matrix', 'trees', or 'tokens', "
            f"not {self.return_type!r}."
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Train tree-BPE on labelled trees and return their transformed form."""

        del y, fit_params
        self._validate_parameters()
        (
            self.tokens_,
            self.code_list_,
            encoded_trees,
            self.max_label_code_,
            self.label_dictionary_,
            self.label_index_dictionary_,
            self.rules_,
        ) = tree_bpe_train(
            X,
            max_vocab_size=self.max_vocab_size,
            min_pair_count=self.min_pair_count,
            validate_data=self.validate_data,
        )
        self.encoder_ = TreeBPEEncoder(
            label_dictionary=self.label_dictionary_,
            label_index_dictionary=self.label_index_dictionary_,
            tokens=self.tokens_,
            code_list=self.code_list_,
            max_label_code=self.max_label_code_,
            validate_data=self.validate_data,
        )
        self.column_label_dictionary_ = _build_column_dictionary(encoded_trees)
        self.column_index_dictionary_ = {
            column: code for code, column in self.column_label_dictionary_.items()
        }
        return self._format_output(encoded_trees)

    def fit(self, X, y=None, **fit_params):
        """Train tree-BPE on labelled trees."""

        self.fit_transform(X, y=y, **fit_params)
        return self

    def transform(self, X, y=None):
        """Transform labelled trees using the learned tree-BPE rules."""

        del y
        check_is_fitted(
            self,
            [
                "tokens_",
                "code_list_",
                "max_label_code_",
                "label_dictionary_",
                "label_index_dictionary_",
                "encoder_",
                "column_label_dictionary_",
            ],
        )
        self._validate_parameters()
        encoded_trees = self.encoder_.encode_all(_as_tree_list(X), return_type="trees")
        return self._format_output(encoded_trees)
