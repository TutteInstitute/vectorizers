import numpy as np
import pytest
import scipy.sparse

from vectorizers.tree_bpe import TreeBytePairEncodingVectorizer, tree_bpe_token


def path_tree(labels):
    n_nodes = len(labels)
    rows = np.arange(n_nodes - 1)
    cols = np.arange(1, n_nodes)
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(n_nodes - 1), (rows, cols)), shape=(n_nodes, n_nodes)
    )
    return adjacency, np.asarray(labels, dtype=object)


def edge_tree(n_nodes, edges, labels):
    rows = [parent for parent, child in edges]
    cols = [child for parent, child in edges]
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(edges)), (rows, cols)), shape=(n_nodes, n_nodes)
    )
    return adjacency, np.asarray(labels, dtype=object)


def test_tree_bpe_learns_expected_edge_rule():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=2,
        return_type="trees",
    )

    transformed = vectorizer.fit_transform(X)

    assert vectorizer.rules_[0].parent_label == "A"
    assert vectorizer.rules_[0].child_label == "B"
    assert vectorizer.rules_[0].count == 2
    assert vectorizer.rules_[0].actual_events == 2
    assert vectorizer.code_list_ == [(1, 2)]
    assert vectorizer.tokens_ == [tree_bpe_token(0)]
    assert transformed[0][1].tolist() == [vectorizer.max_label_code_ + 1, 3]


def test_tree_bpe_matrix_fit_transform_matches_transform():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=2,
        min_pair_count=1,
        return_type="matrix",
    )

    fit_transformed = vectorizer.fit_transform(X)
    transformed = vectorizer.transform(X)

    assert (fit_transformed != transformed).nnz == 0


def test_tree_bpe_return_type_trees():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=2,
        return_type="trees",
    )

    encoded = vectorizer.fit_transform(X)
    adjacency, labels = encoded[0]

    assert scipy.sparse.isspmatrix_csr(adjacency)
    assert adjacency.shape == (2, 2)
    assert labels.tolist() == [vectorizer.max_label_code_ + 1, 3]


def test_tree_bpe_return_type_tokens():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=2,
        return_type="tokens",
    )

    encoded = vectorizer.fit_transform(X)
    adjacency, labels = encoded[0]

    assert scipy.sparse.isspmatrix_csr(adjacency)
    assert labels.tolist() == [tree_bpe_token(0), "C"]


def test_tree_bpe_unknown_label_maps_to_unknown_token():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=2,
        return_type="tokens",
    ).fit(X)

    transformed = vectorizer.transform([path_tree(["A", "B", "E"])])

    assert transformed[0][1].tolist() == [tree_bpe_token(0), "<UNK>"]


def test_tree_bpe_encoder_object_encodes_new_trees():
    X = [path_tree(["A", "B", "C"]), path_tree(["A", "B", "D"])]
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=2,
    ).fit(X)

    adjacency, labels = vectorizer.encoder_.encode(path_tree(["A", "B", "C"]))

    assert scipy.sparse.isspmatrix_csr(adjacency)
    assert labels.tolist() == [vectorizer.max_label_code_ + 1, 3]


def test_tree_bpe_single_node_tree():
    vectorizer = TreeBytePairEncodingVectorizer(return_type="matrix")

    result = vectorizer.fit_transform([path_tree(["ROOT"])])

    assert result.shape == (1, 1)
    assert result.toarray().tolist() == [[1.0]]
    assert vectorizer.tokens_ == []


def test_tree_bpe_counts_raw_edges_before_overlap_filtering():
    tree = edge_tree(9, [(0, child) for child in range(1, 9)], ["A"] + ["B"] * 8)
    vectorizer = TreeBytePairEncodingVectorizer(
        max_vocab_size=1,
        min_pair_count=1,
        return_type="trees",
    ).fit([tree])

    assert vectorizer.rules_[0].count == 8
    assert vectorizer.rules_[0].actual_events == 1


def test_tree_bpe_rejects_non_increasing_edges():
    bad_tree = edge_tree(3, [(0, 2), (2, 1)], ["A", "B", "C"])
    vectorizer = TreeBytePairEncodingVectorizer()

    with pytest.raises(ValueError, match="lower to higher"):
        vectorizer.fit_transform([bad_tree])


def test_tree_bpe_rejects_non_tree():
    bad_tree = edge_tree(3, [(0, 1)], ["A", "B", "C"])
    vectorizer = TreeBytePairEncodingVectorizer()

    with pytest.raises(ValueError, match="n - 1"):
        vectorizer.fit_transform([bad_tree])


def test_tree_bpe_bad_parameters():
    with pytest.raises(ValueError, match="return_type"):
        TreeBytePairEncodingVectorizer(return_type="bad").fit([path_tree(["A"])])
    with pytest.raises(ValueError, match="max_vocab_size"):
        TreeBytePairEncodingVectorizer(max_vocab_size=0).fit([path_tree(["A"])])
    with pytest.raises(ValueError, match="min_pair_count"):
        TreeBytePairEncodingVectorizer(min_pair_count=0).fit([path_tree(["A"])])
