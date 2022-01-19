import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import normalize

import scipy.sparse
import numpy as np
import pandas as pd

from vectorizers import TokenCooccurrenceVectorizer
from vectorizers import NgramVectorizer
from vectorizers import SkipgramVectorizer
from vectorizers import DistributionVectorizer
from vectorizers import HistogramVectorizer
from vectorizers import KDEVectorizer
from vectorizers import LabelledTreeCooccurrenceVectorizer
# Deprecated transformers
# from vectorizers.transformers import SequentialDifferenceTransformer
# from vectorizers.transformers import Wasserstein1DHistogramTransformer
from vectorizers import WassersteinVectorizer
from vectorizers import ApproximateWassersteinVectorizer
from vectorizers import SinkhornVectorizer
from vectorizers import LZCompressionVectorizer, BytePairEncodingVectorizer

from vectorizers.distances import kantorovich1d
from vectorizers.ngram_vectorizer import ngrams_of
from vectorizers._vectorizers import find_bin_boundaries
from vectorizers.tree_token_cooccurrence import (
    build_tree_skip_grams,
    sequence_tree_skip_grams,
)
from vectorizers.preprocessing import remove_node
from vectorizers._window_kernels import (
    harmonic_kernel,
    flat_kernel,
)
from vectorizers.utils import summarize_embedding, categorical_columns_to_list
from vectorizers.mixed_gram_vectorizer import to_unicode

token_data = (
    (1, 3, 1, 4, 2),
    (2, 1, 2, 3, 4, 1, 2, 1, 3, 2, 4),
    (4, 1, 1, 3, 2, 4, 2),
    (1, 2, 2, 1, 2, 1, 3, 4, 3, 2, 4),
    (3, 4, 2, 1, 3, 1, 4, 4, 1, 3, 2),
    (2, 1, 3, 1, 4, 4, 1, 4, 1, 3, 2, 4),
)

text_token_data = (
    ("foo", "pok", "foo", "wer", "bar"),
    (),
    ("bar", "foo", "bar", "pok", "wer", "foo", "bar", "foo", "pok", "bar", "wer"),
    ("wer", "foo", "foo", "pok", "bar", "wer", "bar"),
    ("foo", "bar", "bar", "foo", "bar", "foo", "pok", "wer", "pok", "bar", "wer"),
    ("pok", "wer", "bar", "foo", "pok", "foo", "wer", "wer", "foo", "pok", "bar"),
    (
        "bar",
        "foo",
        "pok",
        "foo",
        "wer",
        "wer",
        "foo",
        "wer",
        "foo",
        "pok",
        "bar",
        "wer",
    ),
)

text_token_data_ngram = (
    ("wer", "pok", "wer"),
    ("bar", "pok", "wer"),
    ("foo", "pok", "wer"),
)
text_token_data_permutation = (("wer", "pok"), ("bar", "pok"), ("foo", "pok", "wer"))
text_token_data_subset = (("foo", "pok"), ("pok", "foo", "foo"))
text_token_data_new_token = (("foo", "pok"), ("pok", "foo", "foo", "zaz"))
text_token_data_ngram_soln = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


mixed_token_data = (
    (1, "pok", 1, 3.1415, "bar"),
    ("bar", 1, "bar", "pok", 3.1415, 1, "bar", 1, "pok", "bar", 3.1415),
    (3.1415, 1, 1, "pok", "bar", 3.1415, "bar"),
    (1, "bar", "bar", 1, "bar", 1, "pok", 3.1415, "pok", "bar", 3.1415),
    ("pok", 3.1415, "bar", 1, "pok", 1, 3.1415, 3.1415, 1, "pok", "bar"),
    ("bar", 1, "pok", 1, 3.1415, 3.1415, 1, 3.1415, 1, "pok", "bar", 3.1415),
)

point_data = [
    np.random.multivariate_normal(
        mean=[0.0, 0.0], cov=[[0.5, 0.0], [0.0, 0.5]], size=50
    ),
    np.random.multivariate_normal(
        mean=[0.5, 0.0], cov=[[0.5, 0.0], [0.0, 0.5]], size=60
    ),
    np.random.multivariate_normal(
        mean=[-0.5, 0.0], cov=[[0.5, 0.0], [0.0, 0.5]], size=80
    ),
    np.random.multivariate_normal(
        mean=[0.0, 0.5], cov=[[0.5, 0.0], [0.0, 0.5]], size=40
    ),
    np.random.multivariate_normal(
        mean=[0.0, -0.5], cov=[[0.5, 0.0], [0.0, 0.5]], size=20
    ),
]

value_sequence_data = [
    np.random.poisson(3.0, size=100),
    np.random.poisson(12.0, size=30),
    np.random.poisson(4.0, size=40),
    np.random.poisson(5.0, size=90),
    np.random.poisson(4.5, size=120),
    np.random.poisson(9.0, size=60),
    np.random.poisson(2.0, size=80),
]

path_graph = scipy.sparse.csr_matrix(
    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
)
path_graph_labels = np.array(["a", "b", "a", "c"])
path_graph_two_out = scipy.sparse.csr_matrix(
    [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
)
unique_labels = np.array(["a", "b", "c", "d"])
shifted_labels = np.array(["b", "c", "d", "e"])
tree_sequence = [(path_graph, unique_labels), (path_graph, shifted_labels)]
label_dictionary = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
sub_dictionary = {"a": 0, "b": 1, "c": 2}

seq_tree_sequence = [
    (scipy.sparse.csr_matrix([[0, 1], [0, 0]]), np.array(["wer", "pok"])),
    (scipy.sparse.csr_matrix([[0, 1], [0, 0]]), np.array(["bar", "pok"])),
    (
        scipy.sparse.csr_matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        np.array(["foo", "pok", "wer"]),
    ),
]

distributions_data = scipy.sparse.rand(
    100, 1000, format="csr", random_state=42, dtype=np.float64
)
vectors_data = np.random.normal(size=(1000, 150))

distributions_data_list = [
    np.array(x) for x in distributions_data.tolil().data
]
vectors_data_list = [
    np.ascontiguousarray(vectors_data[indices]) for indices in distributions_data.tolil().rows
]
generator_reference_dist = np.full(16, 1.0 / 16.0)
generator_reference_vectors = (
        np.mean(distributions_data.toarray(), axis=0) @ vectors_data
) + np.random.normal(scale=0.25 * np.mean(np.abs(vectors_data)), size=(16, 150))


raw_string_data = [
    "asdfj;afoosdaflksapokwerfoobarpokwersdfsadfsadfnbkajyfoopokwer",
    "pokfoo;ohnASDbarfoobarpoksdf sgn;asregtjpoksdfpokpokwer",
    "werqweoijsdcasdfpoktrfoobarpokqwernasdfasdpokpokpok",
    "pokwerpokwqerpokwersadfpokqwepokwerpokpok",
    "foobarfoofooasdfsdfgasdffoobarbazcabfoobarbarbazfoobaz",
    "pokfoopokbarpokwerpokbazgfniusnvbgasgbabgsadfjnkr[pko",
]



def test_LabeledTreeCooccurrenceVectorizer():
    model = LabelledTreeCooccurrenceVectorizer(
        window_radius=2, window_orientation="after"
    )
    result = model.fit_transform(tree_sequence)
    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
    )
    assert np.allclose(result.toarray(), expected_result.toarray())

    result = model.transform(tree_sequence)
    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
    )
    assert np.allclose(result.toarray(), expected_result.toarray())


def test_LabeledTreeCooccurrenceVectorizer_reduced_vocab():
    model = LabelledTreeCooccurrenceVectorizer(
        window_radius=2,
        window_orientation="after",
        token_dictionary=sub_dictionary,
    )
    result = model.fit_transform(tree_sequence)
    assert result.shape == (3, 3)


@pytest.mark.parametrize("min_token_occurrences", [None, 2])
@pytest.mark.parametrize("max_token_occurrences", [None, 2])
@pytest.mark.parametrize("min_document_occurrences", [None, 1])
@pytest.mark.parametrize("max_document_frequency", [None, 0.7])
@pytest.mark.parametrize("window_orientation", ["before", "after", "directional"])
@pytest.mark.parametrize("window_radius", [1, 2])
@pytest.mark.parametrize("kernel_function", ["harmonic", "flat", "geometric"])
@pytest.mark.parametrize("mask_string", [None, "[MASK]"])
def test_equality_of_Tree_and_Token_CooccurrenceVectorizers(
    min_token_occurrences,
    max_token_occurrences,
    min_document_occurrences,
    max_document_frequency,
    window_radius,
    window_orientation,
    kernel_function,
    mask_string,
):
    tree_model = LabelledTreeCooccurrenceVectorizer(
        window_radius=window_radius,
        window_orientation=window_orientation,
        kernel_function=kernel_function,
        min_occurrences=min_token_occurrences,
        max_occurrences=max_token_occurrences,
        max_tree_frequency=max_document_frequency,
        min_tree_occurrences=min_document_occurrences,
        mask_string=mask_string,
    )
    seq_model = TokenCooccurrenceVectorizer(
        window_radii=window_radius,
        window_orientations=window_orientation,
        kernel_functions=kernel_function,
        min_occurrences=min_token_occurrences,
        max_occurrences=max_token_occurrences,
        max_document_frequency=max_document_frequency,
        min_document_occurrences=min_document_occurrences,
        mask_string=mask_string,
        normalize_windows=False,
    )
    assert np.allclose(
        tree_model.fit_transform(seq_tree_sequence).toarray(),
        seq_model.fit_transform(text_token_data_permutation).toarray(),
    )
    assert np.allclose(
        tree_model.fit_transform(seq_tree_sequence).toarray(),
        tree_model.transform(seq_tree_sequence).toarray(),
    )
    assert np.allclose(
        seq_model.fit_transform(text_token_data_permutation).toarray(),
        seq_model.transform(text_token_data_permutation).toarray(),
    )
    assert np.allclose(
        tree_model.transform(seq_tree_sequence).toarray(),
        seq_model.transform(text_token_data_permutation).toarray(),
    )


@pytest.mark.parametrize("min_token_occurrences", [None, 2])
@pytest.mark.parametrize("max_document_frequency", [None, 0.7])
@pytest.mark.parametrize("window_orientation", ["directional"])
@pytest.mark.parametrize("window_radius", [1, 2])
@pytest.mark.parametrize("n_iter", [0, 2])
@pytest.mark.parametrize("kernel_function", ["flat"])
@pytest.mark.parametrize("window_function", ["fixed"])
@pytest.mark.parametrize("mask_string", [None, "[MASK]"])
@pytest.mark.parametrize("nullify_mask", [False, True])
@pytest.mark.parametrize("normalize_windows", [False, True])
@pytest.mark.parametrize("normalization", ["bayesian", "frequentist"])
def test_equality_of_TokenCooccurrenceVectorizer(
    min_token_occurrences,
    max_document_frequency,
    window_radius,
    window_orientation,
    kernel_function,
    window_function,
    mask_string,
    nullify_mask,
    n_iter,
    normalize_windows,
    normalization,
):
    model1 = TokenCooccurrenceVectorizer(
        window_radii=[window_radius],
        kernel_functions=[kernel_function],
        window_functions=[window_function],
        min_occurrences=min_token_occurrences,
        max_document_frequency=max_document_frequency,
        mask_string=mask_string,
        n_iter=n_iter,
        nullify_mask=nullify_mask and mask_string is not None,
        normalize_windows=normalize_windows,
        normalization=normalization,
    )
    model2 = TokenCooccurrenceVectorizer(
        window_radii=window_radius,
        kernel_functions=kernel_function,
        window_functions=window_function,
        min_occurrences=min_token_occurrences,
        max_document_frequency=max_document_frequency,
        mask_string=mask_string,
        n_iter=n_iter,
        nullify_mask=nullify_mask and mask_string is not None,
        normalize_windows=normalize_windows,
        normalization=normalization,
    )
    assert np.allclose(
        model1.fit_transform(text_token_data_permutation).toarray(),
        model2.fit_transform(text_token_data_permutation).toarray(),
    )
    assert np.allclose(
        model1.fit_transform(text_token_data_permutation).toarray(),
        model1.transform(text_token_data_permutation).toarray(),
    )
    assert np.allclose(
        model2.fit_transform(text_token_data_permutation).toarray(),
        model2.transform(text_token_data_permutation).toarray(),
    )


def test_reverse_cooccurrence_vectorizer():
    seq_model1 = TokenCooccurrenceVectorizer(
        window_radii=2,
        window_orientations="after",
        kernel_functions="harmonic",
        mask_string=None,
        normalize_windows=False,
    )
    seq_model2 = TokenCooccurrenceVectorizer(
        window_radii=2,
        window_orientations="before",
        kernel_functions="harmonic",
        mask_string=None,
        normalize_windows=False,
    )
    reversed_after = (seq_model1.fit_transform(text_token_data).toarray().T,)
    before = (seq_model2.fit_transform(text_token_data).toarray(),)
    assert np.allclose(reversed_after, before)


def test_build_tree_skip_grams_contract():
    (result_matrix, result_labels) = build_tree_skip_grams(
        token_sequence=path_graph_labels,
        adjacency_matrix=path_graph,
        kernel_function=flat_kernel,
        kernel_args=dict(),
        window_size=2,
    )
    expected_result = scipy.sparse.csr_matrix(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    assert np.allclose(result_matrix.toarray(), expected_result.toarray())


def test_build_tree_skip_grams_no_contract():
    (result_matrix, result_labels) = build_tree_skip_grams(
        token_sequence=unique_labels,
        adjacency_matrix=path_graph,
        kernel_function=flat_kernel,
        kernel_args=dict([]),
        window_size=2,
    )
    assert np.allclose(result_matrix.toarray(), path_graph_two_out.toarray())
    assert np.array_equal(unique_labels, result_labels)


@pytest.mark.parametrize(
    "window_orientation", ["before", "after", "symmetric", "directional"]
)
def test_sequence_tree_skip_grams(window_orientation):
    result = sequence_tree_skip_grams(
        tree_sequence,
        kernel_function=flat_kernel,
        kernel_args=dict(),
        window_size=2,
        label_dictionary=label_dictionary,
        window_orientation=window_orientation,
    )
    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 2, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
    )
    if window_orientation == "before":
        assert np.allclose(result.toarray(), expected_result.T.toarray())
    elif window_orientation == "after":
        assert np.allclose(result.toarray(), expected_result.toarray())
    elif window_orientation == "symmetric":
        assert np.allclose(
            result.toarray(), (expected_result + expected_result.T).toarray()
        )
    elif window_orientation == "directional":
        assert np.allclose(
            result.toarray(),
            scipy.sparse.hstack([expected_result.T, expected_result]).toarray(),
        )


def test_harmonic_kernel():
    kernel = harmonic_kernel([0, 0, 0, 0])
    assert kernel[0] == 1.0
    assert kernel[-1] == 1.0 / 4.0
    assert kernel[1] == 1.0 / 2.0


def test_flat_kernel():
    kernel = flat_kernel([0] * np.random.randint(2, 10))
    assert np.all(kernel == 1.0)


def test_ngrams_of():
    for ngram_size in (1, 2, 4):
        tokens = np.random.randint(10, size=np.random.poisson(5 + ngram_size))
        ngrams = ngrams_of(tokens, ngram_size)
        if len(tokens) >= ngram_size:
            assert len(ngrams) == len(tokens) - (ngram_size - 1)
        else:
            assert len(ngrams) == 0
        assert np.all(
            [ngrams[i][0] == tokens[i] for i in range(len(tokens) - (ngram_size - 1))]
        )
        assert np.all(
            [
                ngrams[i][-1] == tokens[i + (ngram_size - 1)]
                for i in range(len(tokens) - (ngram_size - 1))
            ]
        )


def test_find_bin_boundaries_min():
    data = np.random.poisson(5, size=1000)
    data = np.append(data, [0, 0, 0])
    bins = find_bin_boundaries(data, 10)
    # Poisson so smallest bin should be at 0
    assert bins[0] == 0.0


def test_find_boundaries_all_dupes():
    data = np.ones(100)
    with pytest.warns(UserWarning):
        bins = find_bin_boundaries(data, 10)
        assert len(bins) == 1


def test_token_cooccurrence_vectorizer_basic():
    vectorizer = TokenCooccurrenceVectorizer(window_orientations="directional")
    result = vectorizer.fit_transform(token_data)
    transform = vectorizer.transform(token_data)
    assert (result != transform).nnz == 0
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after"
    )
    result = vectorizer.fit_transform(token_data)
    transform = vectorizer.transform(token_data)
    assert (result != transform).nnz == 0
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_ngrams():
    vectorizer = TokenCooccurrenceVectorizer(n_iter=0, skip_ngram_size=2)
    result = vectorizer.fit_transform(text_token_data_ngram)
    transform = vectorizer.transform(text_token_data_ngram)
    assert (result != transform).nnz == 0
    assert np.allclose(result.toarray(), text_token_data_ngram_soln)


def test_token_cooccurrence_vectorizer_window_normalization():
    vectorizer = TokenCooccurrenceVectorizer(
        n_iter=1, normalize_windows=True, window_normalization="bayesian"
    )
    result = vectorizer.fit_transform(token_data)
    transform = vectorizer.transform(token_data)
    assert (result != transform).nnz == 0

    vectorizer = TokenCooccurrenceVectorizer(
        n_iter=1, normalize_windows=True, window_normalization="frequentist"
    )
    result = vectorizer.fit_transform(token_data)
    transform = vectorizer.transform(token_data)
    assert (result != transform).nnz == 0


def test_token_cooccurrence_vectorizer_window_args():
    vectorizer_a = TokenCooccurrenceVectorizer(window_functions="variable")
    vectorizer_b = TokenCooccurrenceVectorizer(
        window_functions="variable", window_args={"power": 0.75}
    )
    assert (
        vectorizer_a.fit_transform(token_data) != vectorizer_b.fit_transform(token_data)
    ).nnz == 0


def test_token_cooccurrence_vectorizer_kernel_args():
    vectorizer_a = TokenCooccurrenceVectorizer(
        kernel_functions="geometric",
        mask_string="MASK",
        kernel_args={"normalize": True},
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        kernel_functions="geometric",
        kernel_args={"normalize": True, "p": 0.9},
        mask_string="MASK",
    )
    assert (
        vectorizer_a.fit_transform(token_data) != vectorizer_b.fit_transform(token_data)
    ).nnz == 0


def test_cooccurrence_vectorizer_epsilon():
    vectorizer_a = TokenCooccurrenceVectorizer(epsilon=0, normalization="frequentist")
    vectorizer_b = TokenCooccurrenceVectorizer(
        epsilon=1e-11, normalization="frequentist"
    )
    vectorizer_c = TokenCooccurrenceVectorizer(epsilon=1, normalization="frequentist")
    mat1 = normalize(
        vectorizer_a.fit_transform(token_data).toarray(), axis=0, norm="l1"
    )
    mat2 = vectorizer_b.fit_transform(token_data).toarray()
    assert np.allclose(mat1, mat2)
    assert vectorizer_c.fit_transform(token_data).nnz == 0


def test_cooccurrence_vectorizer_coo_mem():
    vectorizer_a = TokenCooccurrenceVectorizer(
        window_functions="fixed",
        n_iter=0,
        coo_max_memory="2k",
        normalize_windows=False,
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        window_functions="fixed",
        n_iter=0,
        normalize_windows=False,
    )

    mat1 = vectorizer_a.fit_transform(token_data).toarray()
    mat2 = vectorizer_b.fit_transform(token_data).toarray()
    assert np.allclose(mat1, mat2)


def test_cooccurrence_vectorizer_coo_mem_limit():
    vectorizer_a = TokenCooccurrenceVectorizer(
        window_functions="fixed",
        n_iter=0,
        coo_max_memory="1k",
        normalize_windows=False,
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        window_functions="fixed",
        n_iter=0,
        normalize_windows=False,
    )
    np.random.seed(42)
    data = [[np.random.randint(0, 10) for i in range(100)]]
    mat1 = vectorizer_a.fit_transform(data).toarray()
    mat2 = vectorizer_b.fit_transform(data).toarray()
    assert np.allclose(mat1, mat2)


@pytest.mark.parametrize("skip_grams_size", [1, 2])
def test_cooccurrence_vectorizer_em_iter(skip_grams_size):
    vectorizer_a = TokenCooccurrenceVectorizer(
        n_iter=0, skip_ngram_size=skip_grams_size
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        n_iter=2, skip_ngram_size=skip_grams_size, normalization="frequentist"
    )
    assert (
        vectorizer_a.fit_transform(token_data).nnz
        == vectorizer_b.fit_transform(token_data).nnz
    )


def test_cooccurrence_vectorizer_wide_iter():
    vectorizer_a = TokenCooccurrenceVectorizer(
        kernel_functions=["flat", "harmonic"],
        window_radii=[1, 2],
        mix_weights=[1, 1],
        window_functions=("fixed", "variable"),
        window_orientations=["directional", "directional"],
        n_iter=0,
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        kernel_functions=["flat", "harmonic"],
        window_radii=[1, 2],
        mix_weights=[1, 1],
        window_functions=["fixed", "variable"],
        window_orientations=["directional", "directional"],
        normalization="frequentist",
        n_iter=2,
    )
    assert (
        vectorizer_a.fit_transform(token_data).nnz
        == vectorizer_b.fit_transform(token_data).nnz
    )


def test_cooccurrence_vectorizer_wide_transform():
    vectorizer_a = TokenCooccurrenceVectorizer(
        kernel_functions=["flat", "harmonic"],
        window_radii=[1, 2],
        mix_weights=[1, 1],
        window_functions=("fixed", "variable"),
        window_orientations=["directional", "directional"],
        n_iter=2,
    )
    assert (
        vectorizer_a.fit_transform(token_data).nnz
        == vectorizer_a.transform(token_data).nnz
    )


@pytest.mark.parametrize("kernel_function", ["harmonic", "flat", "geometric"])
def test_token_cooccurrence_vectorizer_offset(kernel_function):
    vectorizer_a = TokenCooccurrenceVectorizer(
        kernel_functions=kernel_function, window_radii=1, normalize_windows=False
    )
    vectorizer_b = TokenCooccurrenceVectorizer(
        kernel_functions=kernel_function, window_radii=2, normalize_windows=False
    )
    vectorizer_c = TokenCooccurrenceVectorizer(
        window_radii=2,
        kernel_functions=kernel_function,
        kernel_args={"offset": 1},
        normalize_windows=False,
    )
    mat1 = (
        vectorizer_a.fit_transform(token_data) + vectorizer_c.fit_transform(token_data)
    ).toarray()
    mat2 = vectorizer_b.fit_transform(token_data).toarray()
    assert np.allclose(mat1, mat2)


def test_token_cooccurrence_vectorizer_nullify_mask():
    vectorizer_a = TokenCooccurrenceVectorizer(mask_string="MASK", nullify_mask=True)
    vectorizer_b = TokenCooccurrenceVectorizer(
        mask_string="MASK",
    )
    assert np.allclose(
        vectorizer_a.fit_transform(token_data).toarray()[:-1, :-1],
        vectorizer_b.fit_transform(token_data).toarray()[:-1, :-1],
    )
    assert vectorizer_a.fit_transform(token_data).getrow(-1).nnz == 0
    assert vectorizer_a.fit_transform(token_data).getcol(-1).nnz == 0


def test_token_cooccurrence_vectorizer_orientation():
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="directional", normalize_windows=False
    )
    result = vectorizer.fit_transform(text_token_data)
    assert result.shape == (4, 8)
    # Check the pok preceded by wer value is 1
    row = vectorizer.token_label_dictionary_["pok"]
    col = vectorizer.column_label_dictionary_["pre_0_wer"]
    assert result[row, col] == 1
    result_before = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="before", normalize_windows=False
    ).fit_transform(text_token_data)
    result_after = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    ).fit_transform(text_token_data)
    assert np.all(result_after.toarray() == (result_before.transpose()).toarray())
    assert np.all(
        result.toarray() == np.hstack([result_before.toarray(), result_after.toarray()])
    )


def test_token_cooccurrence_vectorizer_column_order():
    vectorizer = TokenCooccurrenceVectorizer().fit(text_token_data)
    vectorizer_permuted = TokenCooccurrenceVectorizer().fit(text_token_data_permutation)
    assert (
        vectorizer.token_label_dictionary_
        == vectorizer_permuted.token_label_dictionary_
    )


def test_token_cooccurrence_vectorizer_transform():
    vectorizer = TokenCooccurrenceVectorizer(
        window_orientations="directional", normalize_windows=False
    )
    result = vectorizer.fit_transform(text_token_data_subset)
    transform = vectorizer.transform(text_token_data)
    assert result.shape == transform.shape
    assert transform[0, 0] == 17


def test_token_cooccurence_vectorizer_transform_new_vocab():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(text_token_data_subset)
    transform = vectorizer.transform(text_token_data_new_token)
    assert (result != transform).nnz == 0


def test_token_cooccurrence_vectorizer_text():
    vectorizer = TokenCooccurrenceVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)
    transform = vectorizer.transform(text_token_data)
    assert (result != transform).nnz == 0
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    )
    result = vectorizer.fit_transform(text_token_data)
    transform = vectorizer.transform(text_token_data)
    assert (result != transform).nnz == 0
    assert result[1, 2] == 8
    assert result[0, 1] == 6


def test_token_cooccurrence_vectorizer_fixed_tokens():
    vectorizer = TokenCooccurrenceVectorizer(token_dictionary={1: 0, 2: 1, 3: 2})
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    )
    result = vectorizer.fit_transform(token_data)
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_excessive_prune():
    vectorizer = TokenCooccurrenceVectorizer(min_frequency=1.0)
    with pytest.raises(ValueError):
        result = vectorizer.fit_transform(token_data)


def test_token_cooccurrence_vectorizer_min_occur():
    vectorizer = TokenCooccurrenceVectorizer(min_occurrences=3)
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    )
    result = vectorizer.fit_transform(token_data)
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_max_freq():
    vectorizer = TokenCooccurrenceVectorizer(max_frequency=0.2)
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    )
    result = vectorizer.fit_transform(token_data)
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_variable_window():
    vectorizer = TokenCooccurrenceVectorizer(window_functions="variable")
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    vectorizer = TokenCooccurrenceVectorizer(
        window_radii=1, window_orientations="after", normalize_windows=False
    )
    result = vectorizer.fit_transform(token_data)
    assert result[0, 2] == 8
    assert result[1, 0] == 6


def test_token_cooccurrence_vectorizer_mixed():
    vectorizer = TokenCooccurrenceVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit_transform(mixed_token_data)


def test_ngram_vectorizer_basic():
    vectorizer = NgramVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    transform_result = vectorizer.transform(token_data)
    assert np.all(transform_result.data == result.data)
    assert np.all(transform_result.tocoo().col == result.tocoo().col)


def test_ngram_vectorizer_text():
    vectorizer = NgramVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)
    # Ensure that the empty document has an all zero row
    assert len((result[1, :]).data) == 0


def test_ngram_vectorizer_mixed():
    vectorizer = SkipgramVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit_transform(mixed_token_data)


def test_ngram_vectorizer_min_doc():
    vectorizer = NgramVectorizer(min_document_occurrences=2)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 2)
    assert np.all(count_matrix.toarray() == np.array([[1, 1], [1, 0], [1, 1]]))


def test_ngram_vectorizer_min_doc_freq():
    vectorizer = NgramVectorizer(min_document_frequency=0.6)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 2)
    assert np.all(count_matrix.toarray() == np.array([[1, 1], [1, 0], [1, 1]]))


def test_ngram_vectorizer_max_doc():
    vectorizer = NgramVectorizer(max_document_occurrences=1)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 2)
    assert np.all(count_matrix.toarray() == np.array([[0, 0], [1, 0], [0, 1]]))


def test_ngram_vectorizer_max_doc_freq():
    vectorizer = NgramVectorizer(max_document_frequency=0.4)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 2)
    assert np.all(count_matrix.toarray() == np.array([[0, 0], [1, 0], [0, 1]]))


def test_skipgram_vectorizer_basic():
    vectorizer = SkipgramVectorizer()
    result = vectorizer.fit_transform(token_data)
    assert scipy.sparse.issparse(result)
    transform_result = vectorizer.transform(token_data)
    assert np.all(transform_result.data == result.data)
    assert np.all(transform_result.tocoo().col == result.tocoo().col)


def test_skipram_vectorizer_max_doc():
    vectorizer = SkipgramVectorizer(max_document_occurrences=2)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 1)
    assert np.all(count_matrix.toarray() == np.array([[0], [0], [1]]))


def test_skipram_vectorizer_min_doc():
    vectorizer = SkipgramVectorizer(min_document_occurrences=2)
    count_matrix = vectorizer.fit_transform(text_token_data_permutation)
    assert count_matrix.shape == (3, 2)
    assert np.all(count_matrix.toarray() == np.array([[0, 1], [0, 0], [1, 0]]))


def test_skipgram_vectorizer_text():
    vectorizer = SkipgramVectorizer()
    result = vectorizer.fit_transform(text_token_data)
    assert scipy.sparse.issparse(result)
    # Ensure that the empty document has an all zero row
    assert len((result[1, :]).data) == 0


def test_skipgram_vectorizer_mixed():
    vectorizer = SkipgramVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit_transform(mixed_token_data)


def test_distribution_vectorizer_basic():
    vectorizer = DistributionVectorizer(n_components=3)
    result = vectorizer.fit_transform(point_data)
    assert result.shape == (len(point_data), 3)
    transform_result = vectorizer.transform(point_data)
    assert np.all(result == transform_result)


def test_distribution_vectorizer_bad_params():
    vectorizer = DistributionVectorizer(n_components=-1)
    with pytest.raises(ValueError):
        vectorizer.fit(point_data)
    vectorizer = DistributionVectorizer(n_components="foo")
    with pytest.raises(ValueError):
        vectorizer.fit(point_data)
    vectorizer = DistributionVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit(point_data[0])
    vectorizer = DistributionVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit(
            [np.random.uniform(size=(10, np.random.poisson(10))) for i in range(5)]
        )
    vectorizer = DistributionVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit(
            [
                [[1, 2, 3], [1, 2], [1, 2, 3, 4]],
                [
                    [1, 2],
                    [
                        1,
                    ],
                    [1, 2, 3],
                ],
            ]
        )


def test_histogram_vectorizer_basic():
    vectorizer = HistogramVectorizer(n_components=20)
    result = vectorizer.fit_transform(value_sequence_data)
    assert result.shape == (len(value_sequence_data), 20)
    transform_result = vectorizer.transform(value_sequence_data)
    assert np.all(result == transform_result)


def test_histogram_vectorizer_outlier_bins():
    vectorizer = HistogramVectorizer(n_components=20, append_outlier_bins=True)
    result = vectorizer.fit_transform(value_sequence_data)
    assert result.shape == (len(value_sequence_data), 20 + 2)
    transform_result = vectorizer.transform([[-1.0, -1.0, -1.0, 150.0]])
    assert transform_result[0][0] == 3.0
    assert transform_result[0][-1] == 1.0


def test_kde_vectorizer_basic():
    vectorizer = KDEVectorizer(n_components=20)
    result = vectorizer.fit_transform(value_sequence_data)
    assert result.shape == (len(value_sequence_data), 20)
    transform_result = vectorizer.transform(value_sequence_data)
    assert np.all(result == transform_result)

# Deprecated transformers; tests to be removed
# def test_seq_diff_transformer():
#     transformer = SequentialDifferenceTransformer()
#     result = transformer.fit_transform(value_sequence_data)
#     for i in range(len(value_sequence_data)):
#         assert np.allclose(
#             result[i], value_sequence_data[i][1:] - value_sequence_data[i][:-1]
#         )
#
#
# def test_wass1d_transfomer():
#     vectorizer = HistogramVectorizer()
#     histogram_data = vectorizer.fit_transform(value_sequence_data)
#     transformer = Wasserstein1DHistogramTransformer()
#     result = transformer.fit_transform(histogram_data)
#     for i in range(result.shape[0]):
#         for j in range(i + 1, result.shape[0]):
#             assert np.isclose(
#                 kantorovich1d(histogram_data[i], histogram_data[j]),
#                 np.sum(np.abs(result[i] - result[j])),
#             )


def test_wasserstein_vectorizer_basic():
    vectorizer = WassersteinVectorizer(random_state=42)
    result = vectorizer.fit_transform(distributions_data, vectors=vectors_data)
    transform_result = vectorizer.transform(distributions_data, vectors=vectors_data)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)

def test_wasserstein_vectorizer_lists():
    vectorizer = WassersteinVectorizer(random_state=42)
    result = vectorizer.fit_transform(distributions_data_list, vectors=vectors_data_list)
    transform_result = vectorizer.transform(distributions_data_list, vectors=vectors_data_list)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)

def test_wasserstein_vectorizer_generators():
    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    vectorizer = WassersteinVectorizer(random_state=42)
    result = vectorizer.fit_transform(
        distributions_data_generator,
        vectors=vectors_data_generator,
        reference_distribution=generator_reference_dist,
        reference_vectors=generator_reference_vectors,
        n_distributions=distributions_data.shape[0],
        vector_dim=vectors_data.shape[1],
    )
    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    transform_result = vectorizer.transform(
        distributions_data_generator,
        vectors=vectors_data_generator,
        n_distributions=distributions_data.shape[0],
        vector_dim=vectors_data.shape[1],
    )
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)

def test_wasserstein_vectorizer_generators_blockwise():
    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    vectorizer = WassersteinVectorizer(random_state=42, memory_size="50k")
    result = vectorizer.fit_transform(
        distributions_data_generator,
        vectors=vectors_data_generator,
        reference_distribution=generator_reference_dist,
        reference_vectors=generator_reference_vectors,
        n_distributions=distributions_data.shape[0],
        vector_dim=vectors_data.shape[1],
    )
    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    transform_result = vectorizer.transform(
        distributions_data_generator,
        vectors=vectors_data_generator,
        n_distributions=distributions_data.shape[0],
        vector_dim=vectors_data.shape[1],
    )
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)

def test_wasserstein_vectorizer_blockwise():
    vectorizer = WassersteinVectorizer(random_state=42, memory_size="50k")
    result = vectorizer.fit_transform(distributions_data, vectors=vectors_data)
    transform_result = vectorizer.transform(distributions_data, vectors=vectors_data)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


def test_sinkhorn_vectorizer_basic():
    vectorizer = SinkhornVectorizer(random_state=42)
    result = vectorizer.fit_transform(distributions_data, vectors=vectors_data)
    transform_result = vectorizer.transform(distributions_data, vectors=vectors_data)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


def test_sinkhorn_vectorizer_blockwise():
    vectorizer = SinkhornVectorizer(random_state=42, memory_size="50k")
    result = vectorizer.fit_transform(distributions_data, vectors=vectors_data)
    transform_result = vectorizer.transform(distributions_data, vectors=vectors_data)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


def test_wasserstein_vectorizer_list_based():
    lil_data = normalize(distributions_data, norm="l1").tolil()
    distributions = [np.array(x) for x in lil_data.data]
    vectors = [vectors_data[x] for x in lil_data.rows]
    vectorizer = WassersteinVectorizer(random_state=42)
    result = vectorizer.fit_transform(distributions, vectors=vectors)
    transform_result = vectorizer.transform(distributions, vectors=vectors)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


def test_wasserstein_vectorizer_list_based_blockwise():
    lil_data = normalize(distributions_data, norm="l1").tolil()
    distributions = [np.array(x) for x in lil_data.data]
    vectors = [vectors_data[x] for x in lil_data.rows]
    vectorizer = WassersteinVectorizer(random_state=42, memory_size="50k")
    result = vectorizer.fit_transform(distributions, vectors=vectors)
    transform_result = vectorizer.transform(distributions, vectors=vectors)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


def test_wasserstein_vectorizer_list_compared_to_sparse():
    lil_data = normalize(distributions_data.astype(np.float64), norm="l1").tolil()
    distributions = [np.array(x) for x in lil_data.data]
    vectors = [vectors_data[x] for x in lil_data.rows]
    vectorizer_sparse = WassersteinVectorizer(random_state=42)
    result_sparse = vectorizer_sparse.fit_transform(
        distributions_data, vectors=vectors_data
    )
    vectorizer_list = WassersteinVectorizer(random_state=42)
    result_list = vectorizer_list.fit_transform(
        distributions,
        vectors=vectors,
        reference_distribution=vectorizer_sparse.reference_distribution_,
        reference_vectors=vectorizer_sparse.reference_vectors_,
    )
    assert np.allclose(result_sparse, result_list, rtol=1e-3, atol=1e-6)


def test_wasserstein_vectorizer_generator_compared_to_sparse():
    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    vectorizer_sparse = WassersteinVectorizer(random_state=42)
    result_sparse = vectorizer_sparse.fit_transform(
        distributions_data, vectors=vectors_data
    )
    vectorizer_gen = WassersteinVectorizer(random_state=42)
    result_list = vectorizer_gen.fit_transform(
        distributions_data_generator,
        vectors=vectors_data_generator,
        reference_distribution=vectorizer_sparse.reference_distribution_,
        reference_vectors=vectorizer_sparse.reference_vectors_,
        n_distributions=distributions_data.shape[0],
        vector_dim=vectors_data.shape[1]
    )
    assert np.allclose(result_sparse, result_list, rtol=1e-3, atol=1e-6)


def test_approx_wasserstein_vectorizer_basic():
    vectorizer = ApproximateWassersteinVectorizer(random_state=42)
    result = vectorizer.fit_transform(distributions_data, vectors=vectors_data)
    transform_result = vectorizer.transform(distributions_data, vectors=vectors_data)
    assert np.allclose(result, transform_result, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize(
    "wasserstein_class",
    [WassersteinVectorizer, SinkhornVectorizer, ApproximateWassersteinVectorizer],
)
def test_wasserstein_based_vectorizer_bad_params(wasserstein_class):
    with pytest.raises(ValueError):
        vectorizer = wasserstein_class()
        vectorizer.fit(distributions_data)

    with pytest.raises(ValueError):
        vectorizer = wasserstein_class()
        vectorizer.fit(mixed_token_data, vectors=vectors_data)

    with pytest.raises(ValueError):
        vectorizer = wasserstein_class()
        vectorizer.fit(point_data, vectors=vectors_data)

    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    with pytest.raises(ValueError):
        vectorizer = WassersteinVectorizer()
        vectorizer.fit(distributions_data_generator, vectors=vectors_data_generator)

    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    with pytest.raises(ValueError):
        vectorizer = WassersteinVectorizer()
        vectorizer.fit(
            distributions_data_generator,
            vectors=vectors_data_generator,
            reference_vectors=np.random.random((10, vectors_data.shape[1])),
        )

    distributions_data_generator = (x for x in distributions_data_list)
    vectors_data_generator = (x for x in vectors_data_list)
    with pytest.raises(ValueError):
        vectorizer = WassersteinVectorizer(reference_size=20)
        vectorizer.fit(
            distributions_data_generator,
            vectors=vectors_data_generator,
            reference_vectors=np.random.random((10, vectors_data.shape[1])),
        )

@pytest.mark.parametrize(
    "wasserstein_class",
    [WassersteinVectorizer, SinkhornVectorizer],
)
def test_wasserstein_based_vectorizer_bad_metrics(wasserstein_class):
    with pytest.raises(ValueError):
        vectorizer = wasserstein_class(metric="unsupported_metric")
        vectorizer.fit(distributions_data, vectors=vectors_data)

    with pytest.raises(ValueError):
        vectorizer = wasserstein_class(metric=0.75)
        vectorizer.fit(distributions_data, vectors=vectors_data)


def test_node_removal():
    graph = scipy.sparse.random(10, 10, 0.1, format="csr")
    graph.data = np.ones_like(graph.data)
    # # Remove self-loops
    # for i in range(10):
    #     if graph[i, i] != 0:
    #         graph[i, i] = 0
    # graph.eliminate_zeros()
    node_to_remove = np.argmax(np.array(graph.sum(axis=0)).T[0])
    graph_less_node = remove_node(graph, node_to_remove, inplace=False)
    # assert (graph != graph_less_node).sum() > 0
    with pytest.raises(ValueError):
        graph_less_node = remove_node(graph, node_to_remove, inplace=True)
    inplace_graph = graph.tolil()
    remove_node(inplace_graph, node_to_remove, inplace=True)
    assert (inplace_graph != graph_less_node).sum() == 0

    assert np.all([node_to_remove not in row for row in inplace_graph.rows])
    assert len(inplace_graph.rows[node_to_remove]) == 0

    orig_graph = graph.tolil()
    for i, row in enumerate(orig_graph.rows):
        if node_to_remove in row and i != node_to_remove:
            assert np.all(
                np.unique(np.hstack([row, orig_graph.rows[node_to_remove]]))
                == np.unique(np.hstack([inplace_graph.rows[i], [node_to_remove]]))
            )


def test_multi_label_token_cooccurrence():
    vectorizer_a = TokenCooccurrenceVectorizer(
        multi_labelled_tokens=True,
        window_radii=[0, 1, 2],
        window_functions=["fixed", "fixed", "fixed"],
        kernel_functions=["flat", "flat", "flat"],
        window_orientations=["before", "before", "after"],
        normalize_windows=False,
        coo_max_memory="1G",
    )

    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 2, 1, 0, 1, 2, 1],
                [0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 1, 1],
                [1, 1, 0, 2, 2, 1, 2, 3, 2, 3, 3, 4],
                [0, 1, 2, 0, 1, 1, 3, 0, 1, 2, 4, 1],
            ]
        )
    )

    result = vectorizer_a.fit_transform(text_token_data_permutation)
    assert np.allclose(expected_result.toarray(), result.toarray())
    assert (
        vectorizer_a.fit_transform(text_token_data_permutation).nnz
        == vectorizer_a.transform(text_token_data_permutation).nnz
    )


def test_multi_label_token_cooccurrence_range():
    vectorizer_a = TokenCooccurrenceVectorizer(
        multi_labelled_tokens=True,
        window_radii=[1, 1],
        window_functions=["fixed", "fixed"],
        kernel_functions=["flat", "flat"],
        window_orientations=["before", "after"],
        kernel_args=[{"offset": 1}, {"offset": 1}],
        normalize_windows=False,
        coo_max_memory="1G",
    )

    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 2, 1, 1, 1, 2, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
    )

    result = vectorizer_a.fit_transform(text_token_data_permutation)
    assert np.allclose(expected_result.toarray(), result.toarray())


def test_multi_label_token_cooccurrence_harmonic():
    vectorizer_a = TokenCooccurrenceVectorizer(
        multi_labelled_tokens=True,
        window_radii=2,
        window_functions="fixed",
        kernel_functions="harmonic",
        window_orientations="after",
        # kernel_args={'offset': 1},
        normalize_windows=False,
        coo_max_memory="1G",
    )

    expected_result = scipy.sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 1.5, 0.5],
                [0.0, 0.0, 1.0, 1.0],
                [1.5, 1.8333334, 1.3333334, 2.8333335],
                [0.5, 1.3333334, 2.8333335, 0.33333334],
            ]
        )
    )

    result = vectorizer_a.fit_transform(text_token_data_permutation)
    assert np.allclose(expected_result.toarray(), result.toarray())


def test_multi_label_token_cooccurrence_em():
    vectorizer_a = TokenCooccurrenceVectorizer(
        multi_labelled_tokens=True,
        window_radii=[1, 1],
        window_functions=["fixed", "fixed"],
        kernel_functions=["flat", "flat"],
        window_orientations=["before", "after"],
        kernel_args=[{"offset": 1}, {"offset": 1}],
        n_iter=1,
    )

    result = vectorizer_a.fit_transform(text_token_data_permutation)
    result2 = vectorizer_a.transform(text_token_data_permutation)
    assert np.allclose(result2.toarray(), result.toarray())


@pytest.mark.parametrize("null", [True, False])
def test_multi_label_token_cooccurrence_masking(null):
    vectorizer_a = TokenCooccurrenceVectorizer(
        multi_labelled_tokens=True,
        window_radii=[1, 1],
        window_functions=["fixed", "fixed"],
        kernel_functions=["flat", "flat"],
        window_orientations=["before", "after"],
        kernel_args=[{"offset": 1}, {"offset": 1}],
        n_iter=1,
        max_occurrences=2,
        mask_string="xxx",
        nullify_mask=null,
    )

    result = vectorizer_a.fit_transform(text_token_data_permutation)
    result2 = vectorizer_a.transform(text_token_data_permutation)
    assert np.allclose(result2.toarray(), result.toarray())


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("include_values", [True, False])
def test_summarize_embedding_list(dense, include_values):
    vect = NgramVectorizer()
    weight_matrix = vect.fit_transform(text_token_data)
    if dense:
        weight_matrix = weight_matrix.todense()
    summary = summarize_embedding(
        weight_matrix, vect.column_index_dictionary_, include_values=include_values
    )
    expected_result = (
        [
            ["foo", "wer", "pok"],
            [],
            ["bar", "foo", "wer"],
            ["wer", "foo", "bar"],
            ["bar", "foo", "wer"],
            ["wer", "pok", "foo"],
            ["wer", "foo", "pok"],
        ],
        [
            [2.0, 1.0, 1.0],
            [],
            [4.0, 3.0, 2.0],
            [2.0, 2.0, 2.0],
            [4.0, 3.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 2.0],
        ],
    )

    if include_values:
        if dense:
            assert summary[0][2:7] == expected_result[0][2:7]
            assert summary[1][2:7] == expected_result[1][2:7]
        else:
            assert summary == expected_result
    else:
        if dense:
            assert summary[2:7] == expected_result[0][2:7]
        else:
            assert summary == expected_result[0]


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("include_values", [True, False])
def test_summarize_embedding_string(dense, include_values):
    vect = NgramVectorizer()
    weight_matrix = vect.fit_transform(text_token_data)
    if dense:
        weight_matrix = weight_matrix.todense()
    summary = summarize_embedding(
        weight_matrix,
        vect.column_index_dictionary_,
        k=2,
        return_type="string",
        include_values=include_values,
    )
    if include_values:
        expected_result = [
            "foo:2.0,wer:1.0",
            "",
            "bar:4.0,foo:3.0",
            "wer:2.0,foo:2.0",
            "bar:4.0,foo:3.0",
            "wer:3.0,pok:3.0",
            "wer:4.0,foo:4.0",
        ]
    else:
        expected_result = [
            "foo,wer",
            "",
            "bar,foo",
            "wer,foo",
            "bar,foo",
            "wer,pok",
            "wer,foo",
        ]
    if dense:
        assert summary[2:7] == expected_result[2:7]
    else:
        assert summary == expected_result


def test_categorical_columns_to_list():
    df = pd.DataFrame(path_graph.todense(), columns=["a", "b", "c", "d"])
    result_list = categorical_columns_to_list(df, ["a", "c"])
    expected_result = [["a:0", "c:0"], ["a:0", "c:1"], ["a:0", "c:0"], ["a:0", "c:0"]]
    assert expected_result == result_list


def test_categorical_column_to_list_bad_param():
    df = pd.DataFrame(path_graph.todense(), columns=["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        categorical_columns_to_list(df, ["a", "c", "foo"])


def test_lzcompression_vectorizer_basic():
    lzc = LZCompressionVectorizer()
    result1 = lzc.fit_transform(raw_string_data)
    result2 = lzc.transform(raw_string_data)
    assert np.allclose(result1.toarray(), result2.toarray())

def test_lzcompression_vectorizer_badparams():
    with pytest.raises(ValueError):
        lzc =  LZCompressionVectorizer(max_dict_size=-1)
        lzc.fit(raw_string_data)

    with pytest.raises(ValueError):
        lzc =  LZCompressionVectorizer(max_columns=-1)
        lzc.fit(raw_string_data)

def test_bpe_vectorizer_basic():
    bpe = BytePairEncodingVectorizer()
    result1 = bpe.fit_transform(raw_string_data)
    result2 = bpe.transform(raw_string_data)
    assert np.allclose(result1.toarray(), result2.toarray())

def test_bpe_tokens_ngram_matches():
    bpe1 = BytePairEncodingVectorizer(return_type="matrix")
    bpe2 = BytePairEncodingVectorizer(return_type="tokens")


    result1 = bpe1.fit_transform(raw_string_data)
    token_dictionary = {
        to_unicode(code, bpe1.tokens_, bpe1.max_char_code_):n
        for code, n in bpe1.column_label_dictionary_.items()
    }

    tokens = bpe2.fit_transform(raw_string_data)
    result2 = NgramVectorizer(token_dictionary=token_dictionary).fit_transform(tokens)

    assert np.allclose(result1.toarray(), result2.toarray())

def test_bpe_bad_params():
    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(max_vocab_size=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(min_token_occurrence=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(return_type=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(return_type="nonsense")
        bpe.fit(raw_string_data)

