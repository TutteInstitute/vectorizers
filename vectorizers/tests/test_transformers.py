import pytest

from vectorizers.transformers import (
    RowDenoisingTransformer,
    InformationWeightTransformer,
    CategoricalColumnTransformer,
    CountFeatureCompressionTransformer,
    SlidingWindowTransformer,
    SequentialDifferenceTransformer,
    sliding_window_generator,
)
import numpy as np
import scipy.sparse
import pandas as pd
import numba

test_matrix = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_matrix_zero_row = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
test_matrix_zero_row.eliminate_zeros()
test_matrix_zero_column = scipy.sparse.csr_matrix([[1, 2, 0], [4, 5, 0], [7, 8, 0]])
test_matrix_zero_column.eliminate_zeros()

test_df = pd.DataFrame(
    {
        "id": ["one", "two", "one", "two"],
        "A": ["foo", "bar", "pok", "bar"],
        "B": ["x", "k", "c", "d"],
    }
)

test_time_series = [
    np.random.random(size=23),
    np.random.random(size=56),
    np.random.random(size=71),
    np.random.random(size=64),
    np.random.random(size=35),
    np.random.random(size=44),
]

changepoint_position = np.random.randint(11, 100) # changepoint position must be at least window_width in
changepoint_sequence = np.random.poisson(0.75, size=100)
changepoint_sequence[changepoint_position] = 10


@pytest.mark.parametrize("include_column_name", [True, False])
@pytest.mark.parametrize("unique_values", [True, False])
def test_CategoricalColumnTransformer(include_column_name, unique_values):
    result = CategoricalColumnTransformer(
        object_column_name="id",
        descriptor_column_name="A",
        include_column_name=include_column_name,
        unique_values=unique_values,
    ).fit_transform(test_df)

    if include_column_name:
        if unique_values:
            expected_result = pd.Series(
                [["A:foo", "A:pok"], ["A:bar"]], index=["one", "two"]
            )
        else:
            expected_result = pd.Series(
                [["A:foo", "A:pok"], ["A:bar", "A:bar"]], index=["one", "two"]
            )
    else:
        if unique_values:
            expected_result = pd.Series([["foo", "pok"], ["bar"]], index=["one", "two"])
        else:
            expected_result = pd.Series(
                [["foo", "pok"], ["bar", "bar"]], index=["one", "two"]
            )
    assert (result == expected_result).all()


@pytest.mark.parametrize("include_column_name", [True, False])
@pytest.mark.parametrize("unique_values", [True, False])
def test_CategoricalColumnTransformer_multi_column(include_column_name, unique_values):
    result = CategoricalColumnTransformer(
        object_column_name="id",
        descriptor_column_name=["A", "B"],
        include_column_name=include_column_name,
        unique_values=unique_values,
    ).fit_transform(test_df)

    if include_column_name:
        if unique_values:
            expected_result = pd.Series(
                [["A:foo", "A:pok", "B:x", "B:c"], ["A:bar", "B:k", "B:d"]],
                index=["one", "two"],
            )
        else:
            expected_result = pd.Series(
                [["A:foo", "A:pok", "B:x", "B:c"], ["A:bar", "A:bar", "B:k", "B:d"]],
                index=["one", "two"],
            )
    else:
        if unique_values:
            expected_result = pd.Series(
                [["foo", "pok", "x", "c"], ["bar", "k", "d"]], index=["one", "two"]
            )
        else:
            expected_result = pd.Series(
                [["foo", "pok", "x", "c"], ["bar", "bar", "k", "d"]],
                index=["one", "two"],
            )
    assert (result == expected_result).all()


def test_CategoricalColumnTransformer_bad_param():
    with pytest.raises(ValueError):
        CategoricalColumnTransformer(
            object_column_name="id",
            descriptor_column_name=["A", "BAD"],
        ).fit_transform(test_df)


@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer(
    em_precision,
    em_background_prior,
    em_threshold,
    em_prior_strength,
    normalize,
):
    RET = RowDenoisingTransformer(
        em_precision=em_precision,
        em_background_prior=em_background_prior,
        em_threshold=em_threshold,
        em_prior_strength=em_prior_strength,
        normalize=normalize,
    )
    result = RET.fit_transform(test_matrix)
    transform = RET.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer_zero_column(
    em_precision,
    em_background_prior,
    em_threshold,
    em_prior_strength,
    normalize,
):
    RET = RowDenoisingTransformer(
        em_precision=em_precision,
        em_background_prior=em_background_prior,
        em_threshold=em_threshold,
        em_prior_strength=em_prior_strength,
        normalize=normalize,
    )
    result = RET.fit_transform(test_matrix_zero_column)
    transform = RET.transform(test_matrix_zero_column)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer_zero_row(
    em_precision,
    em_background_prior,
    em_threshold,
    em_prior_strength,
    normalize,
):
    RET = RowDenoisingTransformer(
        em_precision=em_precision,
        em_background_prior=em_background_prior,
        em_threshold=em_threshold,
        em_prior_strength=em_prior_strength,
        normalize=normalize,
    )
    result = RET.fit_transform(test_matrix_zero_row)
    transform = RET.transform(test_matrix_zero_row)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("prior_strength", [0.1, 1.0])
@pytest.mark.parametrize("approx_prior", [True, False])
def test_iw_transformer(prior_strength, approx_prior):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength,
        approx_prior=approx_prior,
    )
    result = IWT.fit_transform(test_matrix)
    transform = IWT.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("prior_strength", [0.1, 1.0])
@pytest.mark.parametrize("approx_prior", [True, False])
def test_iw_transformer_supervised(prior_strength, approx_prior):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength,
        approx_prior=approx_prior,
    )
    result = IWT.fit_transform(test_matrix, np.array([0, 1, 1]))
    transform = IWT.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("prior_strength", [0.1, 1.0])
@pytest.mark.parametrize("approx_prior", [True, False])
def test_iw_transformer_zero_column(prior_strength, approx_prior):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength,
        approx_prior=approx_prior,
    )
    result = IWT.fit_transform(test_matrix_zero_column)
    transform = IWT.transform(test_matrix_zero_column)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("prior_strength", [0.1, 1.0])
@pytest.mark.parametrize("approx_prior", [True, False])
def test_iw_transformer_zero_row(prior_strength, approx_prior):
    IWT = InformationWeightTransformer(
        prior_strength=prior_strength,
        approx_prior=approx_prior,
    )
    result = IWT.fit_transform(test_matrix_zero_row)
    transform = IWT.transform(test_matrix_zero_row)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("algorithm", ["randomized", "arpack"])
def test_count_feature_compression_basic(algorithm):
    cfc = CountFeatureCompressionTransformer(n_components=2, algorithm=algorithm)
    result = cfc.fit_transform(test_matrix)
    transform = cfc.transform(test_matrix)
    assert np.allclose(result, transform)


def test_count_feature_compression_warns():
    cfc = CountFeatureCompressionTransformer(n_components=5)
    with pytest.warns(UserWarning):
        result = cfc.fit_transform(test_matrix)


def test_count_feature_compression_bad_input():
    cfc = CountFeatureCompressionTransformer(n_components=2)
    with pytest.raises(ValueError):
        result = cfc.fit_transform(-test_matrix)

    with pytest.raises(ValueError):
        result = cfc.fit_transform(-test_matrix.toarray())

    cfc = CountFeatureCompressionTransformer(n_components=2, algorithm="bad_value")
    with pytest.raises(ValueError):
        result = cfc.fit_transform(test_matrix)


@pytest.mark.parametrize("pad_width", [0, 1])
@pytest.mark.parametrize(
    "kernel",
    [
        "average",
        ("differences", 0, 1, 1),
        ("position_velocity", 2, 1, 1),
        ("weight", np.array([0.1, 0.75, 1.5, 1.0, 0.25])),
        ("gaussian_weight", 2),
        np.random.random((5, 5)),
        numba.njit(lambda x: x.cumsum()),
    ],
)
@pytest.mark.parametrize("sample", [None, (0, 1), np.arange(5), [4, 1, 3, 2, 0]])
def test_sliding_window_transformer_basic(pad_width, kernel, sample):
    swt = SlidingWindowTransformer(
        window_width=5, pad_width=pad_width, kernels=[kernel], window_sample=sample
    )
    result = swt.fit_transform(test_time_series)
    transform = swt.transform(test_time_series)
    for i, point_cloud in enumerate(result):
        for j, point in enumerate(point_cloud):
            assert np.allclose(point, transform[i][j])


@pytest.mark.parametrize("pad_width", [0, 1])
@pytest.mark.parametrize(
    "kernel",
    [
        "average",
        ("differences", 0, 1, 1),
        ("position_velocity", 2, 1, 1),
        ("weight", np.array([0.1, 0.75, 1.5, 1.0, 0.25])),
        ("gaussian_weight", 2),
        np.random.random((5, 5)),
        numba.njit(lambda x: x.cumsum(), cache=True),
    ],
)
@pytest.mark.parametrize("sample", [None, np.arange(5), [4, 1, 3, 2, 0]])
def test_sliding_window_generator_matches_transformer(pad_width, kernel, sample):
    swt = SlidingWindowTransformer(
        window_width=5, pad_width=pad_width, kernels=[kernel], window_sample=sample
    )
    transformer_result = swt.fit_transform(test_time_series)
    test_window = (
        None
        if not callable(kernel)
        else np.asarray(test_time_series[0])[: swt.window_width][swt.window_sample_]
    )
    generator_result = list(
        sliding_window_generator(
            test_time_series,
            test_time_series[0].shape,
            window_width=5,
            pad_width=pad_width,
            kernels=[kernel],
            window_sample=sample,
            test_window=test_window,
        )
    )
    for i, point_cloud in enumerate(transformer_result):
        for j, point in enumerate(point_cloud):
            assert np.allclose(point, generator_result[i][j])

@pytest.mark.parametrize("window_width", [5, 10])
def test_sliding_window_count_changepoint(window_width):
    swt = SlidingWindowTransformer(
        window_width=window_width, kernels=[("count_changepoint", 1.0, 2.0)],
    )
    changepoint_scores = swt.fit_transform([changepoint_sequence])[0].flatten()
    assert np.argmax(changepoint_scores) + window_width - 1 == changepoint_position

@pytest.mark.parametrize("pad_width", [0, 1])
@pytest.mark.parametrize(
    "kernel",
    [
        "average",
        ("differences", 0, 1, 1),
        ("position_velocity", 2, 1, 1),
        ("weight", np.array([0.1, 0.75, 1.5, 1.0, 0.25])),
        np.random.random((5, 5)),
        numba.njit(lambda x: x.cumsum()),
    ],
)
@pytest.mark.parametrize("sample", [None, np.arange(5), [4, 1, 3, 2, 0]])
def test_sliding_window_transformer_basic_w_lists(pad_width, kernel, sample):
    swt = SlidingWindowTransformer(
        window_width=5, pad_width=pad_width, kernels=[kernel], window_sample=sample
    )
    result = swt.fit_transform([list(x) for x in test_time_series])
    transform = swt.transform([list(x) for x in test_time_series])
    for i, point_cloud in enumerate(result):
        for j, point in enumerate(point_cloud):
            assert np.allclose(point, transform[i][j])


def test_sliding_window_transformer_w_sampling():
    swt = SlidingWindowTransformer(window_sample="random", window_sample_size=5)
    result = swt.fit_transform(test_time_series)
    transform = swt.transform(test_time_series)
    for i, point_cloud in enumerate(result):
        for j, point in enumerate(point_cloud):
            assert np.allclose(point, transform[i][j])


def test_sliding_window_transformer_bad_params():
    swt = SlidingWindowTransformer(window_sample="foo")
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(window_sample=("foo", "bar"))
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(window_sample=1.105)
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(window_width=-1)
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(kernels=["not a kernel"])
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(kernels=-1)
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(kernels=np.array([[1, 2, 3], [1, 2, 3]]))
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

def test_seq_diff_transformer_basic():
    sdt = SequentialDifferenceTransformer()
    diffs = sdt.fit_transform(test_time_series)
    transform_diffs = sdt.transform(test_time_series)
    for i, seq_diffs in enumerate(diffs):
        assert np.allclose(np.array(seq_diffs), np.array(transform_diffs[i]))
        assert np.allclose(test_time_series[i][:-1] + np.ravel(seq_diffs), test_time_series[i][1:])