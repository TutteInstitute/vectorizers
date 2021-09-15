import pytest

from vectorizers.transformers import (
    RemoveEffectsTransformer,
    InformationWeightTransformer,
    CategoricalColumnTransformer,
    CountFeatureCompressionTransformer,
    SlidingWindowTransformer,
)
import numpy as np
import scipy.sparse
import pandas as pd

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
    RET = RemoveEffectsTransformer(
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
    RET = RemoveEffectsTransformer(
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
    RET = RemoveEffectsTransformer(
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


def test_sliding_window_transformer_basic():
    swt = SlidingWindowTransformer()
    result = swt.fit_transform(test_time_series)
    transform = swt.transform(test_time_series)
    for i, point_cloud in enumerate(result):
        for j, point in enumerate(point_cloud):
            assert np.allclose(point, transform[i][j])


def test_sliding_window_transformer_basic_w_lists():
    swt = SlidingWindowTransformer()
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

    swt = SlidingWindowTransformer(window_sample=[1.3, 1.1, 1.25, 1.625])
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)

    swt = SlidingWindowTransformer(window_width=-1)
    with pytest.raises(ValueError):
        result = swt.fit_transform(test_time_series)
