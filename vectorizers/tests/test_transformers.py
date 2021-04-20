import pytest

from vectorizers.transformers import (
    RemoveEffectsTransformer,
    InformationWeightTransformer,
)
import numpy as np
import scipy.sparse

test_matrix = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_matrix_zero_row = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
test_matrix_zero_row.eliminate_zeros()
test_matrix_zero_column = scipy.sparse.csr_matrix([[1, 2, 0], [4, 5, 0], [7, 8, 0]])
test_matrix_zero_column.eliminate_zeros()

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
