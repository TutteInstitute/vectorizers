import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import normalize
import scipy.sparse

from warnings import warn


@numba.njit(nogil=True)
def column_kl_divergence_exact_prior(
    count_indices, count_data, baseline_probabilities, prior_strength=0.1
):
    observed_norm = count_data.sum() + prior_strength
    observed_zero_constant = (prior_strength / observed_norm) * np.log(
        prior_strength / observed_norm
    )
    result = 0.0
    count_indices_set = set(count_indices)
    for i in range(baseline_probabilities.shape[0]):
        if i in count_indices_set:
            idx = np.searchsorted(count_indices, i)
            observed_probability = (
                count_data[idx] + prior_strength * baseline_probabilities[i]
            ) / observed_norm
            if observed_probability > 0.0:
                result += observed_probability * np.log(
                    observed_probability / baseline_probabilities[i]
                )
        else:
            result += baseline_probabilities[i] * observed_zero_constant

    return result


@numba.njit(nogil=True)
def column_kl_divergence_approx_prior(
    count_indices, count_data, baseline_probabilities, prior_strength=0.1
):
    observed_norm = count_data.sum() + prior_strength
    observed_zero_constant = (prior_strength / observed_norm) * np.log(
        prior_strength / observed_norm
    )
    result = 0.0
    zero_count_component_estimate = (
        np.mean(baseline_probabilities)
        * observed_zero_constant
        * (baseline_probabilities.shape[0] - count_indices.shape[0])
    )
    result += zero_count_component_estimate
    for i in range(count_indices.shape[0]):
        idx = count_indices[i]
        observed_probability = (
            count_data[i] + prior_strength * baseline_probabilities[idx]
        ) / observed_norm
        if observed_probability > 0.0 and baseline_probabilities[idx] > 0:
            result += observed_probability * np.log(
                observed_probability / baseline_probabilities[idx]
            )

    return result


@numba.njit(nogil=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    baseline_probabilities,
    column_kl_divergence_func,
    prior_strength=0.1,
):
    n_cols = indptr.shape[0] - 1
    weights = np.ones(n_cols)
    for i in range(n_cols):
        weights[i] = column_kl_divergence_func(
            indices[indptr[i] : indptr[i + 1]],
            data[indptr[i] : indptr[i + 1]],
            baseline_probabilities,
            prior_strength=prior_strength,
        )
    return weights


def information_weight(data, prior_strength=0.1, approximate_prior=False):
    """Compute information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution. The Bayesian prior can either be computed
    exactly (the default) at some computational expense, or estimated for a much
    fast computation, often suitable for large or very sparse datasets.

    Parameters
    ----------
    data: scipy sparse matrix (n_samples, n_features)
        A matrix of count data where rows represent observations and
        columns represent features. Column weightings will be learned
        from this data.

    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    approximate_prior: bool (optional, default=False)
        Whether to approximate weights based on the Bayesian prior or perform
        exact computations. Approximations are much faster especialyl for very
        large or very sparse datasets.

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    baseline_counts = np.squeeze(np.array(data.sum(axis=1)))
    baseline_probabilities = baseline_counts / baseline_counts.sum()
    csc_data = data.tocsc()
    csc_data.sort_indices()
    if approximate_prior:
        column_kl_divergence_func = column_kl_divergence_approx_prior
    else:
        column_kl_divergence_func = column_kl_divergence_exact_prior
    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        baseline_probabilities,
        column_kl_divergence_func,
        prior_strength=prior_strength,
    )
    return weights


@numba.njit()
def numba_multinomial_em_sparse(
    indptr,
    inds,
    data,
    background,
    precision=1e-7,
    low_thresh=1e-5,
    bg_prior=5.0,
    prior_strength=0.3,
):
    result = np.zeros(data.shape[0], dtype=np.float32)
    mix_weights = np.zeros(indptr.shape[0] - 1, dtype=np.float32)

    prior = np.array([1.0, bg_prior]) * prior_strength
    mp = 1.0 + 1.0 * np.sum(prior)

    for i in range(indptr.shape[0] - 1):
        indices = inds[indptr[i] : indptr[i + 1]]
        row_data = data[indptr[i] : indptr[i + 1]]

        row_background = np.zeros_like(row_data)
        for idx in range(indices.shape[0]):
            j = indices[idx]
            row_background[idx] = background[j]

        row_background = row_background / row_background.sum()

        mix_param = 0.5
        current_dist = mix_param * row_data + (1.0 - mix_param) * row_background

        last_mix_param = mix_param
        change_magnitude = 1.0

        while (
            change_magnitude > precision
            and mix_param > precision
            and mix_param < 1.0 - precision
        ):

            posterior_dist = current_dist * mix_param
            posterior_dist /= current_dist * mix_param + row_background * (
                1.0 - mix_param
            )

            current_dist = posterior_dist * row_data
            mix_param = (current_dist.sum() + prior[0]) / mp
            current_dist = current_dist / current_dist.sum()

            change_magnitude = np.abs(mix_param - last_mix_param)
            last_mix_param = mix_param

        # zero out any small values
        norm = 0.0
        for n in range(current_dist.shape[0]):
            if current_dist[n] < low_thresh:
                current_dist[n] = 0.0
            else:
                norm += current_dist[n]
        current_dist /= norm

        result[indptr[i] : indptr[i + 1]] = current_dist
        mix_weights[i] = mix_param

    return result, mix_weights


def multinomial_em_sparse(
    matrix,
    background,
    precision=1e-7,
    low_thresh=1e-5,
    bg_prior=5.0,
    prior_strength=0.3,
):
    if scipy.sparse.isspmatrix_csr(matrix):
        result = matrix.copy().astype(np.float32)
    else:
        result = matrix.tocsr().astype(np.float32)
    new_data, mix_weights = numba_multinomial_em_sparse(
        result.indptr,
        result.indices,
        result.data,
        background,
        precision,
        low_thresh,
        bg_prior,
        prior_strength,
    )
    result.data = new_data

    return result, mix_weights


class InformationWeightTransformer(BaseEstimator, TransformerMixin):
    """A data transformer that re-weights columns of count data. Column weights
    are computed as information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution. The Bayesian prior can either be computed
    exactly (the default) at some computational expense, or estimated for a much
    fast computation, often suitable for large or very sparse datasets.

    Parameters
    ----------
    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    approximate_prior: bool (optional, default=False)
        Whether to approximate weights based on the Bayesian prior or perform
        exact computations. Approximations are much faster especialyl for very
        large or very sparse datasets.

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    def __init__(self, prior_strength=1e-4, approx_prior=True):
        self.prior_strength = prior_strength
        self.approx_prior = approx_prior

    def fit(self, X, y=None, **fit_kwds):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        self:
            The trained model.
        """
        if not scipy.sparse.isspmatrix(X):
            X = scipy.sparse.csc_matrix(X)

        self.information_weights_ = information_weight(X, self.prior_strength, self.approx_prior)
        return self

    def transform(self, X):
        """Reweight data ``X`` based on learned information weights of columns.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be transformed. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        result: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The reweighted data.
        """
        result = X @ scipy.sparse.diags(self.information_weights_)
        return result

class RemoveEffectsTransformer(BaseEstimator, TransformerMixin):
    """

       Parameters
       ----------
        normalize = False
            Return the modified count matrix (default) or the L_1 normalization of each row.

        optional EM params:
        * em_precision = 1e-7, (halt EM when the mix_param changes less than this)
        * em_threshold = 1e-5, (set to zero any values below this)
        * em_background_prior = 5.0, (a non-negative number)
        * em_prior_strength = 0.3 (a non-negative number)
       """

    def __init__(
        self,
        em_precision=1.0e-7,
        em_background_prior=1.0,
        em_threshold=1.0e-8,
        em_prior_strength=0.5,
        normalize=False,
    ):
        self.em_threshold = em_threshold
        self.em_background_prior = em_background_prior
        self.em_precision = em_precision
        self.em_prior_strength = em_prior_strength
        self.normalize = normalize

    def fit(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix to used to find the low-rank effects

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        self

        """
        if scipy.sparse.issparse(X):
            X.eliminate_zeros()
            if X.nnz == 0:
                warn("Cannot fit an empty matrix")
                return self
            self.background_model_ = np.squeeze(
                np.array(X.sum(axis=0), dtype=np.float32)
            )
        else:
            self.background_model_ = X.sum(axis=0)

        self.background_model_ /= self.background_model_.sum()

        return self

    def transform(self, X, y=None):
        """

        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that has the effects removed

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X with the low-rank effects removed.

        """

        check_is_fitted(self, ["background_model_"])

        row_sums = np.array(X.sum(axis=1)).T[0]

        result, weights = multinomial_em_sparse(
            normalize(X, norm="l1"),
            self.background_model_,
            low_thresh=self.em_threshold,
            bg_prior=self.em_background_prior,
            precision=self.em_precision,
            prior_strength=self.em_prior_strength,
        )
        self.mix_weights_ = weights
        if not self.normalize:
            result = scipy.sparse.diags(row_sums * weights) * result

        result.eliminate_zeros()

        return result

    def fit_transform(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that is used to deduce the low-rank effects and then has them removed

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X with the low-rank effects removed.

        """
        self.fit(X, **fit_params)
        if X.nnz == 0:
            return X
        return self.transform(X)
