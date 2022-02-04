import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse

MOCK_TARGET = np.ones(1, dtype=np.int64)


@numba.njit(nogil=True)
def column_kl_divergence_exact_prior(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength=0.1,
    target=MOCK_TARGET,
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
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength=0.1,
    target=MOCK_TARGET,
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


@numba.njit(nogil=True)
def supervised_column_kl(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength=0.1,
    target=MOCK_TARGET,
):
    observed = np.zeros_like(baseline_probabilities)
    for i in range(count_indices.shape[0]):
        idx = count_indices[i]
        label = target[idx]
        observed[label] += count_data[i]

    observed += prior_strength * baseline_probabilities
    observed /= observed.sum()

    return np.sum(observed * np.log(observed / baseline_probabilities))


@numba.njit(nogil=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    baseline_probabilities,
    column_kl_divergence_func,
    prior_strength=0.1,
    target=MOCK_TARGET,
):
    n_cols = indptr.shape[0] - 1
    weights = np.ones(n_cols)
    for i in range(n_cols):
        weights[i] = column_kl_divergence_func(
            indices[indptr[i] : indptr[i + 1]],
            data[indptr[i] : indptr[i + 1]],
            baseline_probabilities,
            prior_strength=prior_strength,
            target=target,
        )
    return weights


def information_weight(data, prior_strength=0.1, approximate_prior=False, target=None):
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

    target: ndarray or None (optional, default=None)
        If supervised target labels are available, these can be used to define distributions
        over the target classes rather than over rows, allowing weights to be
        supervised and target based. If None then unsupervised weighting is used.

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    if approximate_prior:
        column_kl_divergence_func = column_kl_divergence_approx_prior
    else:
        column_kl_divergence_func = column_kl_divergence_exact_prior

    baseline_counts = np.squeeze(np.array(data.sum(axis=1)))
    if target is None:
        baseline_probabilities = baseline_counts / baseline_counts.sum()
    else:
        baseline_probabilities = np.zeros(target.max() + 1)
        for i in range(baseline_probabilities.shape[0]):
            baseline_probabilities[i] = baseline_counts[target == i].sum()
        baseline_probabilities /= baseline_probabilities.sum()
        column_kl_divergence_func = supervised_column_kl

    csc_data = data.tocsc()
    csc_data.sort_indices()

    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        baseline_probabilities,
        column_kl_divergence_func,
        prior_strength=prior_strength,
        target=target,
    )
    return weights


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

    def __init__(
        self,
        prior_strength=1e-4,
        approx_prior=True,
        weight_power=2.0,
        supervision_weight=0.95,
    ):
        self.prior_strength = prior_strength
        self.approx_prior = approx_prior
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight

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

        self.information_weights_ = information_weight(
            X, self.prior_strength, self.approx_prior
        )

        if y is not None:
            unsupervised_power = (1.0 - self.supervision_weight) * self.weight_power
            supervised_power = self.supervision_weight * self.weight_power

            self.information_weights_ /= np.mean(self.information_weights_)
            self.information_weights_ = np.power(
                self.information_weights_, unsupervised_power
            )

            target_classes = np.unique(y)
            target_dict = dict(
                np.vstack((target_classes, np.arange(target_classes.shape[0]))).T
            )
            target = np.array(
                [np.int64(target_dict[label]) for label in y], dtype=np.int64
            )
            self.supervised_weights_ = information_weight(
                X, self.prior_strength, self.approx_prior, target=target
            )
            self.supervised_weights_ /= np.mean(self.supervised_weights_)
            self.supervised_weights_ = np.power(
                self.supervised_weights_, supervised_power
            )

            self.information_weights_ = (
                self.information_weights_ * self.supervised_weights_
            )
        else:
            self.information_weights_ /= np.mean(self.information_weights_)
            self.information_weights_ = np.power(
                self.information_weights_, self.weight_power
            )

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
