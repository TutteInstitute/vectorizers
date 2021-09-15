import numba
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from sklearn.preprocessing import normalize
import scipy.sparse
from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds


from warnings import warn

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

    def __init__(self, prior_strength=1e-4, approx_prior=True, weight_power=2.0, supervision_weight=0.95):
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
            self.information_weights_ = np.power(self.information_weights_, unsupervised_power)

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
            self.supervised_weights_ = np.power(self.supervised_weights_, supervised_power)

            self.information_weights_ = (
                self.information_weights_ * self.supervised_weights_
            )
        else:
            self.information_weights_ /= np.mean(self.information_weights_)
            self.information_weights_ = np.power(self.information_weights_, self.weight_power)

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


class Wasserstein1DHistogramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        X = check_array(X)
        normalized_X = normalize(X, norm="l1")
        result = np.cumsum(normalized_X, axis=1)
        self.metric_ = "l1"
        return result


class SequentialDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        result = []

        for sequence in X:
            seq = np.array(sequence)
            result.append(seq[self.offset :] - seq[: -self.offset])

        return result


class CategoricalColumnTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer is useful for describing an object as a bag of the categorical values that
    have been used to represent it within a pandas DataFrame.

    It takes an categorical column name to groupby, object_column_name, and one
    or more categorical columns to be used to describe these objects,
    descriptor_column_name.  Then it returns a Series with an index being the
    unique entries of your object_column_name and the values being a list of
    the appropriate categorical values from your descriptor_column_name.

    It can be thought of as a PivotTableTransformer if you'd like.

    Parameters
    ----------
    object_column_name: string
        The column name from the DataFrame where our object values can be found.
        This will be the thing we are grouping by.

    descriptor_column_name: string or list
        The name or names of the categorical column(s) who's values will be used for describing our
        objects.  If you are using multiple names it's recommended that you set include_column_name=True.

    include_column_name: bool (default = False)
        Should the column name be appended at the beginning of each value?
        This is useful if you intend to combine values from multiple categorical columns
        after the fact.

    unique_values: bool (default = False)
        Should we apply a unique to the values in column before building our list representation?

    """

    def __init__(
        self,
        object_column_name,
        descriptor_column_name,
        include_column_name=False,
        unique_values=False,
    ):
        self.object_column_name = object_column_name
        self.descriptor_column_name = descriptor_column_name
        # Get everything on consistent footing so we don't have to handle multiple cases.
        if type(self.descriptor_column_name) == str:
            self.descriptor_column_name_ = [self.descriptor_column_name]
        else:
            self.descriptor_column_name_ = self.descriptor_column_name
        self.include_column_name = include_column_name
        self.unique_values = unique_values

        if (
            (self.include_column_name is False)
            and (type(self.descriptor_column_name) == list)
            and (len(self.descriptor_column_name) > 1)
        ):
            warn(
                "It is recommended that if you are aggregating "
                "multiple columns that you set include_column_name=True"
            )

    def fit_transform(self, X, y=None, **fit_params):
        """
        This transformer is useful for describing an object as a bag of the categorical values that
        have been used to represent it within a pandas DataFrame.

        It takes an categorical column name to groupby, object_column_name, and one or more
        categorical columns to be used to describe these objects, descriptor_column_name.
        Then it returns a Series with an index being the unique entries of your object_column_name
        and the values being a list of the appropriate categorical values from your descriptor_column_name.

        Parameters
        ----------
        X: pd.DataFrame
            a pandas dataframe with columns who's names match those specified in the object_column_name and
            descriptor_column_name of the constructor.

        Returns
        -------
        pandas Series
            Series with an index being the unique entries of your object_column_name
            and the values being a list of the appropriate categorical values from your descriptor_column_name.
        """
        # Check that the dataframe has the appropriate columns
        required_columns = set([self.object_column_name] + self.descriptor_column_name_)
        if not required_columns.issubset(X.columns):
            raise ValueError(
                f"Sorry the required column(s) {set(required_columns).difference(set(X.columns))} are not "
                f"present in your data frame. \n"
                f"Please either specify a new instance or apply to a different data frame. "
            )

        # Compute a single groupby ahead of time to save on compute
        grouped_frame = X.groupby(self.object_column_name)
        aggregated_columns = []
        for column in self.descriptor_column_name_:
            if self.include_column_name:
                if self.unique_values:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [
                                column + ":" + value
                                for value in x.unique()
                                if pd.notna(value)
                            ]
                        )
                    )
                else:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [
                                column + ":" + value for value in x if pd.notna(value)
                            ]
                        )
                    )
            else:
                if self.unique_values:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [value for value in x.unique() if pd.notna(value)]
                        )
                    )
                else:
                    aggregated_columns.append(
                        grouped_frame[column].agg(
                            lambda x: [value for value in x if pd.notna(value)]
                        )
                    )
        reduced = pd.concat(aggregated_columns, axis="columns").sum(axis=1)
        return reduced

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self


class CountFeatureCompressionTransformer(BaseEstimator, TransformerMixin):
    """Large sparse high dimensional matrices of count based, or strictly
    non-negative features are common. This transformer provides a simple
    but often very effective dimension reduction approach to provide a
    dense representation of the data that is amenable to cosine based distance
    measures.

    Parameters
    ----------
    n_components: int (optional, default=128)
        The number of dimensions to use for the dense reduced representation.

    n_iter: int (optional, default=7)
        If using the ``"randomized"`` algorithm for SVD then use this number of
        iterations for estimate the SVD.

    algorithm: string (optional, default="randomized")
        The algorithm to use internally for the SVD step. Should be one of
            * "arpack"
            * "randomized"

    random_state: int, np.random_state or None (optional, default=None)
        If using the ``"randomized"`` algorithm for SVD then use this as the
        random state (or random seed).
    """

    def __init__(
        self, n_components=128, n_iter=7, algorithm="randomized", random_state=None, rescaling_power=0.5,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.algorithm = algorithm
        self.random_state = random_state
        self.rescaling_power = rescaling_power

    def fit_transform(self, X, y=None, **fit_params):
        """
        Given a dataset of count based features (i.e. strictly positive)
        perform feature compression / dimension reduction to provide
        a dataset with ``self.n_components`` dimensions suitable for
        measuring distances using cosine distance.

        Parameters
        ----------
        X: ndarray or sparse matrix of shape (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        result: ndarray of shape (n_samples, n_components)
            The dimension reduced representation of the input.
        """
        # Handle too large an n_components value somewhat gracefully
        if self.n_components >= X.shape[1]:
            warn(
                f"Warning: n_components is {self.n_components} but input has only {X.shape[1]} features!"
                f"No compression will be performed."
            )
            self.components_ = np.eye(X.shape[1])
            self.component_scaling_ = np.ones(X.shape[1])
            return X

        if scipy.sparse.isspmatrix(X):
            if np.any(X.data < 0.0):
                raise ValueError("All entries in input most be non-negative!")
        else:
            if np.any(X < 0.0):
                raise ValueError("All entries in input most be non-negative!")

        normed_data = normalize(X)
        rescaled_data = scipy.sparse.csr_matrix(normed_data)
        rescaled_data.data = np.power(normed_data.data, self.rescaling_power)
        if self.algorithm == "arpack":
            u, s, v = svds(rescaled_data, k=self.n_components)
        elif self.algorithm == "randomized":
            random_state = check_random_state(self.random_state)
            u, s, v = randomized_svd(
                rescaled_data,
                n_components=self.n_components,
                n_iter=self.n_iter,
                random_state=random_state,
            )
        else:
            raise ValueError("algorithm should be one of 'arpack' or 'randomized'")

        u, v = svd_flip(u, v)
        self.component_scaling_ = np.sqrt(s)
        self.components_ = v
        self.metric_ = "cosine"

        result = u * self.component_scaling_

        return result

    def fit(self, X, y=None, **fit_params):
        """
        Given a dataset of count based features (i.e. strictly positive)
        learn a feature compression / dimension reduction to provide
        a dataset with ``self.n_components`` dimensions suitable for
        measuring distances using cosine distance.

        Parameters
        ----------
        X: ndarray or sparse matrix of shape (n_samples, n_features)
            The input data to be transformed.
        """
        self.fit_transform(self, X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        """
        Given a dataset of count based features (i.e. strictly positive)
        perform the learned feature compression / dimension reduction.

        Parameters
        ----------
        X: ndarray or sparse matrix of shape (n_samples, n_features)
            The input data to be transformed.

        Returns
        -------
        result: ndarray of shape (n_samples, n_components)
            The dimension reduced representation of the input.
        """
        check_is_fitted(
            self,
            ["components_", "component_scaling_"],
        )
        normed_data = normalize(X)
        rescaled_data = scipy.sparse.csr_matrix(normed_data)
        rescaled_data.data = np.power(normed_data.data, self.rescaling_power)

        result = (rescaled_data @ self.components_.T) / self.component_scaling_

        return result
