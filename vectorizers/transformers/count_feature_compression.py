import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
)
from sklearn.preprocessing import normalize
import scipy.sparse
from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds

from warnings import warn


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
        self,
        n_components=128,
        n_iter=7,
        algorithm="randomized",
        random_state=None,
        rescaling_power=0.5,
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
