import numba
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from warnings import warn
from sklearn.neighbors import KernelDensity
from .utils import flatten


@numba.njit(nogil=True)
def min_non_zero_difference(data):
    """Find the minimum non-zero sequential difference in a single dimensional
    array of values. This is useful for determining the minimal reasonable kernel
    bandwidth for a 1-dimensional KDE over a dataset.

    Parameters
    ----------
    data: array
        One dimensional array of values

    Returns
    -------
    min_difference: float
        The minimal difference between sequential values.
    """
    sorted_data = np.sort(data)
    differences = sorted_data[1:] - sorted_data[:-1]
    return np.min(differences[differences > 0])


def jackknife_bandwidths(data, bandwidths, kernel="gaussian"):
    """Perform jack-knife sampling over different bandwidths for KDEs for each
    time-series in the dataset.

    Parameters
    ----------
    data: list of arrays
        A list of (variable length) arrays of values. The values should represent
        "times" of "events".

    bandwidths: array
        The possible bandwidths to try

    kernel: string (optional, default="gaussian")
        The kernel to use for the KDE. Should be accepted by sklearn's KernelDensity
        class.

    Returns
    -------
    result: array of shape (n_bandwidths,)
        The total likelihood of unobserved data over all jackknife samplings and all
        time series in the dataset for each bandwidth.
    """
    result = np.zeros(bandwidths.shape[0])
    for j in range(bandwidths.shape[0]):
        kde = KernelDensity(bandwidth=bandwidths[j], kernel=kernel)
        for i in range(len(data)):
            likelihood = 0.0
            for k in range(len(data[i])):
                if k < len(data[i]) - 1:
                    jackknife_sample = np.hstack([data[i][:k], data[i][k + 1 :]])
                else:
                    jackknife_sample = data[i][:k]
                kde.fit(jackknife_sample[:, None])
                likelihood += np.exp(kde.score(np.array([[data[i][k]]])))

            result[j] += likelihood

    return result


class KDEVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bandwidth=None,
        n_components=50,
        kernel="gaussian",
        evaluation_grid_strategy="uniform",
    ):
        self.n_components = n_components
        self.evaluation_grid_strategy = evaluation_grid_strategy
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y=None, **fit_params):

        combined_data = np.array(flatten(X))

        if self.bandwidth is None:
            # Estimate the bandwidth by looking at training data
            # We do a jack-knife across each time series and
            # find the bandwidth choice that works best over all
            # time series
            min, max = np.min(combined_data), np.max(combined_data)
            avg_n_events = np.mean([len(x) for x in X])
            max_bandwidth = (max - min) / avg_n_events
            min_bandwidth = min_non_zero_difference(combined_data)
            bandwidths = 10.0 ** np.linspace(
                np.log10(min_bandwidth), np.log10(max_bandwidth), 50
            )
            jackknifed_total_likelihoods = jackknife_bandwidths(X, bandwidths)
            self.bandwidth_ = bandwidths[np.argmax(jackknifed_total_likelihoods)]
        else:
            self.bandwidth_ = self.bandwidth

        if self.evaluation_grid_strategy == "uniform":
            min, max = np.min(combined_data), np.max(combined_data)
            self.evaluation_grid_ = np.linspace(min, max, self.n_components)
        elif self.evaluation_grid_strategy == "density":
            uniform_quantile_grid = np.linspace(0, 1.0, self.n_components)
            self.evaluation_grid_ = np.quantile(combined_data, uniform_quantile_grid)
        else:
            raise ValueError(
                "Unrecognized evaluation_grid_strategy; should be one "
                'of: "uniform" or "density"'
            )

        return self

    def transform(self, X):
        check_is_fitted(self, ["bandwidth_", "evaluation_grid_"])

        result = np.empty((len(X), self.n_components), dtype=np.float64)

        for i, sample in enumerate(X):
            kde = KernelDensity(bandwidth=self.bandwidth_, kernel=self.kernel)
            kde.fit(sample[:, None])
            log_probability = kde.score_samples(self.evaluation_grid_[:, None])
            result[i] = np.exp(log_probability)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
