from warnings import warn

import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from .utils import flatten, vectorize_diagram, pairwise_gaussian_ground_distance
import vectorizers.distances as distances


class DistributionVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=20,
        random_state=None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def _validate_params(self):
        if (
            not np.issubdtype(type(self.n_components), np.integer)
            or self.n_components < 2
        ):
            raise ValueError(
                "n_components must be and integer greater than or equal " "to 2."
            )

    def _validate_data(self, X):
        try:
            assert np.isscalar(X[0][0][0])
        except:
            raise ValueError("Input must be a collection of collections of points")

        try:
            dims = [np.array(x).shape[1] for x in X]
        except:
            raise ValueError(
                "Elements of each point collection must be of the same dimension."
            )

        if not hasattr(self, "data_dimension_"):
            self.data_dimension_ = np.mean(dims)

        if not (
            np.max(dims) == self.data_dimension_ or np.min(dims) == self.data_dimension_
        ):
            raise ValueError("Each point collection must be of equal dimension.")

    def fit(self, X, y=None, **fit_params):
        random_state = check_random_state(self.random_state)
        self._validate_params()
        self._validate_data(X)

        combined_data = np.vstack(X)
        combined_data = check_array(combined_data)

        self.mixture_model_ = GaussianMixture(
            n_components=self.n_components, random_state=random_state
        )
        self.mixture_model_.fit(combined_data)
        self.ground_distance_ = pairwise_gaussian_ground_distance(
            self.mixture_model_.means_,
            self.mixture_model_.covariances_,
        )
        self.metric_ = distances.hellinger

    def transform(self, X):
        check_is_fitted(self, ["mixture_model_", "ground_distance_"])
        self._validate_data(X)
        result = np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return np.vstack(
            [vectorize_diagram(diagram, self.mixture_model_) for diagram in X]
        )


def find_bin_boundaries(flat, n_bins):
    """
    Only uniform distribution is currently implemented.
    TODO: Implement Normal
    :param flat: an iterable.
    :param n_bins:
    :return:
    """
    flat.sort()
    flat_csum = np.cumsum(flat)
    bin_range = flat_csum[-1] / n_bins
    bin_indices = [0]
    for i in range(1, len(flat_csum)):
        if (flat_csum[i] >= bin_range * len(bin_indices)) & (
            flat[i] > flat[bin_indices[-1]]
        ):
            bin_indices.append(i)
    bin_values = np.array(flat, dtype=float)[bin_indices]

    if bin_values.shape[0] < n_bins:
        warn(
            f"Could not generate n_bins={n_bins} bins as there are not enough "
            f"distinct values. Please check your data."
        )

    return bin_values


def expand_boundaries(my_interval_index, absolute_range):
    """
    Expands the outer bind on a pandas IntervalIndex to encompase the range specified by the 2-tuple absolute_range.

    Parameters
    ----------
    my_interval_index: pandas IntervalIndex object (right closed)
    absolute_range: 2-tuple.
        (min_value, max_value)

    Returns
    -------
    index: a pandas IntervalIndex
        A pandas IntervalIndex with the boundaries potentially expanded to encompas the absolute range.
    """
    """
    expands the outer bind on a pandas IntervalIndex to encompase the range specified by the 2-tuple absolute_range
    :param my_interval_index:
    :param absolute_range: 2tuple 
    :return: a pandas IntervalIndex
    """
    interval_list = my_interval_index.to_list()
    # Check if the left boundary needs expanding
    if interval_list[0].left > absolute_range[0]:
        interval_list[0] = pd.Interval(
            left=absolute_range[0], right=interval_list[0].right
        )
    # Check if the right boundary needs expanding
    last = len(interval_list) - 1
    if interval_list[last].right < absolute_range[1]:
        interval_list[last] = pd.Interval(
            left=interval_list[last].left, right=absolute_range[1]
        )
    return pd.IntervalIndex(interval_list)


def add_outier_bins(my_interval_index, absolute_range):
    """
    Appends extra bins to either side our our interval index if appropriate.
    That only occurs if the absolute_range is wider than the observed range in your training data.
    :param my_interval_index:
    :param absolute_range:
    :return:
    """
    interval_list = my_interval_index.to_list()
    # Check if the left boundary needs expanding
    if interval_list[0].left > absolute_range[0]:
        left_outlier = pd.Interval(left=absolute_range[0], right=interval_list[0].left)
        interval_list.insert(0, left_outlier)

    last = len(interval_list) - 1
    if interval_list[last].right < absolute_range[1]:
        right_outlier = pd.Interval(
            left=interval_list[last].right, right=absolute_range[1]
        )
        interval_list.append(right_outlier)
    return pd.IntervalIndex(interval_list)


class HistogramVectorizer(BaseEstimator, TransformerMixin):
    """Convert a time series of binary events into a histogram of
    event occurrences over a time frame. If the data has explicit time stamps
    it can be aggregated over hour of day, day of week, day of month, day of year
    , week of year or month of year.

    Parameters
    ----------
    n_components: int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if n_bins < 2.

    strategy: {‘uniform’, ‘quantile’, 'gmm'}, (default=’quantile’)
        The method to use for bin selection in the histogram. In general the
        quantile option, which will select variable width bins based on the
        distribution of the training data, is suggested, but uniformly spaced
        identically sized bins, or soft gns learned from a Gaussian mixture model
        are also available.

    ground_distance: {'euclidean'}
        The distance to induce between bins.

    absolute_range: (minimum_value_possible, maximum_value_possible) (default=(-np.inf, np.inf))
        By default values outside of training data range are included in the extremal bins.
        You can specify these values if you know something about your values (e.g. (0, np.inf) )

    append_outlier_bins: bool (default=False)
        Whether to add extra bins to catch values outside of your training
        data where appropriate? These bins will increase the total number of
        components (to ``n_components + 2`` and will be the first bin (for
        outlying small data) and the last bin (for outlying large data).
    """

    # TODO: time stamps, generic groupby
    def __init__(
        self,
        n_components=20,
        strategy="uniform",
        ground_distance="euclidean",
        absolute_range=(-np.inf, np.inf),
        append_outlier_bins=False,
    ):
        self.n_components = n_components
        self.strategy = strategy
        self.ground_distance = ground_distance  # Not currently making use of this.
        self.absolute_range = absolute_range
        self.append_outlier_bins = append_outlier_bins

    def _validate_params(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Learns the histogram bins.
        Still need to check switch.
        :param X:
        :return:
        """
        flat = flatten(X)
        flat = list(
            filter(
                lambda n: n > self.absolute_range[0] and n < self.absolute_range[1],
                flat,
            )
        )
        if self.strategy == "uniform":
            self.bin_intervals_ = pd.interval_range(
                start=np.min(flat), end=np.max(flat), periods=self.n_components
            )
        if self.strategy == "quantile":
            self.bin_intervals_ = pd.IntervalIndex.from_breaks(
                find_bin_boundaries(flat, self.n_components)
            )
        if self.append_outlier_bins == True:
            self.bin_intervals_ = add_outier_bins(
                self.bin_intervals_, self.absolute_range
            )
        else:
            self.bin_intervals_ = expand_boundaries(
                self.bin_intervals_, self.absolute_range
            )
        self.metric_ = distances.hellinger
        return self

    def _vector_transform(self, vector):
        """
        Applies the transform to a single row of the data.
        """
        return pd.cut(vector, self.bin_intervals_).value_counts()

    def transform(self, X):
        """
        Apply binning to a full data set returning an nparray.
        """
        check_is_fitted(self, ["bin_intervals_"])
        result = np.ndarray((len(X), len(self.bin_intervals_)))
        for i, seq in enumerate(X):
            result[i, :] = self._vector_transform(seq).values
        return result


def temporal_cyclic_transform(datetime_series, periodicity=None):
    """
    TODO: VERY UNFINISHED
    Replaces all time resolutions above the resolution specified with a fixed value.
    This creates a cycle within a datetime series.
    Parameters
    ----------
    datetime_series: a pandas series of datetime objects
    periodicity: string ['year', 'month' , 'week', 'day', 'hour']
        What time period to create cycles.

    Returns
    -------
    cyclic_series: pandas series of datetime objects

    """
    collapse_times = {}
    if periodicity in ["year", "month", "day", "hour"]:
        collapse_times["year"] = 1970
        if periodicity in ["month", "day", "hour"]:
            collapse_times["month"] = 1
            if periodicity in ["day", "hour"]:
                collapse_times["day"] = 1
                if periodicity in ["hour"]:
                    collapse_times["hour"] = 0
        cyclic_series = datetime_series.apply(lambda x: x.replace(**collapse_times))
    elif periodicity == "week":
        raise NotImplementedError("we have not implemented week cycles yet")
    else:
        raise ValueError(
            f"Sorry resolution={periodicity} is not a valid option.  "
            + f"Please select from ['year', 'month', 'week', 'day', 'hour']"
        )
    return cyclic_series


class CyclicHistogramVectorizer(BaseEstimator, TransformerMixin):
    """"""

    def __init__(
        self,
        periodicity="week",
        resolution="day",
    ):
        self.periodicity = periodicity
        self.resolution = resolution

    def _validate_params(self):
        pass

    def fit(self, X, y=None, **fit_params):
        cyclic_data = temporal_cyclic_transform(
            pd.to_datetime(X), periodicity=self.periodicity
        )
        resampled = (
            pd.Series(index=cyclic_data, data=1).resample(self.resolution).count()
        )
        self.temporal_bins_ = resampled.index
        return self


class ProductDistributionVectorizer(BaseEstimator, TransformerMixin):
    pass
