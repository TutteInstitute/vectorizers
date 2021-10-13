import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from vectorizers._window_kernels import _SLIDING_WINDOW_KERNELS


@numba.njit(nogil=True)
def sliding_windows(sequence, width, stride, sample, kernel, pad_width=0, pad_value=0):

    if pad_width > 0:
        new_shape = (2 * pad_width + sequence.shape[0], *sequence.shape[1:])
        new_sequence = np.full(tuple(new_shape), pad_value, dtype=sequence.dtype)
        new_sequence[pad_width : pad_width + sequence.shape[0]] = sequence
        sequence = new_sequence

    last_window_start = sequence.shape[0] - width + 1

    n_rows = int(np.ceil(last_window_start / stride))
    n_cols = kernel.shape[0]
    for i, size in enumerate(sequence.shape):
        if i == 0:
            continue
        n_cols *= size

    result = np.empty((n_rows, n_cols), dtype=np.float64)
    if sample.shape[0] < width:
        for i in range(n_rows):
            result[i] = np.asarray(
                kernel
                @ (sequence[i * stride : i * stride + width])[sample].astype(np.float64)
            ).flatten()
    else:
        for i in range(n_rows):
            result[i] = np.asarray(
                kernel @ (sequence[i * stride : i * stride + width]).astype(np.float64)
            ).flatten()

    return result


def build_kernel(kernel_list, window_size):

    result = np.eye(window_size, dtype=np.float64)

    if kernel_list is None or len(kernel_list) < 1:
        return result

    for kernel in kernel_list:
        if type(kernel) in (tuple, list):
            kernel, *kernel_params = kernel
        else:
            kernel_params = ()

        if type(kernel) == np.ndarray:
            if kernel.shape[1] != result.shape[0]:
                raise ValueError(
                    "Kernel specified as ndarray in kernel_list does not have the"
                    "right shape to occupy this slot in the kernel list!"
                )

            result = kernel @ result
        elif kernel in _SLIDING_WINDOW_KERNELS:
            result = (
                _SLIDING_WINDOW_KERNELS[kernel](result.shape[0], *kernel_params)
                @ result
            )
        else:
            raise ValueError(f"Unrecognized kernel {kernel}")

    if result.ndim < 2:
        result = result[None, :]

    return result


def sliding_window_generator(
    sequences,
    window_width=10,
    window_stride=1,
    window_sample=None,
    kernels=None,
    pad_width=0,
    pad_value=0,
):
    if window_sample is None:
        window_sample_ = np.arange(window_width)
    else:
        window_sample_ = np.asarray(window_sample, dtype=np.int32)

    kernel_ = build_kernel(kernels, window_sample_.shape[0])

    for sequence in sequences:
        yield sliding_windows(
            np.asarray(sequence),
            window_width,
            window_stride,
            window_sample_,
            kernel_,
            pad_width,
            pad_value,
        )

    return


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    """Convert numeric sequence data into point clouds by applying sliding
    windows over the data. This is applicable to things like time-series and
    can be viewed as a Taken's embedding of each time series. This approach
    can usefully be paired with WassersteinVectorizer, SinkhornVectorizer, or
    DistributionVectorizer to turn the point clouds in vectors approximating
    Wasserstein-like distances between the point clouds.

    Parameters
    ----------
    window_width: int (optional, default=10)
        How large of a window to use. This will determine the dimensionality of the
        vector space in which the resulting point clouds will live unless a window
        sample is specified.

    window_stride: int (optional, default=1)
        How far to step along when sliding the window. Setting ``window_stride``
        to the same value as ``window_width`` will ensure non-overlapping windows. The
        default of 1 will generate the maximum number of points in the resulting point
        cloud.

    window_sample: None, int, pair of ints, "random", or 1d array of integers (optional, default=None)
        How to sample from each window. The default on None will simply rake the whole
        window. If an int ``n`` is given this will be be used as a stride sampling every
        ``n``th entry of the window. If a pair of integers ``(n, m)`` this will be
        used as a start and stride sampling every ``m``th entry starting fron the ``n``th
        entry. If "random" is given then a random sampling on ``window_sample_size`` indices
        in the range ``(0, window_width)`` will be used. Finally if an array of integers
        are given this will be used as the selected indices to take from each window.

    window_sample_size: int (optional, default=0)
        If using random sampling from a window this will determine he size of the random sample.
    """

    def __init__(
        self,
        window_width=10,
        window_stride=1,
        window_sample=None,
        window_sample_size=0,
        kernels=None,
        pad_width=0,
        pad_value=0,
    ):

        self.window_width = window_width
        self.window_stride = window_stride
        self.window_sample = window_sample
        self.window_sample_size = window_sample_size
        self.kernels = kernels
        self.pad_width = pad_width
        self.pad_value = pad_value

    def fit(self, X, y=None, **fit_params):
        """
        Given a list of numeric sequences, prepare for
        conversion into a list of point clouds under a
        sliding window embedding.

        Parameters
        ----------
        X: list of array-like
            The input data to be transformed.
        """
        if self.window_sample is None:
            self.window_sample_ = np.arange(self.window_width)
        elif self.window_sample == "random":
            self.window_sample_ = np.random.choice(
                self.window_width, size=self.window_sample_size, replace=False
            )
        elif np.issubdtype(type(self.window_sample), np.integer):
            self.window_sample_ = np.arange(self.window_width, self.window_sample)
        elif type(self.window_sample) in (list, tuple) and len(self.window_sample) == 2:
            start, stride = self.window_sample
            if np.issubdtype(type(start), np.integer) and np.issubdtype(
                type(stride), np.integer
            ):
                self.window_sample_ = np.arange(start, self.window_width, stride)
            else:
                raise ValueError(
                    "If passing a length 2 tuple of start and stride for "
                    "window sample, start and stride must be integers."
                )
        else:
            # Check is we can convert it to an array
            try:
                self.window_sample_ = np.asarray(self.window_sample, dtype=np.int32)
                assert self.window_sample_.ndim == 1
            except:
                raise ValueError(
                    """window_sample should be one of:
                * None
                * an integer stride value
                * a tuple of integers (start, stride)
                * the string "random"
                * a 1d array (or array like) of integer indices to sample
                """
                )

        if self.window_width < 0:
            raise ValueError("window_width must be positive")

        if self.kernels is not None and type(self.kernels) not in (list, tuple):
            raise ValueError("kernels must be None or a list or tuple of kernels to apply")

        self.kernel_ = build_kernel(self.kernels, self.window_sample_.shape[0])

        return self

    def transform(self, X, y=None):
        """
        Given a list of numeric sequences, convert into a list
        of point clouds under a sliding window embedding.

        Parameters
        ----------
        X: list of array-like
            The input data to be transformed.

        Returns
        -------
        result: list of lists of ndarrays
            Each input array like is converted to a list of ndarrays.
        """
        check_is_fitted(self, ["window_sample_", "kernel_"])

        result = []
        for sequence in X:
            result.append(
                sliding_windows(
                    np.asarray(sequence),
                    self.window_width,
                    self.window_stride,
                    self.window_sample_,
                    self.kernel_,
                    self.pad_width,
                    self.pad_value,
                )
            )

        return result
