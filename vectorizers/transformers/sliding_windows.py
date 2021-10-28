import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from vectorizers._window_kernels import _SLIDING_WINDOW_KERNELS, _SLIDING_WINDOW_FUNCTION_KERNELS


@numba.njit(nogil=True)
def sliding_windows(
    sequence,
    width,
    stride,
    sample,
    kernel,
    kernel_output_size,
    kernel_output_dtype,
    pad_width=0,
    pad_value=0,
):

    if pad_width > 0:
        new_shape = (2 * pad_width + sequence.shape[0], *sequence.shape[1:])
        new_sequence = np.full(tuple(new_shape), pad_value, dtype=sequence.dtype)
        new_sequence[pad_width : pad_width + sequence.shape[0]] = sequence
        sequence = new_sequence

    last_window_start = sequence.shape[0] - width + 1

    n_rows = int(np.ceil(last_window_start / stride))
    n_cols = kernel_output_size

    result = np.empty((n_rows, n_cols), dtype=kernel_output_dtype)
    if sample.shape[0] < width:
        for i in range(n_rows):
            result[i] = kernel(sequence[i * stride : i * stride + width][sample])
            # result[i] = np.asarray(
            #     kernel
            #     @ (sequence[i * stride : i * stride + width])[sample].astype(np.float64)
            # ).flatten()
    else:
        for i in range(n_rows):
            result[i] = kernel(sequence[i * stride : i * stride + width])
            # result[i] = np.asarray(
            #     kernel @ (sequence[i * stride : i * stride + width]).astype(np.float64)
            # ).flatten()

    return result


def build_matrix_kernel(kernel_list, window_size, sequence_shape):

    result = np.eye(window_size, dtype=np.float64)

    if kernel_list is not None and len(kernel_list) >= 1:

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

    n_cols = result.shape[0]
    for i, size in enumerate(sequence_shape):
        if i == 0:
            continue
        n_cols *= size

    @numba.njit(nogil=True, fastmath=True)
    def _kernel_func(data):
        return np.asarray(result @ data.astype(np.float64)).flatten()

    return _kernel_func, n_cols, np.float64

def build_callable_kernel(kernel_list, test_window):

    tuple_of_kernels = []
    for kernel in kernel_list:
        if type(kernel) in (tuple, list):
            kernel, *kernel_params = kernel
        else:
            kernel_params = ()

        if type(kernel) == str and kernel in _SLIDING_WINDOW_FUNCTION_KERNELS:
            tuple_of_kernels.append(_SLIDING_WINDOW_FUNCTION_KERNELS[kernel](*kernel_params))
        elif callable(kernel):
            tuple_of_kernels.append(kernel)
        else:
            raise ValueError(f"Bad kernel {kernel} in kernel list")

    tuple_of_kernels = tuple(tuple_of_kernels)

    if len(tuple_of_kernels) == 1:
        _kernel_func = tuple_of_kernels[0]
    else:
        @numba.njit(nogil=True)
        def _kernel_func(data):
            result = data
            for kernel in tuple_of_kernels:
                result = kernel(result)
            return result

    kernel_output = _kernel_func(test_window)

    return _kernel_func, kernel_output.shape[0], kernel_output.dtype

def check_function_kernels(kernels):

    if kernels is None or len(kernels) < 1:
        return False

    for kernel in kernels:
        if callable(kernel):
            return True

        if type(kernel) in (tuple, list):
            kernel, *kernel_params = kernel
        else:
            kernel_params = ()

        if type(kernel) is str and kernel in _SLIDING_WINDOW_FUNCTION_KERNELS:
            return True
    else:
        return False


def sliding_window_generator(
    sequences,
    sequence_shape,
    window_width=10,
    window_stride=1,
    window_sample=None,
    kernels=None,
    pad_width=0,
    pad_value=0,
    test_window=None,
):
    """Return a generator of sliding windows of sequences. Importantly this
    function allows the sequences themselves to be a generator, ensuring that
    the full set of sequences does not need to be a list in memory as long
    as the next element can be created on demand via a generator.

    Parameters
    ----------
    sequences: iterable of iterables
        The sequences over which to generate sets of sliding windows.

    sequence_shape: tuple
        The shape of a given sequence as an ndarray; this allows results
        space to be allocated without having to touch the generator.

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
        If using random sampling from a window this will determine the size of the random sample.

    kernels: list of kernel descriptors or None (optional, default=None)
        If None then no kernels are applied. Otherwise this should be
        the sequence of kernels to be applied to each window. This can be named kernels
        such as:
            * "average"
            * "differences"
            * "position_velocity"
            * "weight"
            * "gaussian_weight"
        The named kernels often take parameters; to pass these supply the kernel as
        a tuple ``("name", param1, param2, ...)``. Check the named kernels for details
        on params.
        Alternatively a kernel can be an (appropriate shaped!) transformation matrix in
        the form of an ndarray which will be left multiply against the window to be applied.
        Finally kernels may also be (numba jitted) functions. Mixing functions and named
        kernels is currently not supported.

    pad_width: int (optional, default=0)
        How much to pad the start and end of each sequence by. This can be useful for
        centering a window on starting and ending points of the sequence for example.

    pad_value: int or float (optional, default=0.0)
        If the sequence is to be padded, use this value as the padding value.

    test_window: ndarray of shape (window_sample_size,) or None (optional, default=None)
        If using callable function kernels it is necessary to provide a single test window
        to allow the code to determine the size and type of the output of the kernels when
        applied to a window. The use of the test window ensures that the sequences generator
        doesn't need to be touched.

    Returns
    -------
    windows_per_sequence: generator
        A generator that returns the windows for each sequence from the
        sequences passed in.
    """
    if window_sample is None:
        window_sample_ = np.arange(window_width)
    else:
        window_sample_ = np.asarray(window_sample, dtype=np.int32)

    if check_function_kernels(kernels):
        if test_window is None:
            raise ValueError("Callable kernels need to also provide a test sequence to "
                             "determine kernel output size and type")
        kernel_, kernel_output_size, kernel_output_dtype = build_callable_kernel(
            kernels, test_window
        )
    else:
        kernel_, kernel_output_size, kernel_output_dtype = build_matrix_kernel(
            kernels, window_sample_.shape[0], sequence_shape
        )

    for sequence in sequences:
        yield sliding_windows(
            np.asarray(sequence),
            window_width,
            window_stride,
            window_sample_,
            kernel_,
            kernel_output_size,
            kernel_output_dtype,
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

    kernels: list of kernel descriptors or None (optional, default=None)
        If None then no kernels are applied. Otherwise this should be
        the sequence of kernels to be applied to each window. This can be named kernels
        such as:
            * "average"
            * "differences"
            * "position_velocity"
            * "weight"
            * "gaussian_weight"
        The named kernels often take parameters; to pass these supply the kernel as
        a tuple ``("name", param1, param2, ...)``. Check the named kernels for details
        on params.
        Alternatively a kernel can be an (appropriate shaped!) transformation matrix in
        the form of an ndarray which will be left multiply against the window to be applied.
        Finally kernels may also be (numba jitted) functions. Mixing functions and named
        kernels is currently not supported.

    pad_width: int (optional, default=0)
        How much to pad the start and end of each sequence by. This can be useful for
        centering a window on starting and ending points of the sequence for example.

    pad_value: int or float (optional, default=0.0)
        If the sequence is to be padded, use this value as the padding value.
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
            raise ValueError(
                "kernels must be None or a list or tuple of kernels to apply"
            )

        use_function_kernels = check_function_kernels(self.kernels)

        if use_function_kernels:
            test_window = np.asarray(X[0])[:self.window_width][self.window_sample_]
            (
                self.kernel_,
                self.kernel_output_size_,
                self.kernel_output_dtype_,
            ) = build_callable_kernel(
                self.kernels, test_window
            )
        else:
            (
                self.kernel_,
                self.kernel_output_size_,
                self.kernel_output_dtype_,
            ) = build_matrix_kernel(
                self.kernels, self.window_sample_.shape[0], np.asarray(X[0]).shape
            )

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
                    self.kernel_output_size_,
                    self.kernel_output_dtype_,
                    self.pad_width,
                    self.pad_value,
                )
            )

        return result


class SequentialDifferenceTransformer(BaseEstimator, TransformerMixin):
    """Generate a sequence of sequential differences for each sequence in
    a list of numeric sequences. This can be useful for converting a sequence
    of timestamps into a sequence of inter-arrival times for example.

    Parameters
    ----------
    stride: int (optional, default=1)
        How far apart to take differences in the sequence.
    """

    def __init__(self, stride=1):
        self.stride = stride

    def fit(self, X, y=None, **fit_params):
        self._sliding_window_transformer = SlidingWindowTransformer(
            window_width=self.stride + 1,
            kernels=[("differences", 0, self.stride, self.stride)],
        )
        self._sliding_window_transformer.fit(X, y=None, **fit_params)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["_sliding_window_transformer"])
        return self._sliding_window_transformer.transform(X)
