import numpy as np
import numba

EPSILON = 1e-8

# The window function


@numba.njit(nogil=True)
def window_at_index(token_sequence, window_size, ind, reverse=False):
    if reverse:
        return np.flip(token_sequence[max(ind - window_size, 0) : ind])
    return token_sequence[ind + 1 : min(ind + window_size + 1, len(token_sequence))]


# Window width functions


@numba.njit(nogil=True)
def variable_window_radii(
    window_size,
    token_frequency,
    mask_index=None,
    power=0.75,
):
    radii = np.power(token_frequency, power - 1)
    radii /= np.sum(radii * token_frequency)
    radii = np.append(radii, min(radii))
    if mask_index is not None:
        radii[mask_index] = 0.0
    result = radii * window_size
    result[(result > 0) * (result < 1)] = 1.0
    np.round(result, 0, result)
    return result.astype(np.int64)


@numba.njit(nogil=True)
def fixed_window_radii(window_size, token_frequency, mask_index=None):
    radii = np.repeat(window_size, len(token_frequency) + 1)
    if mask_index is not None:
        radii[mask_index] = 0.0
    return radii


# Kernel functions


@numba.njit(nogil=True)
def flat_kernel(window, mask_index=None, normalize=False, offset=0):
    result = np.ones(len(window), dtype=np.float64)
    if mask_index is not None:
        result[window == mask_index] = 0.0
    result[0 : min(offset, len(result))] = 0
    if normalize:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result


@numba.njit(nogil=True)
def harmonic_kernel(window, mask_index=None, normalize=False, offset=0):
    result = 1.0 / np.arange(1, len(window) + 1)
    if mask_index is not None:
        result[window == mask_index] = 0.0
    result[0 : min(offset, len(result))] = 0
    if normalize:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result


@numba.njit(nogil=True)
def geometric_kernel(
    window,
    mask_index=None,
    normalize=False,
    offset=0,
    power=0.9,
):
    result = power ** np.arange(0, len(window))

    if mask_index is not None:
        result[window == mask_index] = 0.0
    result[0 : min(offset, len(result))] = 0
    if normalize:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result


@numba.njit(nogil=True)
def update_kernel(
    window,
    kernel,
    mask_index,
    normalize,
):
    result = kernel[: len(window)].astype(np.float64)
    if mask_index is not None:
        result[window == mask_index] = 0
    if normalize:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result


# Parameter lists

_WINDOW_FUNCTIONS = {
    "variable": variable_window_radii,
    "fixed": fixed_window_radii,
}

_KERNEL_FUNCTIONS = {
    "flat": flat_kernel,
    "harmonic": harmonic_kernel,
    "geometric": geometric_kernel,
}

####################################################
# Sliding window multivariate time series kernels
####################################################


def averaging_kernel(n_cols, *kernel_params):
    return np.full(n_cols, 1.0 / n_cols)


def difference_kernel(n_cols, start, step, stride, *kernel_params):
    n_differences = int(np.ceil((n_cols - start - step) // stride))
    result = np.zeros((n_differences, n_cols))
    for i in range(n_differences):
        result[i, start + i * stride] = -1
        result[i, start + i * stride + step] = 1

    return result


def positon_velocity_kernel(n_cols, position_index, step, stride, *kernel_params):
    n_differences_before = int(np.ceil((position_index - step) // stride))
    n_differences_after = int(np.ceil((n_cols - position_index - step) // stride))
    n_differences = n_differences_before + n_differences_after
    result = np.zeros((n_differences + 1, n_cols))
    result[0, position_index] = 1
    for i in range(n_differences_before):
        result[i + 1, position_index - i * stride] = 1
        result[i + 1, position_index - i * stride - step] = -1
    for i in range(n_differences_after):
        result[i + n_differences_before + 1, position_index + i * stride] = -1
        result[i + n_differences_before + 1, position_index + i * stride + step] = 1

    return result


def weight_kernel(n_cols, weights, *kernel_params):
    if weights.shape[0] != n_cols:
        raise ValueError(
            f"Cannot construct a weight kernel of size {n_cols} "
            f"with weights of shape {weights.shape[0]}"
        )

    return np.diag(weights)


def gaussian_weight_kernel(n_cols, sigma, *kernel_params):
    width = n_cols / 2
    xs = np.linspace(-width, width, n_cols)
    weights = 1.0 / (sigma * 2 * np.pi) * np.exp(-((xs / sigma) ** 2) / 2.0)
    return np.diag(weights)


_SLIDING_WINDOW_KERNELS = {
    "average": averaging_kernel,
    "differences": difference_kernel,
    "position_velocity": positon_velocity_kernel,
    "weight": weight_kernel,
    "gaussian_weight": gaussian_weight_kernel,
}

# Copied from the SciPy implementation
@numba.njit()
def binom(n, k):
    n = int(n)
    k = int(k)

    if k > n or n < 0 or k < 0:
        return 0

    m = n + 1
    nterms = min(k, n - k)

    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= m - j
        denominator *= j

    return numerator // denominator

# A couple of changepoint based kernels that can be useful. The goal
# is to detect changepoints in seuquences of count of time interval
# data (where the intervals are between events).
#
# We can model count data with Poisson's and interval data as inter-arrival
# times (which can can convert to count-like data by taking reciprocals.
#
# Essentially we start with a baseline prior given by a gamma distribution,
# and then update the prior with the data in the window up to, but not
# including, the last element. The return value is then the predictive
# posterior (a negative binomial) of observing the final element of
# the window.

def count_changepoint_kernel(alpha=1.0, beta=1):
    @numba.njit()
    def _kernel(window):
        model_window = window[:-1]
        observation = window[-1]
        alpha_prime = alpha + model_window.sum()
        beta_prime = beta + len(model_window)
        nb_r = alpha_prime
        nb_p = 1.0 / (1.0 + beta_prime)

        prob = binom(observation + nb_r - 1, observation) * (1 - nb_p) ** nb_r * nb_p ** observation

        return np.array([-np.log(prob)])

    return _kernel

def inter_arrival_changepoint_kernel(alpha=1.0, beta=1):
    @numba.njit()
    def _kernel(window):
        model_window = 1.0 / (window[:-1] + EPSILON)
        observation = 1.0 / (window[-1] + EPSILON)
        alpha_prime = alpha + model_window.sum()
        beta_prime = beta + len(model_window)
        nb_r = alpha_prime
        nb_p = 1.0 / (1.0 + beta_prime)

        prob = binom(observation + nb_r - 1, observation) * (1 - nb_p) ** nb_r * nb_p ** observation

        return np.array([-np.log(prob)])

    return _kernel

_SLIDING_WINDOW_FUNCTION_KERNELS = {
    "count_changepoint" : count_changepoint_kernel,
    "timespan_changepoint": inter_arrival_changepoint_kernel,
}