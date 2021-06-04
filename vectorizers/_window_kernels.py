import numpy as np
import numba

from vectorizers.utils import flatten


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
