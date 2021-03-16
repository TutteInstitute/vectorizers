import numpy as np
import numba

# The window function


@numba.njit(nogil=True, inline="always")
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
    return np.ceil(radii * window_size)


@numba.njit(nogil=True)
def fixed_window_radii(window_size, token_frequency, mask_index=None):
    radii = np.repeat(window_size, len(token_frequency) + 1)
    if mask_index is not None:
        radii[mask_index] = 0.0
    return radii


# Kernel functions


@numba.njit(nogil=True)
def flat_kernel(window, window_size, mask_index=None, normalized=False):
    result = np.ones(len(window), dtype=np.float32)
    if mask_index is not None:
        result[window == mask_index] = 0.0
    if normalized:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result.astype(np.float32)


@numba.njit(nogil=True)
def triangle_kernel(
    window,
    window_size,
    mask_index=None,
    normalized=False,
):
    start = max(window_size, len(window))
    stop = window_size - len(window)
    result = np.arange(start, stop, -1).astype(np.float32)
    if mask_index is not None:
        result[window == mask_index] = 0.0
    if normalized:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result.astype(np.float32)


@numba.njit(nogil=True)
def harmonic_kernel(window, window_size, mask_index=None, normalized=False):
    result = 1.0 / np.arange(1, len(window) + 1).astype(np.float32)
    if mask_index is not None:
        result[window == mask_index] = 0.0
    if normalized:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result.astype(np.float32)


@numba.njit(nogil=True)
def negative_binomial_kernel(
    window,
    window_size,
    mask_index=None,
    normalized=False,
    power=0.9,
):
    result = (1 - power) * (power ** np.arange(0, len(window), dtype=np.float32))
    if mask_index is not None:
        result[window == mask_index] = 0.0
    if normalized:
        temp = result.sum()
        if temp > 0:
            result /= temp
    return result.astype(np.float32)


# Parameter lists

_WINDOW_FUNCTIONS = {
    "variable": variable_window_radii,
    "fixed": fixed_window_radii,
}

_KERNEL_FUNCTIONS = {
    "flat": flat_kernel,
    "harmonic": harmonic_kernel,
    "triangular": triangle_kernel,
    "negative_binomial": negative_binomial_kernel,
}
