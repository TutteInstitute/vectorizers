import numpy as np
import numba

# Window functions


@numba.njit(nogil=True)
def information_window(token_sequence, window_size, token_frequency):
    result = []

    for i in range(len(token_sequence)):
        counter = 0
        current_entropy = 0.0

        for j in range(i + 1, len(token_sequence)):
            current_entropy -= np.log(token_frequency[int(token_sequence[j])])
            counter += 1
            if current_entropy >= window_size:
                break

        result.append(token_sequence[i + 1 : i + 1 + counter])

    return result


@numba.njit(nogil=True)
def fixed_window(token_sequence, window_size, token_frequency):
    result = []
    for i in range(len(token_sequence)):
        result.append(token_sequence[i + 1 : i + window_size + 1])

    return result


@numba.njit(nogil=True)
def masked_fixed_window(token_sequence, window_size, token_frequency):
    result = []
    for i in range(len(token_sequence)):
        if token_sequence[i] == len(token_frequency):
            result.append(token_sequence[i:i])
        else:
            result.append(
                token_sequence[i + 1 : min(i + window_size + 1, len(token_sequence))]
            )
    return result


@numba.njit(nogil=True)
def masked_fixed_window_reverse(token_sequence, window_size, token_frequency):
    result = []
    for i in range(len(token_sequence)):
        if token_sequence[i] == len(token_frequency):
            result.append(token_sequence[i:i])
        else:
            result.append(
                np.flip(token_sequence[max(i - window_size, 0) : i]).astype(np.int64)
            )
    return result


@numba.njit(nogil=True)
def masked_variable_window(token_sequence, window_size, token_frequency, power=0.25):
    result = []
    keeps = 1 / np.power(token_frequency, power)
    keeps /= np.sum(keeps * token_frequency)

    for i in range(len(token_sequence)):
        if token_sequence[i] == len(token_frequency):
            size = 0
        else:
            size = np.ceil(keeps[int(token_sequence[i])] * window_size)
        result.append(token_sequence[i + 1 : min(i + size + 1, len(token_sequence))])
    return result


@numba.njit(nogil=True)
def masked_variable_window_reverse(
    token_sequence, window_size, token_frequency, power=0.25
):
    result = []
    keeps = 1 / np.power(token_frequency, power)
    keeps /= np.sum(keeps * token_frequency)

    for i in range(len(token_sequence)):
        if token_sequence[i] == len(token_frequency):
            size = 0
        else:
            size = np.ceil(keeps[int(token_sequence[i])] * window_size)
        result.append(
            np.flip(token_sequence[max(i - size, 0) : i]).astype(token_sequence.dtype)
        )
    return result


@numba.njit(nogil=True)
def variable_window(token_sequence, window_size, token_frequency, power=0.25):
    result = []
    keeps = 1 / np.power(token_frequency, power)
    keeps /= np.sum(keeps * token_frequency)

    for i in range(len(token_sequence)):
        size = np.ceil(keeps[int(token_sequence[i])] * window_size)
        result.append(token_sequence[i + 1 : min(i + size + 1, len(token_sequence))])
    return result


@numba.njit(nogil=True)
def variable_window_reverse(token_sequence, window_size, token_frequency, power=0.25):
    result = []
    keeps = 1 / np.power(token_frequency, power)
    keeps /= np.sum(keeps * token_frequency)

    for i in range(len(token_sequence)):
        size = np.ceil(keeps[int(token_sequence[i])] * window_size)
        result.append(
            np.flip(token_sequence[max(i - size, 0) : i]).astype(token_sequence.dtype)
        )
    return result


# Kernel Functions


@numba.njit(nogil=True)
def flat_kernel(window, window_size, token_frequency):
    return np.ones(len(window), dtype=np.float32)


@numba.njit(nogil=True)
def masked_flat_kernel(window, window_size, token_frequency):
    temp = np.ones(len(window), dtype=np.float32)
    temp[window == len(token_frequency)] = 0.0
    return temp


@numba.njit(nogil=True)
def triangle_kernel(window, window_size, token_frequency):
    start = max(window_size, len(window))
    stop = window_size - len(window)
    return np.arange(start, stop, -1).astype(np.float32)


@numba.njit(nogil=True)
def harmonic_kernel(window, window_size, token_frequency):
    result = np.arange(1, len(window) + 1).astype(np.float32)
    return 1.0 / result


@numba.njit(nogil=True)
def negative_binomial_kernel(window, window_size, token_frequency, p=0.9):
    result = np.arange(0, len(window)).astype(np.float32)
    return (1 - p) * (p ** result)


@numba.njit(nogil=True)
def masked_negative_binomial_kernel(window, window_size, token_frequency, p=0.9):
    temp = (1 - p) * (p ** np.arange(0, len(window)).astype(np.float32))
    temp[window == len(token_frequency)] = 0.0
    return temp


_WINDOW_FUNCTIONS = {
    "information": information_window,
    "fixed": fixed_window,
    "masked fixed": masked_fixed_window,
    "masked fixed reverse": masked_fixed_window_reverse,
    "variable": variable_window,
    "variable reverse": variable_window_reverse,
    "masked variable": masked_variable_window,
    "masked variable reverse": masked_variable_window_reverse,
}

_KERNEL_FUNCTIONS = {
    "flat": flat_kernel,
    "harmonic": harmonic_kernel,
    "triangular": triangle_kernel,
    "negative binomial": negative_binomial_kernel,
    "masked flat": masked_flat_kernel,
    "masked negative binomial": masked_negative_binomial_kernel,
}
