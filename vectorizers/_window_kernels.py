import numpy as np
import numba


@numba.njit(nogil=True)
def information_window(token_sequence, window_size, token_frequency, reverse=False):
    result = []

    for i in range(len(token_sequence)):
        counter = 0
        current_entropy = 0.0

        if reverse:
            for j in range(i - 1, 0, -1):
                next_entropy = -np.log2(token_frequency[int(token_sequence[j])])
                next_entropy = max(window_size / 3.0, next_entropy)
                if next_entropy < 1.0e-2:
                    next_entropy = 0.0
                current_entropy += next_entropy
                counter += 1
                if current_entropy >= window_size:
                    break

            result.append(token_sequence[i - 1 : max(0, i - counter - 1) : -1])
        else:
            for j in range(i + 1, len(token_sequence)):
                next_entropy = -np.log2(token_frequency[int(token_sequence[j])])
                next_entropy = max(window_size / 3.0, next_entropy)
                if next_entropy < 1.0e-2:
                    next_entropy = 0.0
                current_entropy += next_entropy
                counter += 1
                if current_entropy >= window_size:
                    break

            result.append(token_sequence[i + 1 : i + 1 + counter])

    return result


@numba.njit(nogil=True)
def fixed_window(token_sequence, window_size, token_frequency, reverse=False):
    result = []
    for i in range(len(token_sequence)):
        if reverse:
            result.append(token_sequence[i - 1 : max(0, i - window_size - 1) : -1])
        else:
            result.append(token_sequence[i + 1 : i + window_size + 1])

    return result


@numba.njit(nogil=True)
def mass_conservation_window(
    token_sequence, window_size, token_frequency, reverse=False
):
    result = []

    for i in range(len(token_sequence)):
        float_window_size = window_size / token_frequency[token_sequence[i]]
        current_window_size = np.uint32(np.round(float_window_size))
        current_window_size = max(1, current_window_size)
        current_window_size = min(50, current_window_size)
        if reverse:
            result.append(
                token_sequence[i - 1 : max(0, i - current_window_size - 1) : -1]
            )
        else:
            result.append(
                token_sequence[
                    i + 1 : min(i + current_window_size + 1, len(token_sequence))
                ]
            )

    return result


@numba.njit(nogil=True)
def flat_kernel(window, window_size):
    return np.ones(len(window), dtype=np.float32)


@numba.njit(nogil=True)
def triangle_kernel(window, window_size):
    start = max(window_size, len(window))
    stop = window_size - len(window)
    return np.arange(start, stop, -1).astype(np.float32)


@numba.njit(nogil=True)
def harmonic_kernel(window, window_size):
    result = np.arange(1, len(window) + 1).astype(np.float32)
    return 1.0 / result


_WINDOW_FUNCTIONS = {
    "information": information_window,
    "fixed": fixed_window,
    "mass_conservation": mass_conservation_window,
}

_SYMMETRIC_WINDOWS = {
    "fixed",
    "information",
}

_KERNEL_FUNCTIONS = {
    "flat": flat_kernel,
    "harmonic": harmonic_kernel,
    "triangular": triangle_kernel,
}
