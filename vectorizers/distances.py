import numba
import numpy as np

EPS = 1e11

@numba.njit()
def hellinger(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        result += np.sqrt(x[i] * y[i])
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    if l1_norm_x == 0 and l1_norm_y == 0:
        return 0.0
    elif l1_norm_x == 0 or l1_norm_y == 0:
        return 1.0
    else:
        return np.sqrt(1 - result / np.sqrt(l1_norm_x * l1_norm_y))


@numba.njit()
def kantorovich1d(x, y):
    pass


@numba.njit()
def circular_kantorovich(x, y):
    pass


@numba.njit()
def total_variation(x, y):
    pass


@numba.njit()
def jensen_shannon_divergence(x, y):
    pass


@numba.njit()
def symmetric_kl_divergence(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    l1_norm_x += EPS * dim
    l1_norm_y += EPS * dim

    for i in range(dim):
        result += ((x[i]+EPS) / l1_norm_x * np.log( ((x[i]+EPS) / l1_norm_x) / ((y[i]+EPS) / l1_norm_y) )
                    + (y[i]+EPS) / l1_norm_y * np.log( ((y[i]+EPS) / l1_norm_y) / ((x[i]+EPS) / l1_norm_x)))
    return result


@numba.njit()
def sparse_hellinger(ind1, data1, ind2, data2):
    pass


@numba.njit()
def sparse_total_variation(ind1, data1, ind2, data2):
    pass


@numba.njit()
def sparse_jensen_shannon_divergence(ind1, data1, ind2, data2):
    pass


@numba.njit()
def sparse_symmetric_kl_divergence(ind1, data1, ind2, data2):
    pass

