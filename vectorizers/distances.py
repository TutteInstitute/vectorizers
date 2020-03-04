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
def kantorovich1d(x, y, p=1):

    # Normalize and do a cumulative sum trick

    x_sum = 0.0
    y_sum = 0.0
    for i in range(x.shape[0]):
        x_sum += x[i]
        y_sum += y[i]

    x_cdf = x / x_sum
    y_cdf = y / y_sum

    for i in range(1, x_cdf.shape[0]):
        x_cdf[i] += x_cdf[i - 1]
        y_cdf[i] += y_cdf[i - 1]

    # Now we just want minkowski distance on the CDFs
    result = 0.0
    if p > 2:
        for i in range(x.shape[0]):
            result += np.abs(x - y) ** p

        return result ** (1.0 /p)

    elif p == 2:
        for i in range(x.shape[0]):
            val = x - y
            result += val * val

        return np.sqrt(result)

    elif p == 1:
        for i in range(x.shape[0]):
            result += np.abs(x - y)

        return result

    else:
        raise ValueError("Invalid p supplied to Kantorvich distance")

@numba.njit()
def circular_kantorovich(x, y):
    pass


@numba.njit()
def total_variation(x, y):
    pass


@numba.njit()
def jensen_shannon_divergence(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]
    m = np.zeros(dim)
    l1_norm_m = 0.0

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]
        m[i] = 0.5*(x[i]+y[i]) + EPS

    l1_norm_x += EPS * dim
    l1_norm_y += EPS * dim
    l1_norm_m = 0.5*(l1_norm_x + l1_norm_y)

    for i in range(dim):
        result += 0.5*((x[i] + EPS) / l1_norm_x * np.log(((x[i] + EPS) / l1_norm_x) / (m[i] / l1_norm_m))
                   + (y[i] + EPS) / l1_norm_y * np.log(((y[i] + EPS) / l1_norm_y) / (m[i] / l1_norm_m)))
    return result


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

