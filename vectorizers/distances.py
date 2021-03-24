import numba
import numpy as np

EPS = 1e-11


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
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i]) ** p

        return result ** (1.0 / p)

    elif p == 2:
        for i in range(x_cdf.shape[0]):
            val = x_cdf[i] - y_cdf[i]
            result += val * val

        return np.sqrt(result)

    elif p == 1:
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i])

        return result

    else:
        raise ValueError("Invalid p supplied to Kantorvich distance")


@numba.njit()
def circular_kantorovich(x, y, p=1):

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

    mu = np.median((x_cdf - y_cdf) ** p)

    # Now we just want minkowski distance on the CDFs shifted by mu
    result = 0.0
    if p > 2:
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i] - mu) ** p

        return result ** (1.0 / p)

    elif p == 2:
        for i in range(x_cdf.shape[0]):
            val = x_cdf[i] - y_cdf[i] - mu
            result += val * val

        return np.sqrt(result)

    elif p == 1:
        for i in range(x_cdf.shape[0]):
            result += np.abs(x_cdf[i] - y_cdf[i] - mu)

        return result

    else:
        raise ValueError("Invalid p supplied to Kantorvich distance")


@numba.njit()
def total_variation(x, y):
    x_sum = 0.0
    y_sum = 0.0
    result = 0.0

    for i in range(x.shape[0]):
        x_sum += x[i]
        y_sum += y[i]

    x_pdf = x / x_sum
    y_pdf = y / y_sum

    for i in range(x.shape[0]):
        result += 0.5 * np.abs(x_pdf[i] - y_pdf[i])

    return result


@numba.njit()
def jensen_shannon_divergence(x, y):
    result = 0.0
    l1_norm_x = 0.0
    l1_norm_y = 0.0
    dim = x.shape[0]

    for i in range(dim):
        l1_norm_x += x[i]
        l1_norm_y += y[i]

    l1_norm_x += EPS * dim
    l1_norm_y += EPS * dim

    pdf_x = (x + EPS) / l1_norm_x
    pdf_y = (y + EPS) / l1_norm_y
    m = 0.5 * (pdf_x + pdf_y)

    for i in range(dim):
        result += 0.5 * (
            pdf_x[i] * np.log(pdf_x[i] / m[i]) + pdf_y[i] * np.log(pdf_y[i] / m[i])
        )
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

    pdf_x = (x + EPS) / l1_norm_x
    pdf_y = (y + EPS) / l1_norm_y

    for i in range(dim):
        result += pdf_x[i] * np.log(pdf_x[i] / pdf_y[i]) + pdf_y[i] * np.log(
            pdf_y[i] / pdf_x[i]
        )

    return result


#
# --- Sparse support functions
#


# Just reproduce a simpler version of numpy unique (not numba supported yet)
@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
    return aux[flag]


# Just reproduce a simpler version of numpy union1d (not numba supported yet)
@numba.njit()
def arr_union(ar1, ar2):
    if ar1.shape[0] == 0:
        return ar2
    elif ar2.shape[0] == 0:
        return ar1
    else:
        return arr_unique(np.concatenate((ar1, ar2)))


@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def sparse_sum(ind1, data1, ind2, data2):
    result_ind = arr_union(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = i1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = i2
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit()
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = arr_intersect(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


# Return dense vectors supported on the union of the non-zero valued indices
@numba.njit()
def dense_union(ind1, data1, ind2, data2):
    result_ind = arr_union(ind1, ind2)
    result_data1 = np.zeros(result_ind.shape[0], dtype=np.float32)
    result_data2 = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_data1[nnz] = data1[i1]
                result_data2[nnz] = data2[i2]
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_data1[nnz] = data1[i1]
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_data2[nnz] = data2[i2]
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        val = data1[i1]
        if val != 0:
            result_data1[nnz] = data1[i1]
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_data2[nnz] = data2[i2]
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros
    result_data1 = result_data1[:nnz]
    result_data2 = result_data2[:nnz]

    return result_data1, result_data2


#
# --- Sparse distance functions
#


@numba.njit()
def sparse_hellinger(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = np.sum(data1)
    norm2 = np.sum(data2)
    sqrt_norm_prod = np.sqrt(norm1 * norm2)

    for i in range(aux_data.shape[0]):
        result += np.sqrt(aux_data[i])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    elif result > sqrt_norm_prod:
        return 0.0
    else:
        return np.sqrt(1.0 - (result / sqrt_norm_prod))


@numba.njit()
def sparse_total_variation(ind1, data1, ind2, data2):
    norm1 = np.sum(data1)
    norm2 = np.sum(data2)
    aux_inds, aux_data = sparse_diff(ind1, data1 / norm1, ind2, data2 / norm2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += 0.5 * np.abs(aux_data[i])
    return result


# Because of the EPS values and the need to normalize after adding them (and then average those for jensen_shannon)
# it seems like we might as well just take the dense union (dense vectors supported on the union of indices)
# and call the dense distance functions


@numba.njit()
def sparse_jensen_shannon_divergence(ind1, data1, ind2, data2):
    dense_data1, dense_data2 = dense_union(ind1, data1, ind2, data2)
    return jensen_shannon_divergence(dense_data1, dense_data2)


@numba.njit()
def sparse_symmetric_kl_divergence(ind1, data1, ind2, data2):
    dense_data1, dense_data2 = dense_union(ind1, data1, ind2, data2)
    return symmetric_kl_divergence(dense_data1, dense_data2)
