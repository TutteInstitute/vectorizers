import numpy as np
import numba
from collections import namedtuple

CooArray = namedtuple("CooArray", ["row", "col", "val", "key", "ind", "min", "depth"])

COO_QUICKSORT_LIMIT = 1 << 16
COO_MEM_MULTIPLIER = 1.5


@numba.njit(nogil=True)
def set_array_size(token_sequences, window_array):
    tot_len = np.zeros(window_array.shape[0]).astype(np.float64)
    window_array = window_array.astype(np.float64)
    for seq in token_sequences:
        counts = np.bincount(seq, minlength=window_array.shape[1]).astype(np.float64)
        tot_len += np.dot(
            window_array, counts
        ).T  # NOTE: numba only does dot products with floats
    return tot_len.astype(np.int64)


@numba.njit(nogil=True)
def merge_sum_duplicates(coo):
    new_depth = True
    for i in range(coo.depth[0]):
        if coo.min[i] <= 0:
            coo.min[:i] = -coo.ind[0]
            coo.min[i] = coo.ind[0]
            new_depth = False
            break
        else:
            array_len = coo.ind[0] - np.abs(coo.min[i + 1]) + 1
            result_row = np.zeros(array_len)
            result_col = np.zeros(array_len)
            result_val = np.zeros(array_len)
            result_key = np.zeros(array_len)
            ptr1 = np.abs(coo.min[i + 1])
            ptr2 = coo.min[i]
            result_ptr = 0
            result_key[0] = -1

            while ptr1 < coo.min[i] and ptr2 < coo.ind[0]:
                if coo.key[ptr1] <= coo.key[ptr2]:
                    this_ptr = ptr1
                    ptr1 += 1
                else:
                    this_ptr = ptr2
                    ptr2 += 1

                if coo.key[this_ptr] == result_key[result_ptr]:
                    result_val[result_ptr] += coo.val[this_ptr]
                else:
                    result_ptr += 1
                    result_val[result_ptr] = coo.val[this_ptr]
                    result_row[result_ptr] = coo.row[this_ptr]
                    result_col[result_ptr] = coo.col[this_ptr]
                    result_key[result_ptr] = coo.key[this_ptr]

            if ptr1 >= coo.min[i]:
                while ptr2 < coo.ind[0]:
                    this_ptr = ptr2
                    ptr2 += 1

                    if coo.key[this_ptr] == result_key[result_ptr]:
                        result_val[result_ptr] += coo.val[this_ptr]
                    else:
                        result_ptr += 1
                        result_val[result_ptr] = coo.val[this_ptr]
                        result_row[result_ptr] = coo.row[this_ptr]
                        result_col[result_ptr] = coo.col[this_ptr]
                        result_key[result_ptr] = coo.key[this_ptr]
            else:
                while ptr1 < coo.min[i]:
                    this_ptr = ptr1
                    ptr1 += 1

                    if coo.key[this_ptr] == result_key[result_ptr]:
                        result_val[result_ptr] += coo.val[this_ptr]
                    else:
                        result_ptr += 1
                        result_val[result_ptr] = coo.val[this_ptr]
                        result_row[result_ptr] = coo.row[this_ptr]
                        result_col[result_ptr] = coo.col[this_ptr]
                        result_key[result_ptr] = coo.key[this_ptr]

            coo.row[np.abs(coo.min[i + 1]) : coo.ind[0]] = result_row[1:]
            coo.col[np.abs(coo.min[i + 1]) : coo.ind[0]] = result_col[1:]
            coo.val[np.abs(coo.min[i + 1]) : coo.ind[0]] = result_val[1:]
            coo.key[np.abs(coo.min[i + 1]) : coo.ind[0]] = result_key[1:]
            coo.ind[0] = np.abs(coo.min[i + 1]) + result_ptr

    if new_depth:
        coo.min[: coo.depth[0]] = -coo.ind[0]
        coo.min[coo.depth[0]] = coo.ind[0]
        coo.depth[0] += 1


@numba.njit(nogil=True)
def merge_all_sum_duplicates(coo):
    new_min = np.zeros(coo.depth[0])
    ptr = 0
    for i in range(coo.depth[0]):
        if coo.min[i] > 0:
            new_min[ptr] = coo.min[i]
            ptr += 1
    coo.min[: coo.depth[0]] = new_min
    merge_sum_duplicates(coo)


@numba.njit(nogil=True)
def coo_sum_duplicates(coo):
    upper_lim = coo.ind[0]
    lower_lim = np.abs(coo.min[0])

    perm = np.argsort(coo.key[lower_lim:upper_lim])

    coo.row[lower_lim:upper_lim] = coo.row[lower_lim:upper_lim][perm]
    coo.col[lower_lim:upper_lim] = coo.col[lower_lim:upper_lim][perm]
    coo.val[lower_lim:upper_lim] = coo.val[lower_lim:upper_lim][perm]
    coo.key[lower_lim:upper_lim] = coo.key[lower_lim:upper_lim][perm]

    sum_ind = lower_lim
    this_row = coo.row[lower_lim]
    this_col = coo.col[lower_lim]
    this_val = np.float32(0)
    this_key = coo.key[lower_lim]

    for i in range(lower_lim, upper_lim):
        if coo.key[i] == this_key:
            this_val += coo.val[i]
        else:
            coo.row[sum_ind] = this_row
            coo.col[sum_ind] = this_col
            coo.val[sum_ind] = this_val
            coo.key[sum_ind] = this_key
            this_row = coo.row[i]
            this_col = coo.col[i]
            this_val = coo.val[i]
            this_key = coo.key[i]
            sum_ind += 1

    if this_key != coo.key[upper_lim]:
        coo.row[sum_ind] = this_row
        coo.col[sum_ind] = this_col
        coo.val[sum_ind] = this_val
        coo.key[sum_ind] = this_key
        sum_ind += 1

    coo.ind[0] = sum_ind
    merge_sum_duplicates(coo)


@numba.njit(nogil=True)
def coo_increase_mem(coo):

    temp = coo.row
    new_size = np.int32(max(np.round(COO_MEM_MULTIPLIER * temp.shape[0]), COO_QUICKSORT_LIMIT+1))
    new_row = np.zeros(new_size, dtype=np.int32)
    new_row[: temp.shape[0]] = temp

    temp = coo.col
    new_size = np.int32(max(np.round(COO_MEM_MULTIPLIER * temp.shape[0]), COO_QUICKSORT_LIMIT+1))
    new_col = np.zeros(new_size, dtype=np.int32)
    new_col[: temp.shape[0]] = temp

    temp = coo.val
    new_size = np.int32(max(np.round(COO_MEM_MULTIPLIER * temp.shape[0]), COO_QUICKSORT_LIMIT+1))
    new_val = np.zeros(new_size, dtype=np.float32)
    new_val[: temp.shape[0]] = temp

    temp = coo.key
    new_size = np.int32(max(np.round(COO_MEM_MULTIPLIER * temp.shape[0]), COO_QUICKSORT_LIMIT+1))
    new_key = np.zeros(new_size, dtype=np.int64)
    new_key[: temp.shape[0]] = temp

    temp = coo.min
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * (temp.shape[0]+2)))
    new_min = np.zeros(new_size, dtype=np.int64)
    new_min[: temp.shape[0]] = temp

    coo = CooArray(
        new_row,
        new_col,
        new_val,
        new_key,
        coo.ind,
        new_min,
        coo.depth,
    )

    return coo


@numba.njit(nogil=True)
def coo_append(coo, tup):
    coo.row[coo.ind[0]] = tup[0]
    coo.col[coo.ind[0]] = tup[1]
    coo.val[coo.ind[0]] = tup[2]
    coo.key[coo.ind[0]] = tup[3]
    coo.ind[0] += 1

    if (coo.ind[0] - np.abs(coo.min[0])) >= COO_QUICKSORT_LIMIT:
        coo_sum_duplicates(coo)
        if (coo.key.shape[0] - np.abs(coo.min[0])) <= COO_QUICKSORT_LIMIT:
            merge_all_sum_duplicates(coo)
            if coo.ind[0] >= 0.95 * coo.key.shape[0]:
                coo = coo_increase_mem(coo)

    if coo.ind[0] == coo.key.shape[0] - 1:
        coo_sum_duplicates(coo)
        if (coo.key.shape[0] - np.abs(coo.min[0])) <= COO_QUICKSORT_LIMIT:
            merge_all_sum_duplicates(coo)
            if coo.ind[0] >= 0.95 * coo.key.shape[0]:
                coo = coo_increase_mem(coo)

    return coo


@numba.njit(nogil=True)
def sum_coo_entries(seq):
    seq.sort()
    this_coord = (seq[0][0], seq[0][1])
    this_sum = 0
    reduced_data = []
    for entry in seq:
        if (entry[0], entry[1]) == this_coord:
            this_sum += entry[2]
        else:
            reduced_data.append((this_coord[0], this_coord[1], this_sum))
            this_sum = entry[2]
            this_coord = (entry[0], entry[1])

    reduced_data.append((this_coord[0], this_coord[1], this_sum))

    return reduced_data

@numba.njit(nogil=True)
def em_update_matrix(
    posterior_data,
    prior_indices,
    prior_indptr,
    prior_data,
    n_unique_tokens,
    target_gram_ind,
    windows,
    kernels,
):
    """
    Updated the csr matrix from one round of EM on the given (hstack of) n
    cooccurrence matrices provided in csr format.

    Parameters
    ----------
    posterior_data: numpy.array
        The csr data of the hstacked cooccurrence matrix to be updated

    prior_indices:  numpy.array
        The csr indices of the hstacked cooccurrence matrix

    prior_indptr: numpy.array
        The csr indptr of the hstacked cooccurrence matrix

    prior_data: numpy.array
        The csr data of the hstacked cooccurrence matrix

    n_unique_tokens: int
        The number of unique tokens

    target_gram_ind: int
        The index of the target ngram to update

    windows: List of List of int
        The indices of the tokens in the windows

    kernels: List of List of floats
        The kernel values of the entries in the windows.

    Returns
    -------
    posterior_data: numpy.array
        The data of the updated csr matrix after an update of EM.
    """
    total_win_length = np.sum(np.array([len(w) for w in windows]))
    window_posterior = np.zeros(total_win_length)
    context_ind = np.zeros(total_win_length, dtype=np.int64)
    win_offset = np.append(
        np.zeros(1, dtype=np.int64),
        np.cumsum(np.array([len(w) for w in windows])),
    )[:-1]

    col_ind = prior_indices[
        prior_indptr[target_gram_ind] : prior_indptr[target_gram_ind + 1]
    ]

    for w, window in enumerate(windows):
        for i, context in enumerate(window):
            if kernels[w][i] > 0:
                context_ind[i + win_offset[w]] = np.searchsorted(
                    col_ind, context + w * n_unique_tokens
                )
                # assert(col_ind[context_ind[i + win_offset[w]]] == context+w * n_unique_tokens)
                if (
                    col_ind[context_ind[i + win_offset[w]]]
                    == context + w * n_unique_tokens
                ):
                    window_posterior[i + win_offset[w]] = (
                        kernels[w][i]
                        * prior_data[
                            prior_indptr[target_gram_ind]
                            + context_ind[i + win_offset[w]]
                        ]
                    )
                else:
                    window_posterior[i + win_offset[w]] = 0

    temp = window_posterior.sum()
    if temp > 0:
        window_posterior /= temp

    # Partial M_step - Update the posteriors
    for w, window in enumerate(windows):
        for i, context in enumerate(window):
            val = window_posterior[i + win_offset[w]]
            if val > 0:
                posterior_data[
                    prior_indptr[target_gram_ind] + context_ind[i + win_offset[w]]
                ] += val

    return posterior_data
