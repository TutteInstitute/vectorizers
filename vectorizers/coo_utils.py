import numpy as np
import numba
from collections import namedtuple

CooArray = namedtuple("CooArray", ["row", "col", "val", "key", "ind", "min", "depth"])

COO_QUICKSORT_LIMIT = 1 << 16
COO_MEM_MULTIPLIER = 2.0


@numba.njit(nogil=True)
def set_array_size(token_sequences, window_array):
    tot_len = np.zeros(window_array.shape[0]).astype(np.float64)
    window_array = window_array.astype(np.float64)
    for seq in token_sequences:
        counts = np.bincount(seq, minlength=window_array.shape[1]).astype(np.float64)
        tot_len += np.dot(
            window_array, counts
        ).T  # NOTE: numba only does dot products with floats
    tot_len[tot_len <= COO_QUICKSORT_LIMIT] = COO_QUICKSORT_LIMIT + 1
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
def coo_sum_duplicates(coo, kind):
    upper_lim = coo.ind[0]
    lower_lim = np.abs(coo.min[0])

    perm = np.argsort(coo.key[lower_lim:upper_lim], kind=kind)

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
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * temp.shape[0]))
    new_row = np.zeros(new_size, dtype=np.int32)
    new_row[: temp.shape[0]] = temp

    temp = coo.col
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * temp.shape[0]))
    new_col = np.zeros(new_size, dtype=np.int32)
    new_col[: temp.shape[0]] = temp

    temp = coo.val
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * temp.shape[0]))
    new_val = np.zeros(new_size, dtype=np.float32)
    new_val[: temp.shape[0]] = temp

    temp = coo.key
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * temp.shape[0]))
    new_key = np.zeros(new_size, dtype=np.int64)
    new_key[: temp.shape[0]] = temp

    temp = coo.min
    new_size = np.int32(np.round(COO_MEM_MULTIPLIER * temp.shape[0]))
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
        coo_sum_duplicates(coo, kind="quicksort")
        if (coo.key.shape[0] - np.abs(coo.min[0])) <= COO_QUICKSORT_LIMIT:
            merge_all_sum_duplicates(coo)
            if coo.ind[0] >= 0.95 * coo.key.shape[0]:
                coo = coo_increase_mem(coo)

    if coo.ind[0] == coo.key.shape[0] - 1:
        coo_sum_duplicates(coo, kind="quicksort")
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
def update_coo_entries(seq, tup):
    place = np.searchsorted(seq, tup)
    if seq[place][1:2] == tup[1:2]:
        seq[place][3] += tup[3]
        return seq
    elif seq[place - 1][1:2] == tup[1:2]:
        seq[place - 1][3] += tup[3]
        return seq
    return seq.insert(place, tup)
