import numpy as np
import numba
from collections import namedtuple

CooArray = namedtuple("CooArray", ["row", "col", "val", "key", "ind", "min"])


@numba.njit(nogil=True, inline="always")
def coo_sum_duplicates(coo, kind):
    upper_lim = coo.ind[0]
    lower_lim = coo.min[0]

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
    coo.min[0] = coo.ind[0]


@numba.njit(nogil=True, inline="always")
def coo_append(coo, tup):
    coo.row[coo.ind[0]] = tup[0]
    coo.col[coo.ind[0]] = tup[1]
    coo.val[coo.ind[0]] = tup[2]
    coo.key[coo.ind[0]] = tup[3]
    coo.ind[0] += 1

    if (coo.ind[0] - coo.min[0]) >= 1 << 18:
        coo_sum_duplicates(coo, kind="quicksort")
        if coo.key.shape[0] - coo.min[0] <= 1 << 18:
            coo.min[0] = 0.0
            coo_sum_duplicates(coo, kind="mergesort")
            if coo.ind[0] >= 0.95 * coo.key.shape[0]:
                raise ValueError(
                    f"The coo matrix array is over memory limit.  Increase coo_max_bytes to process data."
                )

    if coo.ind[0] == coo.key.shape[0]:
        coo.min[0] = 0.0
        coo_sum_duplicates(coo, kind="mergesort")
        if coo.ind[0] >= 0.95 * coo.key.shape[0]:
            raise ValueError(
                f"The coo matrix array is over memory limit.  Increase coo_max_bytes to process data."
            )


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
