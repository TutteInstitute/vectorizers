import numpy as np
import numba
import scipy.sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT32 = np.iinfo(np.int32).max

@numba.njit()
def identity_hash(x):
    return x


@numba.njit
def unicode_to_uint8(string):
    result = []
    for char in string:
        ord_char = ord(char)
        if ord_char > 255:
            extended_char = True
            result.append(0)
        else:
            extended_char = False
        while ord_char > 255:
            result.append(ord_char & 0xff)
            ord_char = ord_char >> 8
        result.append(ord_char)
        if extended_char:
            result.append(0)

    return result

# Duplicated from https://gist.github.com/stevesimmons/58b5e113a41c5c23775d17cc83929d88
@numba.njit
def murmurhash(key, seed) -> int:
    length = len(key)
    n, t = divmod(length, 4)

    h = seed
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # Process whole blocks of 4 bytes
    for i in range(n):
        k1 = (key[4 * i] << 24) + (key[4 * i + 1] << 16) + (key[4 * i + 2] << 8) + key[4 * i + 3]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # ROTL32
        h ^= (k1 * c2) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF  # ROTL32
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Process tail of 1-3 bytes if present
    if t > 0:
        k1 = (key[4 * n] << 16)
        if t > 1:
            k1 += key[4 * n + 1] << 8
        if t > 2:
            k1 += key[4 * n + 2]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF  # ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h ^= k1

    h ^= length  # Include length to give different values for 1-3 tails of \0 bytes

    # Finalise by mixing the bits
    x = h
    x ^= (x >> 16)
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= (x >> 13)
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x


def make_hash(size, seed):
    @numba.njit
    def hash(string):
        byte_array = unicode_string_to_int_array(string)
        raw_hash = murmurhash(byte_array, seed)
        return raw_hash % size

    return hash


@numba.njit()
def unicode_string_to_int_array(string):
    result = np.empty(len(string), dtype=np.int64)
    for i in range(len(string)):
        result[i] = ord(string[i])
    return result


@numba.njit(nogil=True)
def lempel_ziv_based_encode(string, dictionary, hash_function=identity_hash, max_size=1<<20):
    current_size = len(dictionary)

    start = 0
    for end in range(len(string)):
        ngram = hash_function(string[start:end])
        if ngram in dictionary:
            dictionary[ngram] += 1
        elif current_size >= max_size:
            start = end
        else:
            dictionary[ngram] = 1
            current_size += 1
            start = end

    return dictionary


@numba.njit(nogil=True)
def counts_to_csr_data(count_dict, column_dict):
    indices = []
    data = []
    col_dict_size = len(column_dict)

    for ngram, count in count_dict.items():
        if ngram in column_dict:
            col = column_dict[ngram]
        else:
            column_dict[ngram] = col_dict_size
            col = col_dict_size
            col_dict_size += 1

        indices.append(col)
        data.append(count)

    return indices, data


@numba.njit(nogil=True)
def byte_pair_based_encode(string, dictionary, hash_function=identity_hash, max_size=1<<20):
    current_size = len(dictionary)
    string_array = unicode_string_to_int_array(string)

    pass

class LZCompressionVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, max_dict_size=1<<16, max_columns=1<<16, hash_function=identity_hash, base_dictionary=None, random_state=None):

        self.max_dict_size = max_dict_size
        self.max_columns = max_columns
        self.hash_function = hash_function
        self.base_dictionary = base_dictionary
        self.random_state = random_state

    def fit_transform(self, X, y=None, **fit_params):

        if self.max_columns is not None:
            random_state = check_random_state(self.random_state)
            hash_function = make_hash(self.max_columns, np.int32(random_state.randint(MAX_INT32)))
        else:
            hash_function = self.hash_function

        if self.max_columns is not None:
            self.column_label_dictionary_ = numba.typed.Dict.empty(numba.types.int32, numba.types.int64)
        else:
            self.column_label_dictionary_ = numba.typed.Dict.empty(numba.types.unicode_type, numba.types.int64)

        dict_size = 0
        indptr = [0]
        indices = []
        data = []

        for string in X:

            # Reset the input dict
            if self.max_columns is not None:
                input_dict = numba.typed.Dict.empty(numba.types.int32, numba.types.int64)
            else:
                input_dict = numba.typed.Dict.empty(numba.types.unicode_type, numba.types.int64)

            if self.base_dictionary is not None:
                for key, val in base_dictionary.items():
                    input_dict[key] = val

            encoding_dict = lempel_ziv_based_encode(string, input_dict, hash_function, self.max_dict_size)

            new_indices, new_data = counts_to_csr_data(encoding_dict, self.column_label_dictionary_)
            indices.extend(new_indices)
            data.extend(new_data)

            indptr.append(indptr[-1] + len(encoding_dict))

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32

        indices = np.asarray(indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        data = np.asarray(data, dtype=np.intc)

        self._train_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.column_label_dictionary_)),
            dtype=np.float32,
        )
        self._train_matrix.sort_indices()

        return self._train_matrix

    def transform(self, X, y=None):
        indptr = []
        indices = []
        data = []

        for string in X:

            if self.base_dictionary is None:
                input_dict = {}
            else:
                input_dict = base_dictionary.copy()

            encoding_dict = lempel_ziv_based_encode(string, input_dict, self.hash_function, self.max_size)

            for ngram in encoding_dict.keys():
                if ngram in self.column_label_dictionary_:
                    col = self.column_label_dictionary_[ngram]

                indices.append(col)
                data.append(val)

            indptr.append(indptr[-1] + len(encoding_dict))

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32

        indices = np.asarray(indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        data = np.asarray(data, dtype=np.intc)

        result = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(self.column_label_dictionary_)),
            dtype=np.float32,
        )
        result.sort_indices()

        return result


