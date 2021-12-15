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
def count_pairs(char_list):
    result = {}
    for array in char_list:
        for i in range(array.shape[0] - 1):
            pair = (array[i], array[i + 1])
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result


@numba.njit(inline="always", nogil=True)
def pair_length(pair, pair_lengths, max_char_code):
    left_length = 1 if pair[0] <= max_char_code else pair_lengths[pair[0]]
    right_length = 1 if pair[1] <= max_char_code else pair_lengths[pair[1]]
    return left_length + right_length


@numba.njit(
    nogil=True,
    locals=dict(
        i=numba.uint32,
        skip_char=numba.boolean,
        len_char_list=numba.uint32,
        last_char_added=numba.int64,
    )
)
def contract_and_count_pairs(char_list, pair_to_contract, pair_counts, new_code=-1):
    skip_char = False
    len_char_list = len(char_list)
    last_char_added = -1
    new_char_list = np.zeros(len_char_list, dtype=np.int64)
    new_char_index = 0

    for i in range(len_char_list - 1):
        if skip_char:
            skip_char = False
            continue

        if char_list[i] == pair_to_contract[0] and char_list[i + 1] == pair_to_contract[1]:
            if i > 0:
                prior_pair = (last_char_added, char_list[i])
                if prior_pair in pair_counts:
                    pair_counts[prior_pair] -= 1
                new_prior_pair = (last_char_added, new_code)
                if new_prior_pair in pair_counts:
                    pair_counts[new_prior_pair] += 1
                else:
                    pair_counts[new_prior_pair] = 1

            if i < len_char_list - 2:
                next_pair = (char_list[i + 1], char_list[i + 2])
                if next_pair in pair_counts:
                    pair_counts[next_pair] -= 1
                new_next_pair = (new_code, char_list[i + 2])
                if new_next_pair in pair_counts:
                    pair_counts[new_next_pair] += 1
                else:
                    pair_counts[new_next_pair] = 1

            new_char_list[new_char_index] = new_code
            last_char_added = new_code
            skip_char = True
        else:
            new_char_list[new_char_index] = char_list[i]
            last_char_added = char_list[i]

        new_char_index += 1

    if not skip_char:
        new_char_list[new_char_index] = char_list[i + 1]
        new_char_index += 1

    return new_char_list[:new_char_index], pair_counts


@numba.njit(nogil=True)
def pruning_max_freq_pair(count_dict, code_lengths, max_char_code, min_count=1):
    result = (-1, -1)
    max_count = 0
    best_length = 0
    keys_to_kill = set([(-1, -1)])
    for pair, count in count_dict.items():
        if count > max_count:
            result = pair
            max_count = count
            best_length = pair_length(pair, code_lengths, max_char_code)
        elif count == max_count:
            length = pair_length(pair, code_lengths, max_char_code)
            if length > best_length or (length == best_length and pair > result):
                result = pair
                max_count = count
                best_length = length
        elif count <= min_count:
            keys_to_kill.add(pair)

    if len(keys_to_kill) > 0:
        for key in keys_to_kill:
            if key[0] >= 0:
                count_dict.pop(key)

    if max_count == 1:
        return (-1, -1), 0

    return result, max_count


@numba.njit(inline="always", nogil=True)
def to_unicode(code, tokens, max_char_code):
    if code <= max_char_code:
        return chr(code)
    else:
        return tokens[code - max_char_code - 1]


@numba.njit(inline="always", nogil=True)
def to_code_num(code, code_list, max_char_code):
    if code <= max_char_code:
        return [code]
    else:
        return code_list[code - max_char_code - 1]


@numba.njit(inline="always", nogil=True)
def pair_to_string(pair, tokens, max_char_code):
    return to_unicode(pair[0], tokens, max_char_code) + to_unicode(pair[1], tokens, max_char_code)


@numba.njit(inline="always", nogil=True)
def pair_to_list(pair, code_list, max_char_code):
    return to_code_num(pair[0], code_list, max_char_code) + to_code_num(pair[1], code_list, max_char_code)


@numba.njit()
def bpe_train(char_list, vocab_size=10000, min_count=1):
    # Initialize compressed chars
    compressed_chars = [np.empty(len(chars), dtype=np.int64) for chars in char_list]
    max_char_code = 0
    for i, chars in enumerate(char_list):
        for j, c in enumerate(chars):
            c_val = ord(c)
            compressed_chars[i][j] = c_val
            if c_val > max_char_code:
                max_char_code = c_val

    # Initialize coding, counts, and lengths
    new_code = max_char_code + 1
    pair_counts = count_pairs(compressed_chars)
    current_min_count = np.max(np.array(list(pair_counts.values()))) // 2
    code_lengths = {-1: 1}

    # Initialize code words and lengths so numba gets the types right
    pair_to_replace, count = pruning_max_freq_pair(
        pair_counts, code_lengths, max_char_code, min_count=current_min_count
    )
    tokens = [chr(pair_to_replace[0]) + chr(pair_to_replace[1])]
    code_list = [pair_to_replace]
    code_lengths[new_code] = pair_length(pair_to_replace, code_lengths, max_char_code)

    while len(tokens) < vocab_size:
        for i, char_array in enumerate(compressed_chars):
            compressed_chars[i], pair_counts = contract_and_count_pairs(
                char_array, pair_to_replace, pair_counts, new_code
            )

        pair_counts.pop(pair_to_replace)
        new_code += 1
        pair_to_replace, count = pruning_max_freq_pair(
            pair_counts, code_lengths, max_char_code, min_count=current_min_count
        )

        if current_min_count > min_count and count <= current_min_count:
            current_min_count = max(current_min_count // 2, min_count)
            pair_counts = count_pairs(compressed_chars)
            pair_to_replace, count = pruning_max_freq_pair(
                pair_counts, code_lengths, max_char_code,
                min_count=current_min_count
            )

        if pair_to_replace[0] >= 0 and pair_counts[pair_to_replace] > 1:
            tokens.append(pair_to_string(pair_to_replace, tokens, max_char_code))
            code_list.append(pair_to_replace)
            code_lengths[new_code] = pair_length(pair_to_replace, code_lengths, max_char_code)
        else:
            break

    return tokens, code_list, compressed_chars, max_char_code


@numba.njit()
def contract_pair(char_list, pair_to_contract, new_code=-1):
    skip_char = False
    len_char_list = len(char_list)
    new_char_list = np.zeros(len_char_list, dtype=np.int64)
    new_char_index = 0

    for i in range(len_char_list - 1):
        if skip_char:
            skip_char = False
            continue

        if char_list[i] == pair_to_contract[0] and char_list[i + 1] == pair_to_contract[1]:
            new_char_list[new_char_index] = new_code
            skip_char = True
        else:
            new_char_list[new_char_index] = char_list[i]

        new_char_index += 1

    if not skip_char:
        new_char_list[new_char_index] = char_list[i + 1]
        new_char_index += 1

    return new_char_list[:new_char_index]


@numba.njit(nogil=True)
def bpe_encode(chars, code_list, max_char_code):
    compressed_chars = np.empty(len(chars), dtype=np.int64)
    for i, c in enumerate(chars):
        compressed_chars[i] = ord(c)

    new_code = max_char_code + 1
    for code_pair in code_list:
        compressed_chars = contract_pair(compressed_chars, code_pair, new_code=new_code)
        new_code += 1

    return compressed_chars

def bpe_decode(code_array, tokens, max_char_code):
    result = [
        chr(c) if c <= max_char_code else tokens[c - max_char_code - 1]
        for c in code_array
    ]
    return "".join(result)

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
                for key, val in self.base_dictionary.items():
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
                input_dict = self.base_dictionary.copy()

            encoding_dict = lempel_ziv_based_encode(string, input_dict, self.hash_function, self.max_size)

            for ngram in encoding_dict.keys():
                if ngram in self.column_label_dictionary_:
                    col = self.column_label_dictionary_[ngram]
                    val = encoding_dict[ngram]
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


class BytePairEncodingVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, max_vocab_size=10000, min_token_occurrence=1, return_type="matrix"):

        self.max_vocab_size = max_vocab_size
        self.min_token_occurence = min_token_occurrence
        self.return_type = return_type


    def fit_transform(self, X, y=None, **fit_params):
        if self.return_type not in ("matrix", "tokens", "sequences"):
            raise ValueError(f"return_type must be one of 'matrix' 'tokens', or 'sequences', not {self.return_type}")
        if not type(self.max_vocab_size) == int or self.max_vocab_size <= 0:
            raise ValueError(f"max_vocab_size should be a non-zero positive integer not {self.max_vocab_size}")
        if not type(self.min_token_occurence) == int or self.min_token_occurence <= 0:
            raise ValueError(f"min_token_occurence should be a non-zero positive integer not {self.min_token_occurence}")

        (
            self.tokens_,
            self.code_list_,
            encodings,
            self.max_char_code_,
        ) = bpe_train(X, vocab_size=self.max_vocab_size, min_count=self.min_token_occurence)

        if self.return_type == "sequences":
            return encodings
        elif self.return_type == "tokens":
            result = []
            for row in encodings:
                result.append(
                    [
                        chr(x) if x <= self.max_char_code_ else self.tokens_[x - self.max_char_code_ - 1]
                        for x in row
                    ]
                )
            return result
        elif self.return_type == "matrix":
            unique_codes = np.unique(np.hstack(encodings))
            self.column_label_dictionary_ = dict(zip(unique_codes, np.arange(unique_codes.shape[0])))
            indices = []
            data = []
            indptr = [0]

            for row in encodings:
                indices.extend([self.column_label_dictionary_[x] for x in row])
                data.extend([1 for i in range(row.shape[0])])
                indptr.append(len(indices))

            result = scipy.sparse.csr_matrix((data, indices, indptr), dtype=np.float32)
            result.sum_duplicates()

            return result
        else:
            raise ValueError(f"return_type must be one of 'matrix' 'tokens', or 'sequences', not {self.return_type}")

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["tokens_", "code_list_", "max_char_code_"])

        encodings = [bpe_encode(string) for string in X]

        if self.return_type == "sequences":
            return encodings
        elif self.return_type == "tokens":
            result = []
            for row in encodings:
                result.append(
                    [
                        chr(x) if x <= self.max_char_code_ else self.tokens_[x - self.max_char_code_ - 1]
                        for x in row
                    ]
                )
            return result
        elif self.return_type == "matrix":
            indices = []
            data = []
            indptr = [0]

            for row in encodings:
                indices.extend([self.column_label_dictionary_[x] for x in row])
                data.extend([1 for i in range(row.shape[0])])
                indptr.append(len(indices))

            result = scipy.sparse.csr_matrix((data, indices, indptr), dtype=np.float32)
            result.sum_duplicates()

            return result
        else:
            raise ValueError(f"return_type must be one of 'matrix' 'tokens', or 'sequences', not {self.return_type}")

