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
@numba.njit()
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
    """Convert a dictionary of count data to necessary inputs for a csr matrix.

    Parameters
    ----------
    count_dict: dict of object to int
        The count of (hashed) objects; often ngrams.

    column_dict: dict of object to int
        The indices of the columns of the (hashed) objects often ngrams
        New entries will be added as necessary.

    Returns
    -------
    indices:  list of int
        The indices for the row

    data: list of int
        The data for the row
    """
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
    """Generate pair counts for all pairs of codes in char_list.

    Parameters
    ----------
    char_list: list of arrays of int64
        A list of encoded arrays for which pairs will be counted

    Returns
    -------
    pair_counts: dict of pairs to int
        A dictionary mapping pairs of codes to the count of the total
        number of occurrences of the pair in encoded arrays.
    """
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
    """Generate a new encoding by replacing ``pair_to_contract`` with ``new_code``
    and simultaneously updating the ``pair_counts`` dictionary to reflect the new
    pair counts resulting from the merges. This allows counting and merging to be done
    in a single pass.

    Parameters
    ----------
    char_list: array of int64
        The current encoding to be improved by contracting ``pair_to_contract`` to ``new_code``

    pair_to_contract: pair of int64
        The pair of codes to be contracted to a single new code

    pair_counts: dict of pairs to int
        The current counts of pairs that are being kept track of. This dict will
        be updated to reflect the new counts resulting from the contractions.

    new_code: int
        The new code value to replace ``pair_to_contract`` with.

    Returns
    -------
    new_char_list: array of int64
        The new array of codes with ``pair_to_contract`` merged to ``new_code``
        wherever it occurs.

    pair_counts: dict of pairs to int
        Updated counts for all the pairs in the original pair_counts dict, plus
        any newly created pairs involving the new code. Note that pairs that are
        not in the passed in pair_counts will not be decremented as required.
    """
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
    """Find the maximum frequency pair given a dictionary of counts of pair occurrences.
    Ties are broken first on the lengths (as string token representation) of the pairs,
    and then lexicographically on the pair's code values.

    During the search for the max, we will also find pairs to be removed from the
    dictionary based on the ``min_count`` value. This allows the dictionary to remain
    smaller than it otherwise would.

    Parameters
    ----------
    count_dict: dict of pairs to ints
        The counts of the number of occurrences of pairs

    code_lengths: dict of codes to ints
        The lengths of different codes as string token representations

    max_char_code: int
        The maximum code value of any single character in the learned code

    min_count: int
        The minimum number of occurrences of a pair for it to remain
        in the dictionary. Pairs with fewer than this number of occurences
        will be pruned out.

    Returns
    -------
    best_pair: pair of int64s
        The pair of codes that are the most frequent

    max_count: int
        the number of occurrences of the best pair.
    """
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
    """Train a byte pair encoding on a given list of strings.

    Parameters
    ----------
    char_list: list of strings
        The strings to learn a byte pair encoding scheme from

    vocab_size: int
        The maximum number of new codes representing byte sequences to learn.

    min_count: int
        The minimum number of occurrences a pair must have to be considered for merging.

    Returns
    -------
    tokens: list of strings
        The string representations of the new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code_ + 1``.

    code_list: list of pairs of int64s
        The pairs merged to create new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code_ + 1``.

    compressed_chars: list of arrays of int64s
        The encoded versions of the input strings

    max_char_code: int64
        The maximum value of character codes in the training set. For ascii strings
        this is simply 255, but for unicode strings it may be significantly larger. Code
        values associated with new learned tokens begin at ``max_char_code_ + 1``.
    """
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
    """Generate a new array on codes by contracting ``pair_to_contract`` to
    the code ``new_code``.

    Parameters
    ----------
    char_list: array of int64
        The array to apply pair contraction to

    pair_to_contract: pair of int64
        The code pair to be contracted to a new code value

    new_code: int64
        The new code value to use in place of ``pair_to_contract``

    Returns
    -------
    new_char_list: array of int64
        The new array of codes with ``pair_to_contract`` merged to ``new_code``
        wherever it occurs.
    """
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
    """Encode a string given a BPE code_list

    Parameters
    ----------
    chars: unicode_type
        THe string to be encoded

    code_list: list of code pairs (int64, int64)
        The learned encoding dictionary

    max_char_code: int64
        The maximum allowed code char for the given learned encoding

    Returns:
    --------
    compressed_array: ndarray of int64
        The array of the encoded representation
    """
    compressed_chars = np.empty(len(chars), dtype=np.int64)
    for i, c in enumerate(chars):
        code = ord(c)
        compressed_chars[i] = code if code <= max_char_code else 0

    new_code = max_char_code + 1
    for code_pair in code_list:
        compressed_chars = contract_pair(compressed_chars, code_pair, new_code=new_code)
        new_code += 1

    return compressed_chars

def bpe_decode(code_array, tokens, max_char_code):
    """Decode a BPE code array into a string

    Parameters
    ----------
    code_array: array of int64
        The code array to decode

    tokens: list of unicode_type
        The string representations of learned codes

    max_char_code: int64
        The maximum allowed code char for the given learned encoding

    Returns
    -------

    """
    result = [
        chr(c) if c <= max_char_code else tokens[c - max_char_code - 1]
        for c in code_array
    ]
    return "".join(result)

class LZCompressionVectorizer(BaseEstimator, TransformerMixin):
    """Create a vector representations of arbitrary string or byte sequences
    by using Lempel-Ziv compression. Each string is converted into the
    vector of how many times each LZ encoding string is used. This means
    that, for example, the Jaccard distance between such vectors is a
    close approximation of the normalized compression distance between the
    two strings under an LZ compression scheme.

    Parameters
    ----------
    max_dict_size: int (optional, default=65536)
        The maximum number of entries to allow in any LZ encoding dictionary

    max_columns: int (optional, default=65536)
        The maximum total number of columns to allow in the resulting
        vectorization. This is the total number of unique encoding
        strings over all the LZ encoding dictionaries for the entire
        training set. If specified a hashing trick will be used
        to constrain the columns.

    hash_function: function (optional, default=indentity_hash)
        A hash function taking encoding strings to unique identifiers.
        This can be  used to constrain the total number of columns produced
        via a hashing trick similar to a hashing vectorizer.

    base_dictionary: dict or None (optional, default=None)
        An initial dictionary to use for LZ compression. For example if the
        initial dictionary is the set of ascii characters this will give
        LZW compression.

    random_state: int, numpy.random.random_state or None (optional, default=None)
        The random state used in hash construction if using a fixed
        maximum number of columns.

    Attributes
    ----------
    column_label_dictionary_: dict
        A mapping from encoding dictionary entries (or hashes thereof)
        to the index of the column associated to that feature

    hash_function_: function
        The actual hash_function used for hashing; this may differ
        from the ``hash_function`` attribute if, for example, a
        max number of columns was specified.

    metric_: string
        The preferred metric to use with the resulting vetcorization.
        This can be passed to other sklearn compatible models that require a metric.

    References
    ----------
    * Raff, E., & Nicholas, C. (2017, August). An alternative to NCD for large sequences, Lempel-Ziv Jaccard distance.
      In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1007-1015).

    * Bennett, C. H., Gács, P., Li, M., Vitányi, P. M., & Zurek, W. H. (1998). Information distance.
      IEEE Transactions on information theory, 44(4), 1407-1423.

    * Cilibrasi, R., & Vitányi, P. M. (2005). Clustering by compression.
      IEEE Transactions on Information theory, 51(4), 1523-1545.
    """

    def __init__(
            self,
            max_dict_size=1<<16,
            max_columns=1<<16,
            hash_function=identity_hash,
            base_dictionary=None,
            random_state=None
    ):


        self.max_dict_size = max_dict_size
        self.max_columns = max_columns
        self.hash_function = hash_function
        self.base_dictionary = base_dictionary
        self.random_state = random_state

    def fit_transform(self, X, y=None, **fit_params):
        """Train the transformer on a list of strings ``X`` and
        return the resulting vectorization.

         Parameters
         ----------
         X: list of strings
             The strings or byte sequences to apply LZ compression to

         y: None (optional, default=None)
             Ignored.

        Returns
        -------
        vectorization: scipy.sparse.csr_matrix
            The transformed training data.
        """
        if self.max_dict_size <= 1:
            raise ValueError("max_dict_size must be at least 2")
        if self.max_columns is not None:
            if self.max_columns <= 1:
                raise ValueError("max_columns must be at least 2")
            random_state = check_random_state(self.random_state)
            self.hash_function_ = make_hash(self.max_columns, np.int32(random_state.randint(MAX_INT32)))
            self.column_label_dictionary_ = numba.typed.Dict.empty(numba.types.int32, numba.types.int64)
        else:
            self.hash_function_ = self.hash_function
            self.column_label_dictionary_ = numba.typed.Dict.empty(numba.types.unicode_type, numba.types.int64)

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

            encoding_dict = lempel_ziv_based_encode(string, input_dict, self.hash_function_, self.max_dict_size)

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
        self.metric_ = "jaccard"

        return self._train_matrix

    def fit(self, X, y=None, **fit_params):
        """Train the transformer on a list of strings ``X``

        Parameters
        ----------
        X: list of strings
            The strings or byte sequences to apply LZ compression to

        y: None (optional, default=None)
            Ignored.

        Returns
        -------
        self:
            The trained model.
        """
        self.fit_transform(X, y=y, **fit_params)
        return self

    def transform(self, X, y=None):
        """Transform a list of strings ``X`` into a LZ compression
        vectorization using the learned feature space.

         Parameters
         ----------
         X: list of strings
             The strings or byte sequences to apply LZ compression to

         y: None (optional, default=None)
             Ignored.

        Returns
        -------
        vectorization: scipy.sparse.csr_matrix
            The transformed data.
        """
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

            encoding_dict = lempel_ziv_based_encode(string, input_dict, self.hash_function_, self.max_dict_size)

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
    """Create vector representations of strings using a Byte Pair Encoding. This can
    be viewed as a kind of compression vectorizer (since BPE is also a compression scheme),
    or as a form a learned tokenization applicable to arbitrary strings or byte sequences
    that don't have standard tokenizers available.

    The vectorizer has multiple output types specified by ``return_type``:
      * "matrix": an occurrence count matrix as a compression vectorizer
      * "sequences": a list of arrays of integer codes providing the encodings of each string
      * "tokens": a list of lists of string tokens with the vectorizer acting as a tokenizer

    Passing the "tokens" ``return_type`` to the ``NgramVectorizer`` with n=1 will result in
    the same output as the "matrix" ``return_type``, but obviously other ngram sizes are possible
    and may be more useful.

    The "tokens" ``return_type`` can also potentially be used with the
    ``TokenCooccurrenceVectorizer`` to learn vector representations of the individual tokens.

    Parameters
    ----------
    max_vocab_size: int (optional, default=10000)
        The maximum number of distinct codes to use in the encoding dictionary

    min_token_occurrence: int (optional, default=1)
        The minimum number of occurrences of a pair for it to be considered for
        adding as a new token to the encoding dictionary.

    return_type: string (optional, default="matrix")
        The type of data to return upon transforming.
          * "matrix": an occurrence count matrix as a compression vectorizer
          * "sequences": a list of arrays of integer codes providing the encodings of each string
          * "tokens": a list of lists of string tokens with the vectorizer acting as a tokenizer

    Attributes
    ----------
    code_list_: list of pairs of ints
        The pairs merged to create new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code_ + 1``.

    tokens_: list of strings
        The string representations of the new codes. The ``i``th entry is associated to
        the code value ``i + max_char_code_ + 1``.

    max_char_code_: int
        The maximum value of character codes in the training set. For ascii strings
        this is simply 255, but for unicode strings it may be significantly larger. Code
        values associated with new learned tokens begin at ``max_char_code_ + 1``.
    """

    def __init__(self, max_vocab_size=10000, min_token_occurrence=1, return_type="matrix"):
        self.max_vocab_size = max_vocab_size
        self.min_token_occurence = min_token_occurrence
        self.return_type = return_type


    def fit_transform(self, X, y=None, **fit_params):
        """Train the transformer on a list of strings ``X`` and
        return the resulting vectorization.

         Parameters
         ----------
         X: list of strings
             The strings or byte sequences to learn byte pair encoding from and then encode

         y: None (optional, default=None)
             Ignored.

        Returns
        -------
         vectorization: scipy.sparse.csr_matrix or list of array of int or list of lists of strings
             The transformed data, with the return type depending on the value of ``return_type``.
        """
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
        """Train the transformer on a list of strings ``X``

        Parameters
        ----------
        X: list of strings
            The strings or byte sequences to learn byte pair encoding from.

        y: None (optional, default=None)
            Ignored.

        Returns
        -------
        self:
            The trained model.
        """
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X, y=None):
        """Transform a list of strings ``X`` into a byte pair encoding
         vectorization using the learned byte pair encoding.

          Parameters
          ----------
          X: list of strings
              The strings or byte sequences to apply byte pair encoding to.

          y: None (optional, default=None)
              Ignored.

         Returns
         -------
         vectorization: scipy.sparse.csr_matrix or list of array of int or list of lists of strings
             The transformed data, with the return type depending on the value of ``return_type``.
         """
        check_is_fitted(self, ["tokens_", "code_list_", "max_char_code_"])

        encodings = [
            bpe_encode(string, self.code_list_, self.max_char_code_)
            for string in X
        ]

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

