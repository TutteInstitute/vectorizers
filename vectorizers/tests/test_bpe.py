import numpy as np
import pytest  # noqa

from vectorizers import BytePairEncodingVectorizer
from vectorizers import NgramVectorizer
from vectorizers.mixed_gram_vectorizer import to_unicode

from . import raw_string_data


def test_bpe_vectorizer_basic():
    bpe = BytePairEncodingVectorizer()
    result1 = bpe.fit_transform(raw_string_data)
    result2 = bpe.transform(raw_string_data)
    assert np.allclose(result1.toarray(), result2.toarray())


def test_bpe_tokens_ngram_matches():
    bpe1 = BytePairEncodingVectorizer(return_type="matrix")
    bpe2 = BytePairEncodingVectorizer(return_type="tokens")

    result1 = bpe1.fit_transform(raw_string_data)
    token_dictionary = {
        to_unicode(code, bpe1.tokens_, bpe1.max_char_code_): n
        for code, n in bpe1.column_label_dictionary_.items()
    }

    tokens = bpe2.fit_transform(raw_string_data)
    result2 = NgramVectorizer(token_dictionary=token_dictionary).fit_transform(tokens)

    assert np.allclose(result1.toarray(), result2.toarray())


def test_bpe_bad_params():
    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(max_vocab_size=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(min_token_occurrence=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(return_type=-1)
        bpe.fit(raw_string_data)

    with pytest.raises(ValueError):
        bpe = BytePairEncodingVectorizer(return_type="nonsense")
        bpe.fit(raw_string_data)
