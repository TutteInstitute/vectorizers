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


def test_bpe_trash_token():
    bpe = BytePairEncodingVectorizer(return_type="sequences").fit(raw_string_data)
    tokenized_no_trash = bpe.transform(raw_string_data)
    assert len(tokenized_no_trash) == len(raw_string_data)
    assert not any(0 in tokens for tokens in tokenized_no_trash)
    tokenized_with_trash = bpe.transform(["asdf{qwer"])
    assert len(tokenized_with_trash) == 1
    assert 0 in tokenized_with_trash[0]


def test_bpe_set_max_char_code():
    MCC = 65535
    bpe = BytePairEncodingVectorizer(
        max_char_code=MCC,
        return_type="sequences"
    ).fit(raw_string_data)
    tokens = bpe.transform(raw_string_data)
    largest_char = max(max(ord(c) for c in s) for s in raw_string_data)
    assert largest_char < 126
    assert all(
        all(
            token <= largest_char or token > MCC
            for token in seq
        )
        for seq in tokens
    )
    tokens_strange = bpe.transform([chr(126) + chr(2000) + chr(60000)])
    assert 1 == len(tokens_strange)
    assert np.all([126, 2000, 60000] == tokens_strange[0])


def test_bpe_set_max_char_code_too_low():
    bpe = BytePairEncodingVectorizer(max_char_code=50).fit(raw_string_data)
    assert max(max(ord(c) for c in s) for s in raw_string_data) == bpe.max_char_code_


@pytest.mark.parametrize(
    "name,max_expected",
    [
        ("ascii", 127),
        ("common", 2047),
        ("bmp", 65535),
        ("unicode", 1_114_111),
    ]
)
def test_bpe_max_char_code_limits(name, max_expected):
    assert max_expected == BytePairEncodingVectorizer(
        max_char_code=name
    ).fit(raw_string_data).max_char_code_


def test_bpe_max_char_code_limit_wrong():
    with pytest.raises(ValueError):
        BytePairEncodingVectorizer(max_char_code="utf8").fit(raw_string_data)


def test_bpe_contract_pair_single_token_training():
    seqs_tokens = BytePairEncodingVectorizer(return_type="tokens").fit_transform([
        "asdfqwerty",
        "asdf",
        "qwzxasdfcv"
    ])
    assert [
        ["asdf", "qw", "e", "r", "t", "y"],
        ["asdf"],
        ["qw", "z", "x", "asdf", "c", "v"],
    ] == seqs_tokens


def test_bpe_contract_pair_single_token_inference():
    bpe = BytePairEncodingVectorizer(return_type="tokens").fit([
        "asdfqwerty",
        "asdfg",
        "qwzxasdfcv",
    ])
    assert [["asdf"]] == bpe.transform(["asdf"])
