import itertools
from collections.abc import Iterable

def flatten(list_of_seq):
    assert(isinstance(list_of_seq, Iterable))
    if type(list_of_seq[0]) in (list, tuple):
        return tuple(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq

