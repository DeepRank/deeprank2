import numpy as np
from deeprank2.domain.aminoacidlist import amino_acids
from deeprank2.domain.aminoacidlist import cysteine
from deeprank2.domain.aminoacidlist import lysine
from deeprank2.domain.aminoacidlist import pyrrolysine
from deeprank2.domain.aminoacidlist import selenocysteine


# Exceptions selenocysteine and pyrrolysine are due to them having the same index as their canonical counterpart.
# This is not an issue while selenocysteine and pyrrolysine are not part of amino_acids.
# However, the code to deal with them is already included below
EXCEPTIONS = [
    [cysteine, selenocysteine],
    [lysine, pyrrolysine],
]


def test_all_different_onehot():
    for amino_acid in amino_acids:
        for other in amino_acids:
            if other != amino_acid:
                try:
                    assert not np.all(amino_acid.onehot == other.onehot)
                except AssertionError as e:
                    if other in EXCEPTIONS[0] and amino_acid in EXCEPTIONS[0]:
                        assert np.all(amino_acid.onehot == other.onehot)
                    elif other in EXCEPTIONS[1] and amino_acid in EXCEPTIONS[1]:
                        assert np.all(amino_acid.onehot == other.onehot)
                    else:
                        raise AssertionError(f"one-hot index {amino_acid.index} is occupied by both {amino_acid} and {other}") from e
