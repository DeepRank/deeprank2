import numpy as np
from deeprankcore.domain.aminoacidlist import amino_acids
from deeprankcore.domain.aminoacidlist import cysteine, selenocysteine, lysine, pyrrolysine

EXCEPTIONS = [
    [cysteine, selenocysteine],
    [lysine, pyrrolysine],
]

def test_all_different_onehot():
    for amino_acid in amino_acids:
        for other in amino_acids:
            if other != amino_acid:
                if other in EXCEPTIONS[0] and amino_acid in EXCEPTIONS[0]:
                    assert np.all(amino_acid.onehot == other.onehot)
                elif other in EXCEPTIONS[1] and amino_acid in EXCEPTIONS[1]:
                    assert np.all(amino_acid.onehot == other.onehot)
                else:
                    assert not np.all(
                        amino_acid.onehot == other.onehot
                    ), f"one-hot index {amino_acid.index} is occupied by both {amino_acid} and {other}"
