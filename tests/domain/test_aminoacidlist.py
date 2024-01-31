import numpy as np

from deeprank2.domain.aminoacidlist import amino_acids, cysteine, lysine, pyrrolysine, selenocysteine

# Exceptions selenocysteine and pyrrolysine are due to them having the same index as their canonical counterpart.
# This is not an issue while selenocysteine and pyrrolysine are not part of amino_acids.
# However, the code to deal with them is already included below
EXCEPTIONS = [
    [cysteine, selenocysteine],
    [lysine, pyrrolysine],
]


def test_all_different_onehot() -> None:
    for aa1, aa2 in zip(amino_acids, amino_acids, strict=True):
        if aa1 == aa2:
            continue

        try:
            assert not np.all(aa1.onehot == aa2.onehot)
        except AssertionError as e:
            if (aa1 in EXCEPTIONS[0] and aa2 in EXCEPTIONS[0]) or (aa1 in EXCEPTIONS[1] and aa2 in EXCEPTIONS[1]):
                assert np.all(aa1.onehot == aa2.onehot)
            else:
                raise AssertionError(f"One-hot index {aa1.index} is occupied by both {aa1} and {aa2}") from e
