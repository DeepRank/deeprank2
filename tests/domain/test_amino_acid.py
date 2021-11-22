from deeprank_gnn.domain.amino_acid import amino_acids


def test_all_different_onehot():

    codes = {}

    for amino_acid in amino_acids:
        assert amino_acid.onehot not in codes, "{} is occupied by both {} and {}".format(amino_acid.onehot, codes[amino_acid.onehot], amino_acid)
        codes[amino_acid.onehot] = amino_acid
