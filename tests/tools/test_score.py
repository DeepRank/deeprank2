

from deeprankcore.tools.score import get_all_scores


def test_get_all_scores():

    _ = get_all_scores("tests/data/pdb/1ATN/1ATN_1w.pdb", "tests/data/ref/1ATN/1ATN.pdb")
