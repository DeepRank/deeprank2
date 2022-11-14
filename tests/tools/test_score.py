

from deeprankcore.tools.target import compute_targets


def test_compute_targets():

    _ = compute_targets("tests/data/pdb/1ATN/1ATN_1w.pdb", "tests/data/ref/1ATN/1ATN.pdb")
