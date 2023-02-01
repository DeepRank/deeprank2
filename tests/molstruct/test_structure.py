
import pickle
from pdb2sql import pdb2sql

from deeprankcore.utils.buildgraph import get_structure


def test_serialization():

    path = "tests/data/pdb/101M/101M.pdb"
    pdb = pdb2sql(path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    s = pickle.dumps(structure)
    loaded_structure = pickle.loads(s)

    assert loaded_structure == structure
