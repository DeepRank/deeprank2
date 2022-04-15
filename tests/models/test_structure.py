import pickle

from pdb2sql import pdb2sql

from deeprank_gnn.models.structure import AtomicElement
from deeprank_gnn.tools.pdb import get_structure


def test_element():
    AtomicElement.C.onehot


def test_pickle():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    s = pickle.dumps(structure)

    pickle.loads(s)
