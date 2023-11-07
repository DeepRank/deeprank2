import pickle
from multiprocessing.connection import _ForkingPickler

from deeprank2.molstruct.structure import PDBStructure
from deeprank2.utils.buildgraph import get_structure
from pdb2sql import pdb2sql


def _get_structure(path) -> PDBStructure:
    pdb = pdb2sql(path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    assert structure is not None

    return structure


def test_serialization_pickle():
    structure = _get_structure("tests/data/pdb/101M/101M.pdb")

    s = pickle.dumps(structure)
    loaded_structure = pickle.loads(s)

    assert loaded_structure == structure
    assert loaded_structure.get_chain("A") == structure.get_chain("A")
    assert loaded_structure.get_chain("A").get_residue(0) == structure.get_chain("A").get_residue(0)
    assert loaded_structure.get_chain("A").get_residue(0).amino_acid == structure.get_chain("A").get_residue(0).amino_acid
    assert loaded_structure.get_chain("A").get_residue(0).atoms[0] == structure.get_chain("A").get_residue(0).atoms[0]


def test_serialization_fork():
    structure = _get_structure("tests/data/pdb/101M/101M.pdb")

    s = _ForkingPickler.dumps(structure)
    loaded_structure = _ForkingPickler.loads(s)

    assert loaded_structure == structure
    assert loaded_structure.get_chain("A") == structure.get_chain("A")
    assert loaded_structure.get_chain("A").get_residue(0) == structure.get_chain("A").get_residue(0)
    assert loaded_structure.get_chain("A").get_residue(0).amino_acid == structure.get_chain("A").get_residue(0).amino_acid
    assert loaded_structure.get_chain("A").get_residue(0).atoms[0] == structure.get_chain("A").get_residue(0).atoms[0]
