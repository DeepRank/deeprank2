from pdb2sql import pdb2sql
from deeprank_gnn.tools.pdb import get_structure, get_residue_contact_pairs
from deeprank_gnn.domain.amino_acid import valine
from deeprank_gnn.models.structure import AtomicElement
from deeprank_gnn.models.environment import Environment


def test_get_structure_complete():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    assert structure is not None

    assert len(structure.chains) == 1
    chain = structure.chains[0]
    assert chain.id == "A"

    assert len(chain.residues) == 154
    residue = chain.residues[1]
    assert residue.number == 1
    assert residue.chain == chain
    assert residue.amino_acid == valine

    assert len(residue.atoms) == 7
    atom = residue.atoms[1]
    assert atom.name == "CA"
    assert atom.position[0] == 27.263  # x coord from PDB file
    assert atom.element == AtomicElement.C
    assert atom.residue == residue


def test_get_structure_from_nmr_with_dna():
    pdb_path = "tests/data/pdb/1A6B/1A6B.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    assert structure is not None
    assert structure.chains[0].residues[0].amino_acid is None  # DNA


def test_residue_contact_pairs():

    environment = Environment(pdb_root="tests/data/pdb", device="cpu")

    pair_distances = get_residue_contact_pairs(environment, "1ATN", "A", "B", 8.5)

    assert len(pair_distances) > 0
    assert list(pair_distances.values())[0] < 8.5
