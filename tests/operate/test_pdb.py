from pdb2sql import pdb2sql
from deeprank_gnn.operate.pdb import get_structure
from deeprank_gnn.domain.amino_acid import valine
from deeprank_gnn.models.structure import AtomicElement


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
