import numpy
from pdb2sql import pdb2sql
from deeprankcore.tools.pdb import (
    get_structure,
    get_residue_contact_pairs,
    get_surrounding_residues,
    find_neighbour_atoms,
)
from deeprankcore.models.amino_acid import valine
from deeprankcore.models.structure import AtomicElement


def test_get_structure_complete():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

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
        pdb._close() # pylint: disable=protected-access

    assert structure is not None
    assert structure.chains[0].residues[0].amino_acid is None  # DNA


def test_residue_contact_pairs():

    # get_residue_contact_pairs(pdb_path: str, structure: Structure,
    # chain_id1: str, chain_id2: str, distance_cutoff: float)

    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "1ATN")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue_pairs = get_residue_contact_pairs(pdb_path, structure, "A", "B", 8.5)

    assert len(residue_pairs) > 0


def test_surrounding_residues():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    all_residues = structure.get_chain("A").residues

    # A nicely centered residue
    residue = [r for r in all_residues if r.number == 138][0]

    close_residues = get_surrounding_residues(structure, residue, 10.0)

    assert len(close_residues) > 0, "no close residues found"
    assert len(close_residues) < len(all_residues), "all residues were picked"
    assert residue in close_residues, "the centering residue wasn't included"


def test_neighbour_atoms():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    atoms = structure.get_atoms()

    atom_pairs = find_neighbour_atoms(atoms, 4.5)

    assert len(atom_pairs) > 0, "no atom pairs found"
    assert len(atom_pairs) < numpy.square(len(atoms)), "every two atoms were paired"

    for atom1, atom2 in atom_pairs:
        assert atom1 != atom2, f"atom {atom1} was paired with itself"
