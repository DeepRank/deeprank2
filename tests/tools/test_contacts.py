import numpy
from pdb2sql import pdb2sql
from deeprank_gnn.tools.pdb import get_structure, get_atomic_contacts, get_residue_contacts


def _get_atomic_contact(structure, residue1_number, atom1_name, residue2_number, atom2_name):

    atom1 = None
    for residue in structure.chains[0].residues:
        if residue.number == residue1_number:
            for atom in residue.atoms:
                if atom.name == atom1_name:
                    atom1 = atom

    assert atom1 is not None, "atom 1 not found"

    atom2 = None
    for residue in structure.chains[0].residues:
        if residue.number == residue2_number:
            for atom in residue.atoms:
                if atom.name == atom2_name:
                    atom2 = atom

    assert atom2 is not None, "atom 2 not found"


    contacts = get_atomic_contacts([atom1, atom2])
    return contacts[0]


def test_atomic_contacts():

    pdb = pdb2sql("tests/data/pdb/101M/101M.pdb")
    try:
        structure = get_structure(pdb, "101m")
    finally:
        pdb._close()

    # MET 0: N - CA, very close, should have positive vanderwaals energy
    close_contact = _get_atomic_contact(structure, 0, "N", 0, "CA")
    assert not numpy.isnan(close_contact.vanderwaals_potential)
    assert close_contact.vanderwaals_potential > 0.0, close_contact.vanderwaals_potential

    # MET 0 N - ASP 27 CB, very far, should have negative vanderwaals energy
    far_contact =  _get_atomic_contact(structure, 0, "N", 27, "CB")
    assert not numpy.isnan(far_contact.vanderwaals_potential)
    assert far_contact.vanderwaals_potential < 0.0, far_contact.vanderwaals_potential

    # MET 0 N - PHE 138 CG, intermediate distance,
    # should have more negative vanderwaals energy than the war interactions
    intermediate_contact = _get_atomic_contact(structure, 0, "N", 138, "CG")
    assert not numpy.isnan(intermediate_contact.vanderwaals_potential)
    assert intermediate_contact.vanderwaals_potential < far_contact.vanderwaals_potential, \
        intermediate_contact.vanderwaals_potential

    # ARG 139 CZ - GLU 136 OE2, very close attractive electrostatic energy
    close_attractive_contact = _get_atomic_contact(structure, 139, "CZ", 136, "OE2")
    assert not numpy.isnan(close_attractive_contact.electrostatic_potential)
    assert close_attractive_contact.electrostatic_potential < 0.0, close_attractive_contact.electrostatic_potential

    # ARG 139 CZ - ASP 20 OD2, far attractive electrostatic energy
    far_attractive_contact = _get_atomic_contact(structure, 139, "CZ", 20, "OD2")
    assert not numpy.isnan(far_attractive_contact.electrostatic_potential)
    assert far_attractive_contact.electrostatic_potential > close_attractive_contact.electrostatic_potential
    assert far_attractive_contact.electrostatic_potential < 0.0, far_attractive_contact.electrostatic_potential

    # GLU 109 OE2 - GLU 105 OE1, repulsive electrostatic energy
    repulsive_contact = _get_atomic_contact(structure, 109, "OE2", 105, "OE1")
    assert not numpy.isnan(repulsive_contact.electrostatic_potential)
    assert repulsive_contact.electrostatic_potential > 0.0, repulsive_contact.electrostatic_potential


def test_residue_contacts():

    # Take a small structure, so that the memory consumption is low!
    pdb = pdb2sql("tests/data/pdb/1crn/1CRN.pdb")
    try:
        structure = get_structure(pdb, "1crn")
    finally:
        pdb._close()

    contacts = get_residue_contacts(structure.chains[0].residues)

    assert not numpy.any(numpy.isnan([contact.electrostatic_potential for contact in contacts]))
    assert not numpy.any(numpy.isnan([contact.vanderwaals_potential for contact in contacts]))
    assert numpy.all([contact.distance > 0.0 for contact in contacts])
