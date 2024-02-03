from pdb2sql import pdb2sql

from deeprank2.domain.aminoacidlist import arginine, glutamate
from deeprank2.utils.buildgraph import get_structure
from deeprank2.utils.parsing import atomic_forcefield


def test_atomic_forcefield() -> None:
    pdb = pdb2sql("tests/data/pdb/101M/101M.pdb")
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()  # noqa: SLF001

    # The arginine C-zeta should get a positive charge
    arg = next(r for r in structure.get_chain("A").residues if r.amino_acid == arginine)
    cz = next(a for a in arg.atoms if a.name == "CZ")
    assert atomic_forcefield.get_charge(cz) == 0.640

    # The glutamate O-epsilon should get a negative charge
    glu = next(r for r in structure.get_chain("A").residues if r.amino_acid == glutamate)
    oe2 = next(a for a in glu.atoms if a.name == "OE2")
    assert atomic_forcefield.get_charge(oe2) == -0.800

    # The forcefield should treat terminal oxygen differently
    oxt = next(a for a in structure.get_atoms() if a.name == "OXT")
    o = next(a for a in oxt.residue.atoms if a.name == "O")
    assert atomic_forcefield.get_charge(oxt) == -0.800
    assert atomic_forcefield.get_charge(o) == -0.800
