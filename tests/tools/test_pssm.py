from pdb2sql import pdb2sql
from deeprankcore.tools.pssm import parse_pssm
from deeprankcore.tools.pdb import get_structure
from deeprankcore.models.amino_acid import alanine


def test_add_pssm():
    pdb = pdb2sql("tests/data/pdb/1ATN/1ATN_1w.pdb")
    try:
        structure = get_structure(pdb, "1ATN")
    finally:
        pdb._close() # pylint: disable=protected-access

    for chain in structure.chains:
        with open(f"tests/data/pssm/1ATN/1ATN.{chain.id}.pdb.pssm", "rt", encoding="utf-8") as f:
            chain.pssm = parse_pssm(f, chain)

    # Verify that each residue is present and that the data makes sense:
    for chain in structure.chains:
        for residue in chain.residues:
            assert residue in chain.pssm
            assert isinstance(chain.pssm[residue].information_content, float)
            assert isinstance(chain.pssm[residue].conservations[alanine], float)
