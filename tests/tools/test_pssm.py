from pdb2sql import pdb2sql
from deeprank_gnn.tools.pssm import add_pssms
from deeprank_gnn.tools.pdb import get_structure
from deeprank_gnn.domain.amino_acid import alanine


def test_add_pssm():
    pdb = pdb2sql("tests/data/pdb/1ATN/1ATN_1w.pdb")
    try:
        structure = get_structure(pdb, "1ATN")
    finally:
        pdb._close()

    add_pssms(structure, {"A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
                          "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"})

    # Verify that each residue is present and that the data makes sense:
    for chain in structure.chains:
        for residue in chain.residues:
            assert residue in chain.pssm
            assert type(chain.pssm[residue].information_content) == float
            assert type(chain.pssm[residue].conservations[alanine]) == float
