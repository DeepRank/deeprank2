from pdb2sql import pdb2sql
import numpy
from deeprankcore.models.amino_acid import alanine
from deeprankcore.tools.pssm import parse_pssm
from deeprankcore.models.variant import SingleResidueVariant
from deeprankcore.feature.pssm import add_features
from deeprankcore.tools.graph import build_atomic_graph
from deeprankcore.tools.pdb import get_structure, get_surrounding_residues
from deeprankcore.domain.features import nodefeats as Nfeat



def test_add_features():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101m")
    finally:
        pdb._close() # pylint: disable=protected-access

    chain = structure.get_chain("A")
    with open("tests/data/pssm/101M/101M.A.pdb.pssm", "rt", encoding="utf-8") as f:
        chain.pssm = parse_pssm(f, chain)

    variant_residue = chain.residues[25]

    variant = SingleResidueVariant(variant_residue, alanine)

    residues = get_surrounding_residues(structure, variant_residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-25-atom", 4.5)
    add_features(pdb_path, graph, variant)

    for feature_name in (
        Nfeat.PSSM,
        Nfeat.DIFFCONSERVATION,
        Nfeat.CONSERVATION,
        Nfeat.INFOCONTENT,
    ):
        assert numpy.any([node.features[feature_name] != 0.0 for node in graph.nodes])
