import numpy
from pdb2sql import pdb2sql

from deeprankcore.feature.atom import add_features
from deeprankcore.tools.graph import build_atomic_graph
from deeprankcore.models.structure import Chain, Residue
from deeprankcore.tools.pdb import get_structure, get_surrounding_residues
from deeprankcore.domain.features import nodefeats


def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue

    raise ValueError(f"Not found: {number}")


def test_add_features(): # copied first part, up to add_features, from test_sasa

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = _get_residue(structure.chains[0], 108)
    residues = get_surrounding_residues(structure, residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-108-atom", 4.5)
    add_features(pdb_path, graph)

    assert not any(numpy.isnan(node.features[nodefeats.ATOMCHARGE]) for node in graph.nodes)
    assert not any(numpy.isnan(node.features[nodefeats.PDBOCCUPANCY]) for node in graph.nodes)
