from pdb2sql import pdb2sql

from deeprankcore.models.amino_acid import serine
from deeprankcore.models.variant import SingleResidueVariant
from deeprankcore.feature.sasa import add_features
from deeprankcore.tools.graph import build_residue_graph
from deeprankcore.tools.pdb import get_structure, get_surrounding_residues
from deeprankcore.domain.features import nodefeats
from deeprankcore.feature.amino_acid import add_features # noqa


def test_add_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = structure.chains[0].residues[25]
    variant = SingleResidueVariant(residue, serine)  # GLY -> SER

    residues = get_surrounding_residues(structure, residue, 10.0)
    assert len(residues) > 0

    graph = build_residue_graph(residues, "101m-25", 4.5)

    add_features(pdb_path, graph, variant)

    for node in graph.nodes:
        if node.id == variant.residue:  # GLY -> SER
            assert node.features[nodefeats.DIFFSIZE] > 0
            assert node.features[nodefeats.DIFFHBDONORS] > 0
