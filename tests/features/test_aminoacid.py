from pdb2sql import pdb2sql
from deeprankcore.domain.aminoacidlist import serine
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.features.surfacearea import add_features
from deeprankcore.utils.graph import build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_surrounding_residues
from deeprankcore.domain import nodefeatures
from deeprankcore.features.components import add_features # noqa


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
            assert node.features[nodefeatures.DIFFSIZE] > 0
            assert node.features[nodefeatures.DIFFHBDONORS] > 0
