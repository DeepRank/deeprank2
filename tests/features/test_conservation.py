import numpy as np
from . import build_testgraph
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.features.irc import add_features
from deeprankcore.domain.aminoacidlist import alanine
from deeprankcore.molstruct.variant import SingleResidueVariant


def test_conservation_residue():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    graph = build_testgraph(pdb_path, 10, 'residue', 25)
    chain = graph.nodes[0].id.chain
    variant_residue = chain.residues[25]
    variant = SingleResidueVariant(variant_residue, alanine)

    add_features(pdb_path, graph, variant)

    for feature_name in (
        Nfeat.PSSM,
        Nfeat.DIFFCONSERVATION,
        Nfeat.CONSERVATION,
        Nfeat.INFOCONTENT,
    ):
        assert np.any([node.features[feature_name] != 0.0 for node in graph.nodes]), f'all 0s found for {feature_name}'


def test_conservation_atom():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    graph = build_testgraph(pdb_path, 10, 'atom', 25)
    chain = graph.nodes[0].id.residue.chain
    variant_residue = chain.residues[25]
    variant = SingleResidueVariant(variant_residue, alanine)

    add_features(pdb_path, graph, variant)

    for feature_name in (
        Nfeat.PSSM,
        Nfeat.DIFFCONSERVATION,
        Nfeat.CONSERVATION,
        Nfeat.INFOCONTENT,
    ):
        assert np.any([node.features[feature_name] != 0.0 for node in graph.nodes]), f'all 0s found for {feature_name}'
