from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.domain.aminoacidlist import alanine
from deeprankcore.utils.parsing.pssm import parse_pssm
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.features.conservation import add_features
from deeprankcore.utils.graph import build_atomic_graph
from deeprankcore.utils.buildgraph import get_structure, get_surrounding_residues
from . import build_testgraph


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
