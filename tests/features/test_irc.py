import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.features.irc import add_features
from deeprank2.utils.graph import Graph

from . import build_testgraph


def _run_assertions(graph: Graph):
    assert not np.any(
        [np.isnan(node.features[Nfeat.IRCTOTAL])
            for node in graph.nodes]
    ), 'nan found'
    assert np.any(
        [node.features[Nfeat.IRCTOTAL] > 0
            for node in graph.nodes]
    ), 'no contacts'

    assert np.all(
        [node.features[Nfeat.IRCTOTAL] == sum(node.features[IRCtype] for IRCtype in Nfeat.IRC_FEATURES[:-1])
            for node in graph.nodes]
    ), 'incorrect total'


def test_irc_residue():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail='residue',
        influence_radius=8.5,
        max_edge_length=8.5,
    )
    add_features(pdb_path, graph)
    _run_assertions(graph)


def test_irc_atom():
    pdb_path = "tests/data/pdb/1A0Z/1A0Z.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail='residue',
        influence_radius=4.5,
        max_edge_length=4.5,
    )
    add_features(pdb_path, graph)
    _run_assertions(graph)
