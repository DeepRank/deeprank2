import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.features.exposure import add_features
from deeprank2.utils.graph import Graph

from . import build_testgraph


def _run_assertions(graph: Graph):
    assert np.any(
        node.features[Nfeat.HSE] != 0.0 for node in graph.nodes
    ), 'hse'

    assert np.any(
        node.features[Nfeat.RESDEPTH] != 0.0 for node in graph.nodes
    ), 'resdepth'


def test_exposure_residue():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail='residue',
        influence_radius=8.5,
        max_edge_length=8.5,
    )
    add_features(pdb_path, graph)
    _run_assertions(graph)


def test_exposure_atom():
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail='atom',
        influence_radius=4.5,
        max_edge_length=4.5,
    )
    add_features(pdb_path, graph)
    _run_assertions(graph)
