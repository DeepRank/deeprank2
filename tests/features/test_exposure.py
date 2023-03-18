import numpy as np
from . import build_testgraph
from deeprankcore.utils.graph import Graph
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.features.exposure import add_features


def _run_assertions(graph: Graph):
    assert np.any(
        node.features[Nfeat.HSE] != 0.0 for node in graph.nodes
    ), 'hse'

    assert np.any(
        node.features[Nfeat.RESDEPTH] != 0.0 for node in graph.nodes
    ), 'resdepth'


def test_residue_features():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph = build_testgraph(pdb_path, 'residue', 8.5)

    add_features(pdb_path, graph)
    _run_assertions(graph)


def test_atom_features():
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    graph = build_testgraph(pdb_path, 'atom', 4.5)

    add_features(pdb_path, graph)
    _run_assertions(graph)
  