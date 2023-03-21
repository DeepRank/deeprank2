import numpy as np
from . import build_testgraph
from deeprankcore.domain import nodestorage as Nfeat

from deeprankcore.features.secondary_structure import add_features
from deeprankcore.utils.graph import Graph


def _run_assertions(graph: Graph, node_info_list: list):
    node_info_list.sort()
    
    # Check that all nodes have exactly 1 secondary structure type
    assert np.all([node.features[Nfeat.SECSTRUCT].sum() == 1.0 for node in graph.nodes]), 'sum != 1'

    # Example of a coil (C)
    example1 = (90, 'D')
    example1_list = [node_info for node_info in node_info_list if (node_info[0] == example1[0] and node_info[1] == example1[1])]
    assert len(example1_list) > 0, f'no nodes detected in {example1[1]} {example1[0]}'
    assert np.all(
        [np.array_equal(node_info[2], np.array([0., 0., 1.]))
            for node_info in example1_list]
    ), f'sec struct for {example1[1]} {example1[0]} is not C'

    # Example of an extended region (E)
    example2 = (113, 'C')
    example2_list = [node_info for node_info in node_info_list if (node_info[0] == example2[0] and node_info[1] == example2[1])]
    assert len(example2_list) > 0, f'no nodes detected in {example2[1]} {example2[0]}'
    assert np.all(
        [np.array_equal(node_info[2], np.array([0., 1., 0.]))
            for node_info in example2_list]
    ), f'sec struct for {example2[1]} {example2[0]} is not E'

    # Example of an extended helix (H)
    example3 = (121, 'C')
    example3_list = [node_info for node_info in node_info_list if (node_info[0] == example3[0] and node_info[1] == example3[1])]
    assert len(example3_list) > 0, f'no nodes detected in {example3[1]} {example3[0]}'
    assert np.all(
        [np.array_equal(node_info[2], np.array([1., 0., 0.]))
            for node_info in example3_list]
    ), f'sec struct for {example3[1]} {example3[0]} is not H'

    
def test_secondary_structure_residue():
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    graph = build_testgraph(pdb_path, 8.5, 'residue')
    add_features(pdb_path, graph)

    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.number, 
                       node.id.chain.id, 
                       node.features[Nfeat.SECSTRUCT]] 
                            for node in graph.nodes]
    _run_assertions(graph, node_info_list)


def test_secondary_structure_atom():
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    graph = build_testgraph(pdb_path, 4.5, 'atom')
    add_features(pdb_path, graph)
    
    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.residue.number, 
                       node.id.residue.chain.id, 
                       node.features[Nfeat.SECSTRUCT]] 
                            for node in graph.nodes]
    _run_assertions(graph, node_info_list)
