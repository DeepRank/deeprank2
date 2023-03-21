import numpy as np
from . import build_testgraph
from deeprankcore.domain import nodestorage as Nfeat

from deeprankcore.features.secondary_structure import add_features
from deeprankcore.utils.graph import Graph


def _run_assertions(graph: Graph, node_info_list: list):
    node_info_list.sort()
    
    # Check that all nodes have exactly 1 secondary structure type
    assert np.all([node.features[Nfeat.SECSTRUCT].sum() == 1.0 for node in graph.nodes]), 'sum != 1'

    residues = [
        (90, 'D', np.array([0., 0., 1.]), 'C'),
        (113, 'C', np.array([0., 1., 0.]), 'E'),
        (121, 'C', np.array([1., 0., 0.]), 'H'),
    ]

    for res in residues:
        node_list = [node_info for node_info in node_info_list if (node_info[0] == res[0] and node_info[1] == res[1])]
        assert len(node_list) > 0, f'no nodes detected in {res[1]} {res[0]}'
        assert np.all(
            [np.array_equal(node_info[2], res[2])
                for node_info in node_list]
        ), f'sec struct for {res[1]} {res[0]} is not {res[3]}'        


    
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
