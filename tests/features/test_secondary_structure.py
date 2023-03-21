import numpy as np
from . import build_testgraph
from deeprankcore.domain import nodestorage as Nfeat

from deeprankcore.features.secondary_structure import add_features
from deeprankcore.utils.graph import Graph


def _run_assertions(graph: Graph, node_info_list: list):
    node_info_list.sort()
    
    # Check that all nodes have exactly 1 secondary structure type
    assert np.all([node.features[Nfeat.SECSTRUCT].sum() == 1.0 for node in graph.nodes]), 'one hot error'


    # check ground truth examples
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
        ), f'Ground truth examples: res {res[1]} {res[0]} is not {res[3]}'


    # check entire DSSP file
    # residue number @ pos 5-10, chain_id @ pos 11, secondary structure @ pos 16
    with open('tests/data/dssp/1ak4.dssp.txt') as file:
        dssp_lines = [line.rstrip() for line in file]

    for node in node_info_list:
        dssp_line = [line for line in dssp_lines 
                        if (line[5:10] == str(node[0]).rjust(5) and line[11] == node[1])][0]
        dssp_code = dssp_line[16]
        if dssp_code in [' ', 'S', 'T']:
            assert np.array_equal(node[2],np.array([0., 0., 1.])), f'Full file test: res {node[1]}{node[0]} is not C'
        elif dssp_code in ['B', 'E']:
            assert np.array_equal(node[2],np.array([0., 1., 0.])), f'Full file test: res {node[1]}{node[0]} is not E'
        elif dssp_code in ['G', 'H', 'I']:
            assert np.array_equal(node[2],np.array([1., 0., 0.])), f'Full file test: res {node[1]}{node[0]} is not H'
        else:
            raise ValueError(f'Unexpected secondary structure type found at {node[1]}{node[0]}')
    
    
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
