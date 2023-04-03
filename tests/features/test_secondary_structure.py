import numpy as np

from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.features.secondary_structure import (SecondarySctructure,
                                                       _classify_secstructure,
                                                       add_features)

from . import build_testgraph


def test_secondary_structure_residue():
    test_case = '9api' # properly formatted pdb file
    pdb_path = f"tests/data/pdb/{test_case}/{test_case}.pdb"
    graph = build_testgraph(pdb_path, 10, 'residue')
    add_features(pdb_path, graph)

    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.number, 
                       node.id.chain.id, 
                       node.features[Nfeat.SECSTRUCT]] 
                            for node in graph.nodes]
    print(node_info_list)

    # Check that all nodes have exactly 1 secondary structure type
    assert np.all([np.sum(node.features[Nfeat.SECSTRUCT]) == 1.0 for node in graph.nodes]), 'one hot encoding error'

    # check ground truth examples
    residues = [
        (267, 'A', ' ', SecondarySctructure.COIL),
        (46, 'A', 'S', SecondarySctructure.COIL),
        (104, 'A', 'T', SecondarySctructure.COIL),
        # (None, '', 'P', SecondarySctructure.COIL), # not found in test file
        (194, 'A', 'B', SecondarySctructure.STRAND),
        (385, 'B', 'E', SecondarySctructure.STRAND),
        (235, 'A', 'G', SecondarySctructure.HELIX),
        (263, 'A', 'H', SecondarySctructure.HELIX),
        # (0, '', 'I', SecondarySctructure.HELIX), # not found in test file
    ]

    for res in residues:
        node_list = [node_info for node_info in node_info_list if (node_info[0] == res[0] and node_info[1] == res[1])]
        assert len(node_list) > 0, f'no nodes detected in {res[1]} {res[0]}'
        assert np.all(
            [np.array_equal(node_info[2], _classify_secstructure(res[2]).onehot)
                for node_info in node_list]
        ), f'Ground truth examples: res {res[1]} {res[0]} is not {(res[2])}.'
        assert np.all(
            [np.array_equal(node_info[2], res[3].onehot)
                for node_info in node_list]
        ), f'Ground truth examples: res {res[1]} {res[0]} is not {res[3]}.'


def test_secondary_structure_atom():
    test_case = '1ak4' # ATOM list
    pdb_path = f"tests/data/pdb/{test_case}/{test_case}.pdb"
    graph = build_testgraph(pdb_path, 4.5, 'atom')
    add_features(pdb_path, graph)
    
    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.residue.number, 
                       node.id.residue.chain.id, 
                       node.features[Nfeat.SECSTRUCT]] 
                            for node in graph.nodes]

    # check entire DSSP file
    # residue number @ pos 5-10, chain_id @ pos 11, secondary structure @ pos 16
    with open(f'tests/data/dssp/{test_case}.dssp.txt', encoding="utf8") as file:
        dssp_lines = [line.rstrip() for line in file]

    for node in node_info_list:
        dssp_line = [line for line in dssp_lines 
                        if (line[5:10] == str(node[0]).rjust(5) and line[11] == node[1])][0]
        dssp_code = dssp_line[16]
        if dssp_code in [' ', 'S', 'T']:
            assert np.array_equal(node[2],SecondarySctructure.COIL.onehot), f'Full file test: res {node[1]}{node[0]} is not a COIL'
        elif dssp_code in ['B', 'E']:
            assert np.array_equal(node[2],SecondarySctructure.STRAND.onehot), f'Full file test: res {node[1]}{node[0]} is not a STRAND'
        elif dssp_code in ['G', 'H', 'I']:
            assert np.array_equal(node[2],SecondarySctructure.HELIX.onehot), f'Full file test: res {node[1]}{node[0]} is not a HELIX'
        else:
            raise ValueError(f'Unexpected secondary structure type found at {node[1]}{node[0]}')
