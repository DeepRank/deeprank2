from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.molstruct.structure import PDBStructure
from deeprankcore.features.secondary_structure import add_features
from deeprankcore.utils.graph import build_residue_graph, build_atomic_graph, Graph
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.domain import nodestorage as Nfeat

def _load_pdb_structure(pdb_path: str, id_: str) -> PDBStructure:
    """
    Load PDB structure from a PDB file.

    Args:
        pdb_path (str): The file path of the PDB file.
        id_ (str): The PDB structure ID.

    Returns:
        PDBStructure: The loaded PDB structure.
    """
    pdb = pdb2sql(pdb_path)
    try:
        return get_structure(pdb, id_)
    finally:
        pdb._close()  # pylint: disable=protected-access


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
