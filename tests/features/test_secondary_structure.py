from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.molstruct.structure import PDBStructure
from deeprankcore.features.secondary_structure import add_features
from deeprankcore.utils.graph import build_residue_graph, build_atomic_graph, Graph
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.utils.buildgraph import get_residue_contact_pairs
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
    
    # Check if the sum of secondary structure features equals 1.0 for all nodes
    assert np.any(
        node.features[Nfeat.SECSTRUCT].sum() == 1.0 for node in graph.nodes
    )

    # Check example 1
    assert node_info_list[0][0] == 1
    assert node_info_list[0][1] == 'D'
    assert np.array_equal(node_info_list[0][2], np.array([0., 0., 1.]))

    # Check example 2
    assert node_info_list[255][0] == 129
    assert node_info_list[255][1] == 'C'
    assert np.array_equal(node_info_list[255][2], np.array([0., 1., 0.]))

    # Check example 3
    assert node_info_list[226][0] == 114
    assert node_info_list[226][1] == 'D'
    assert np.array_equal(node_info_list[226][2], np.array([1., 0., 0.]))

    
def test_secondary_structure_residue():
    # Load test PDB file and create a residue graph
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    structure = _load_pdb_structure(pdb_path, "1ak4")
    residues = structure.chains[0].residues + structure.chains[1].residues
    graph = build_residue_graph(residues, "1ak4", 8.5)

    # Add secondary structure features to the graph nodes
    add_features(pdb_path, graph)

    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.number, node.id.chain.id, node.features[Nfeat.SECSTRUCT]] for node in graph.nodes]

    _run_assertions(graph, node_info_list)


def test_secondary_structure_atom():
    pdb_path = "tests/data/pdb/1A0Z/1A0Z.pdb"
    structure = _load_pdb_structure(pdb_path, "1A0Z")

    atoms = set([])
    for residue1, residue2 in get_residue_contact_pairs(
        pdb_path, structure, "A", "B", 4.5
    ):
        for atom in residue1.atoms:
            atoms.add(atom)
        for atom in residue2.atoms:
            atoms.add(atom)
    atoms = list(atoms)

    graph = build_atomic_graph(atoms, "1A0Z", 4.5)

    add_features(pdb_path, graph)
    
    # Create a list of node information (residue number, chain ID, and secondary structure features)
    node_info_list = [[node.id.residue.number, node.id.residue.chain.id, node.features[Nfeat.SECSTRUCT]] for node in graph.nodes]

    _run_assertions(graph, node_info_list)
