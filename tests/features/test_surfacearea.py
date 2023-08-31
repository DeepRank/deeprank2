import numpy as np
from deeprank2.features.surfacearea import add_features

from deeprank2.domain import nodestorage as Nfeat

from . import build_testgraph


def _find_residue_node(graph, chain_id, residue_number):
    for node in graph.nodes:
        residue = node.id
        if residue.chain.id == chain_id and residue.number == residue_number:
            return node
    raise ValueError(f"Not found: {chain_id} {residue_number}")


def _find_atom_node(graph, chain_id, residue_number, atom_name):
    for node in graph.nodes:
        atom = node.id
        if (
            atom.residue.chain.id == chain_id
            and atom.residue.number == residue_number
            and atom.name == atom_name
        ):
            return node
    raise ValueError(f"Not found: {chain_id} {residue_number} {atom_name}")


def test_bsa_residue():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph = build_testgraph(pdb_path, 8.5, 'residue')
    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_residue_node(graph, "B", 93)
    assert node.features[Nfeat.BSA] > 0.0


def test_bsa_atom():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph = build_testgraph(pdb_path, 4.5, 'atom')
    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_atom_node(graph, "B", 93, "OD1")
    assert node.features[Nfeat.BSA] > 0.0


def test_sasa_residue():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, _ = build_testgraph(pdb_path, 10, 'residue', 108)
    add_features(pdb_path, graph)

    # check for NaN
    assert not any(
        np.isnan(node.features[Nfeat.SASA]) for node in graph.nodes
    )

    # surface residues should have large area
    surface_residue_node = _find_residue_node(graph, "A", 105)
    assert surface_residue_node.features[Nfeat.SASA] > 25.0

    # buried residues should have small area
    buried_residue_node = _find_residue_node(graph, "A", 72)
    assert buried_residue_node.features[Nfeat.SASA] < 25.0


def test_sasa_atom():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, _ = build_testgraph(pdb_path, 10, 'atom', 108)
    add_features(pdb_path, graph)

    # check for NaN
    assert not any(
        np.isnan(node.features[Nfeat.SASA]) for node in graph.nodes
    )

    # surface atoms should have large area
    surface_atom_node = _find_atom_node(graph, "A", 105, "OE2")
    assert surface_atom_node.features[Nfeat.SASA] > 25.0

    # buried atoms should have small area
    buried_atom_node = _find_atom_node(graph, "A", 72, "CG")
    assert buried_atom_node.features[Nfeat.SASA] == 0.0
