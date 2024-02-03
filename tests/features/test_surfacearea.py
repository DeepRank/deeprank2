import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.features.surfacearea import add_features
from deeprank2.utils.graph import Graph, Node

from . import build_testgraph


def _find_residue_node(
    graph: Graph,
    chain_id: str,
    residue_number: int,
) -> Node:
    for node in graph.nodes:
        residue = node.id
        if residue.chain.id == chain_id and residue.number == residue_number:
            return node
    msg = f"Not found: {chain_id} {residue_number}"
    raise ValueError(msg)


def _find_atom_node(
    graph: Graph,
    chain_id: str,
    residue_number: int,
    atom_name: str,
) -> Node:
    for node in graph.nodes:
        atom = node.id
        if atom.residue.chain.id == chain_id and atom.residue.number == residue_number and atom.name == atom_name:
            return node
    msg = f"Not found: {chain_id} {residue_number} {atom_name}"
    raise ValueError(msg)


def test_bsa_residue() -> None:
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail="residue",
        influence_radius=8.5,
        max_edge_length=8.5,
    )
    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_residue_node(graph, "B", 93)
    assert node.features[Nfeat.BSA] > 0.0


def test_bsa_atom() -> None:
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail="atom",
        influence_radius=4.5,
        max_edge_length=4.5,
    )
    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_atom_node(graph, "B", 93, "OD1")
    assert node.features[Nfeat.BSA] > 0.0


def test_sasa_residue() -> None:
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail="residue",
        influence_radius=10,
        max_edge_length=10,
        central_res=108,
    )
    add_features(pdb_path, graph)

    # check for NaN
    assert not any(np.isnan(node.features[Nfeat.SASA]) for node in graph.nodes)

    # surface residues should have large area
    surface_residue_node = _find_residue_node(graph, "A", 105)
    assert surface_residue_node.features[Nfeat.SASA] > 25.0

    # buried residues should have small area
    buried_residue_node = _find_residue_node(graph, "A", 72)
    assert buried_residue_node.features[Nfeat.SASA] < 25.0


def test_sasa_atom() -> None:
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail="atom",
        influence_radius=10,
        max_edge_length=10,
        central_res=108,
    )
    add_features(pdb_path, graph)

    # check for NaN
    assert not any(np.isnan(node.features[Nfeat.SASA]) for node in graph.nodes)

    # surface atoms should have large area
    surface_atom_node = _find_atom_node(graph, "A", 105, "OE2")
    assert surface_atom_node.features[Nfeat.SASA] > 25.0

    # buried atoms should have small area
    buried_atom_node = _find_atom_node(graph, "A", 72, "CG")
    assert buried_atom_node.features[Nfeat.SASA] == 0.0
