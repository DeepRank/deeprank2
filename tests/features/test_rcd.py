import numpy as np
from pdb2sql import pdb2sql
from deeprankcore.features.rcd import add_features
from deeprankcore.utils.graph import build_residue_graph, build_atomic_graph
from deeprankcore.utils.buildgraph import get_structure, get_residue_contact_pairs
from deeprankcore.domain import nodestorage as Nfeat


def _load_pdb_structure(pdb_path: str, id_: str):
    pdb = pdb2sql(pdb_path)
    try:
        return get_structure(pdb, id_)
    finally:
        pdb._close() # pylint: disable=protected-access


def test_residue_features():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    structure = _load_pdb_structure(pdb_path, "1ATN_1w")

    residues = set([])
    for residue1, residue2 in get_residue_contact_pairs(
        pdb_path, structure, "A", "B", 8.5
    ):
        residues.add(residue1)
        residues.add(residue2)
    residues = list(residues)

    graph = build_residue_graph(residues, "1ATN-1w", 8.5)

    add_features(pdb_path, graph)


    assert np.any(
        node.features[Nfeat.RCDTOTAL] > 0 for node in graph.nodes
    )
    

def test_atom_features():

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

    atom_graph = build_atomic_graph(atoms, "1A0Z", 4.5)

    add_features(pdb_path, atom_graph)
