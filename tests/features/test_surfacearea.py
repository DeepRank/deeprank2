from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.domain.aminoacidlist import alanine
from deeprankcore.molstruct.structure import PDBStructure, Chain
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.features.surfacearea import add_features
from deeprankcore.operations.graph import build_residue_graph, build_atomic_graph
from deeprankcore.operations.buildgraph import (
    get_structure,
    get_residue_contact_pairs,
    get_surrounding_residues)
from deeprankcore.domain import nodefeatures



def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue

    raise ValueError(f"Not found: {number}")

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


def _load_pdb_structure(pdb_path: str, id_: str) -> PDBStructure:
    pdb = pdb2sql(pdb_path)
    try:
        return get_structure(pdb, id_)
    finally:
        pdb._close() # pylint: disable=protected-access


def test_bsa_residue():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    structure = _load_pdb_structure(pdb_path, "1ATN_1w")

    residues = set([])
    for residue1, residue2 in get_residue_contact_pairs(
        pdb_path, structure, "A", "B", 8.5
    ):
        residues.add(residue1)
        residues.add(residue2)

    graph = build_residue_graph(residues, "1ATN-1w", 8.5)

    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_residue_node(graph, "B", 93)

    assert node.features[nodefeatures.BSA] > 0.0


def test_bsa_atom():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    structure = _load_pdb_structure(pdb_path, "1ATN_1w")

    atoms = set([])
    for residue1, residue2 in get_residue_contact_pairs(
        pdb_path, structure, "A", "B", 8.5
    ):
        for atom in residue1.atoms:
            atoms.add(atom)
        for atom in residue2.atoms:
            atoms.add(atom)
    atoms = list(atoms)

    graph = build_atomic_graph(atoms, "1ATN-1w", 8.5)

    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_atom_node(graph, "B", 93, "OD1")

    assert node.features[nodefeatures.BSA] > 0.0

def test_sasa_residue():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = _get_residue(structure.chains[0], 108)
    variant = SingleResidueVariant(residue, alanine)

    residues = get_surrounding_residues(structure, residue, 10.0)
    assert len(residues) > 0

    graph = build_residue_graph(residues, "101M-108-res", 4.5)
    add_features(pdb_path, graph, variant)

    # check for NaN
    assert not any(
        np.isnan(node.features[nodefeatures.SASA]) for node in graph.nodes
    )

    # surface residues should have large area
    surface_residue_node = _find_residue_node(graph, "A", 105)
    assert surface_residue_node.features[nodefeatures.SASA] > 25.0

    # buried residues should have small area
    buried_residue_node = _find_residue_node(graph, "A", 72)
    assert (
        buried_residue_node.features[nodefeatures.SASA] < 25.0
    ), buried_residue_node.features[nodefeatures.SASA]


def test_sasa_atom():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = _get_residue(structure.chains[0], 108)
    variant = SingleResidueVariant(residue, alanine)

    residues = get_surrounding_residues(structure, residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-108-atom", 4.5)
    add_features(pdb_path, graph, variant)

    # check for NaN
    assert not any(
        np.isnan(node.features[nodefeatures.SASA]) for node in graph.nodes
    )

    # surface atoms should have large area
    surface_atom_node = _find_atom_node(graph, "A", 105, "OE2")
    assert surface_atom_node.features[nodefeatures.SASA] > 25.0

    # buried atoms should have small area
    buried_atom_node = _find_atom_node(graph, "A", 72, "CG")
    assert (
        buried_atom_node.features[nodefeatures.SASA] == 0.0
    ), buried_atom_node.features[nodefeatures.SASA]
