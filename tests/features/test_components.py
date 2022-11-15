import numpy as np
from pdb2sql import pdb2sql
from deeprankcore.domain import nodefeatures as Nfeat
from deeprankcore.domain.aminoacidlist import serine
from deeprankcore.utils.graph import build_atomic_graph, build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_surrounding_residues
from deeprankcore.molstruct.structure import Chain
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.features import components, surfacearea


def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue

    raise ValueError(f"Not found: {number}")


def test_atom_features(): # copied first part, up to add_features, from test_sasa

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = _get_residue(structure.chains[0], 108)
    residues = get_surrounding_residues(structure, residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-108-atom", 4.5)
    components.add_features(pdb_path, graph)

    assert not any(np.isnan(node.features[Nfeat.ATOMCHARGE]) for node in graph.nodes)
    assert not any(np.isnan(node.features[Nfeat.PDBOCCUPANCY]) for node in graph.nodes)


def test_aminoacid_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close() # pylint: disable=protected-access

    residue = structure.chains[0].residues[25]
    variant = SingleResidueVariant(residue, serine)  # GLY -> SER

    residues = get_surrounding_residues(structure, residue, 10.0)
    assert len(residues) > 0

    graph = build_residue_graph(residues, "101m-25", 4.5)

    surfacearea.add_features(pdb_path, graph, variant)

    for node in graph.nodes:
        if node.id == variant.residue:  # GLY -> SER
            assert node.features[Nfeat.DIFFSIZE] > 0
            assert node.features[Nfeat.DIFFHBDONORS] > 0
