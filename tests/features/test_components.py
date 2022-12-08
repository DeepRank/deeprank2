import numpy as np
from pdb2sql import pdb2sql
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.domain.aminoacidlist import serine, glycine
from deeprankcore.utils.graph import build_atomic_graph, build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_surrounding_residues
from deeprankcore.molstruct.structure import Chain
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.features.components import add_features


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
    surr_residues = get_surrounding_residues(structure, residue, 10.0)
    atoms = set([])
    for residue in surr_residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-108-atom", 4.5)
    add_features(pdb_path, graph)

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
    surr_residues = get_surrounding_residues(structure, residue, 10.0)
    variant = SingleResidueVariant(residue, serine)  # GLY -> SER
    assert len(surr_residues) > 0

    graph = build_residue_graph(surr_residues, "101m-25", 4.5)
    add_features(pdb_path, graph, variant)
    for node in graph.nodes:
        if node.id == variant.residue:  # GLY -> SER
            assert sum(node.features[Nfeat.RESTYPE]) == 1
            assert node.features[Nfeat.RESTYPE][glycine.index] == 1
            assert node.features[Nfeat.PROPERTYX] == glycine.propertyX
            assert (node.features[Nfeat.POLARITY] == glycine.polarity.onehot).all
            assert node.features[Nfeat.RESSIZE] == glycine.size
            assert node.features[Nfeat.RESMASS] == glycine.mass
            assert node.features[Nfeat.RESPI] == glycine.pI
            assert node.features[Nfeat.HBDONORS] == glycine.hydrogen_bond_donors
            assert node.features[Nfeat.HBACCEPTORS] == glycine.hydrogen_bond_acceptors

            assert sum(node.features[Nfeat.VARIANTRES]) == 1
            assert node.features[Nfeat.VARIANTRES][serine.index] == 1
            assert node.features[Nfeat.DIFFX] == serine.propertyX - glycine.propertyX
            assert (node.features[Nfeat.DIFFPOLARITY] == serine.polarity.onehot - glycine.polarity.onehot).all
            assert node.features[Nfeat.DIFFSIZE] == serine.size - glycine.size
            assert node.features[Nfeat.DIFFMASS] == serine.mass - glycine.mass
            assert node.features[Nfeat.DIFFPI] == serine.pI - glycine.pI
            assert node.features[Nfeat.DIFFHBDONORS] == serine.hydrogen_bond_donors - glycine.hydrogen_bond_donors
            assert node.features[Nfeat.DIFFHBACCEPTORS] == serine.hydrogen_bond_acceptors - glycine.hydrogen_bond_acceptors
