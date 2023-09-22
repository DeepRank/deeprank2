import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain.aminoacidlist import glycine, serine
from deeprank2.features.components import add_features

from . import build_testgraph


def test_atom_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, _ = build_testgraph(
        pdb_path=pdb_path,
        detail='atom',
        interaction_radius=10,
        max_edge_distance=10,
        central_res=25,
    )
    add_features(pdb_path, graph)
    assert not any(np.isnan(node.features[Nfeat.ATOMCHARGE]) for node in graph.nodes)
    assert not any(np.isnan(node.features[Nfeat.PDBOCCUPANCY]) for node in graph.nodes)


def test_aminoacid_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"
    graph, variant = build_testgraph(
        pdb_path=pdb_path,
        detail='residue',
        interaction_radius=10,
        max_edge_distance=10,
        central_res=25,
        variant=serine,
    )
    add_features(pdb_path, graph, variant)
    node = graph.nodes[25].id

    for node in graph.nodes:
        if node.id == variant.residue:  # GLY -> SER
            assert sum(node.features[Nfeat.RESTYPE]) == 1
            assert node.features[Nfeat.RESTYPE][glycine.index] == 1
            assert node.features[Nfeat.RESCHARGE] == glycine.charge
            assert (node.features[Nfeat.POLARITY] == glycine.polarity.onehot).all
            assert node.features[Nfeat.RESSIZE] == glycine.size
            assert node.features[Nfeat.RESMASS] == glycine.mass
            assert node.features[Nfeat.RESPI] == glycine.pI
            assert node.features[Nfeat.HBDONORS] == glycine.hydrogen_bond_donors
            assert node.features[Nfeat.HBACCEPTORS] == glycine.hydrogen_bond_acceptors

            assert sum(node.features[Nfeat.VARIANTRES]) == 1
            assert node.features[Nfeat.VARIANTRES][serine.index] == 1
            assert node.features[Nfeat.DIFFCHARGE] == serine.charge - glycine.charge
            assert (node.features[Nfeat.DIFFPOLARITY] == serine.polarity.onehot - glycine.polarity.onehot).all
            assert node.features[Nfeat.DIFFSIZE] == serine.size - glycine.size
            assert node.features[Nfeat.DIFFMASS] == serine.mass - glycine.mass
            assert node.features[Nfeat.DIFFPI] == serine.pI - glycine.pI
            assert node.features[Nfeat.DIFFHBDONORS] == serine.hydrogen_bond_donors - glycine.hydrogen_bond_donors
            assert node.features[Nfeat.DIFFHBACCEPTORS] == serine.hydrogen_bond_acceptors - glycine.hydrogen_bond_acceptors
