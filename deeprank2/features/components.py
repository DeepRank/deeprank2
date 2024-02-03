import logging

import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph
from deeprank2.utils.parsing import atomic_forcefield

_log = logging.getLogger(__name__)


def add_features(  # noqa:D103
    pdb_path: str,  # noqa: ARG001
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,
) -> None:
    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue

            node.features[Nfeat.ATOMTYPE] = atom.element.onehot
            node.features[Nfeat.PDBOCCUPANCY] = atom.occupancy
            node.features[Nfeat.ATOMCHARGE] = atomic_forcefield.get_charge(atom)
        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

        node.features[Nfeat.RESTYPE] = residue.amino_acid.onehot
        node.features[Nfeat.RESCHARGE] = residue.amino_acid.charge
        node.features[Nfeat.POLARITY] = residue.amino_acid.polarity.onehot
        node.features[Nfeat.RESSIZE] = residue.amino_acid.size
        node.features[Nfeat.RESMASS] = residue.amino_acid.mass
        node.features[Nfeat.RESPI] = residue.amino_acid.pI
        node.features[Nfeat.HBDONORS] = residue.amino_acid.hydrogen_bond_donors
        node.features[Nfeat.HBACCEPTORS] = residue.amino_acid.hydrogen_bond_acceptors

        if single_amino_acid_variant is not None:
            wildtype = single_amino_acid_variant.wildtype_amino_acid
            variant = single_amino_acid_variant.variant_amino_acid

            if residue == single_amino_acid_variant.residue:
                node.features[Nfeat.VARIANTRES] = variant.onehot
                node.features[Nfeat.DIFFCHARGE] = variant.charge - wildtype.charge
                node.features[Nfeat.DIFFPOLARITY] = variant.polarity.onehot - wildtype.polarity.onehot
                node.features[Nfeat.DIFFSIZE] = variant.size - wildtype.size
                node.features[Nfeat.DIFFMASS] = variant.mass - wildtype.mass
                node.features[Nfeat.DIFFPI] = variant.pI - wildtype.pI
                node.features[Nfeat.DIFFHBDONORS] = variant.hydrogen_bond_donors - wildtype.hydrogen_bond_donors
                node.features[Nfeat.DIFFHBACCEPTORS] = variant.hydrogen_bond_acceptors - wildtype.hydrogen_bond_acceptors
            else:
                node.features[Nfeat.VARIANTRES] = residue.amino_acid.onehot
                node.features[Nfeat.DIFFCHARGE] = 0
                node.features[Nfeat.DIFFPOLARITY] = np.zeros(residue.amino_acid.polarity.onehot.shape)
                node.features[Nfeat.DIFFSIZE] = 0
                node.features[Nfeat.DIFFMASS] = 0
                node.features[Nfeat.DIFFPI] = 0
                node.features[Nfeat.DIFFHBDONORS] = 0
                node.features[Nfeat.DIFFHBACCEPTORS] = 0
