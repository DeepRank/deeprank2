from typing import Optional

import numpy
from deeprankcore.models.graph import Graph
from deeprankcore.models.variant import SingleResidueVariant
from deeprankcore.models.structure import Atom, Residue
from deeprankcore.domain.features import nodefeats as Nfeat

def add_features( # pylint: disable=unused-argument
    pdb_path: str, graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}") 

        node.features[Nfeat.RESTYPE] = residue.amino_acid.onehot
        node.features[Nfeat.RESCHARGE] = residue.amino_acid.charge
        node.features[Nfeat.RESSIZE] = residue.amino_acid.size
        node.features[Nfeat.POLARITY] = residue.amino_acid.polarity.onehot
        node.features[Nfeat.HBDONORS] = residue.amino_acid.count_hydrogen_bond_donors
        node.features[Nfeat.HBACCEPTORS] = residue.amino_acid.count_hydrogen_bond_acceptors

        if single_amino_acid_variant is not None:

            wildtype = single_amino_acid_variant.wildtype_amino_acid
            variant = single_amino_acid_variant.variant_amino_acid

            if residue == single_amino_acid_variant.residue:
                node.features[Nfeat.VARIANTRES] = variant.onehot
                node.features[Nfeat.DIFFCHARGE] = variant.charge - wildtype.charge
                node.features[Nfeat.DIFFSIZE] = variant.size - wildtype.size
                node.features[Nfeat.DIFFPOLARITY] = variant.polarity.onehot - wildtype.polarity.onehot
                node.features[Nfeat.DIFFHBDONORS] = variant.count_hydrogen_bond_donors - wildtype.count_hydrogen_bond_donors
                node.features[Nfeat.DIFFHBACCEPTORS] = variant.count_hydrogen_bond_acceptors - wildtype.count_hydrogen_bond_acceptors
            else:
                node.features[Nfeat.VARIANTRES] = residue.amino_acid.onehot
                node.features[Nfeat.DIFFCHARGE] = 0
                node.features[Nfeat.DIFFSIZE] = 0
                node.features[Nfeat.DIFFPOLARITY] = numpy.zeros(residue.amino_acid.polarity.onehot.shape)
                node.features[Nfeat.DIFFHBDONORS] = 0
                node.features[Nfeat.DIFFHBACCEPTORS] = 0