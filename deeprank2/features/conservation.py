import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain.aminoacidlist import amino_acids
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph


def add_features(  # noqa:D103
    pdb_path: str,  # noqa: ARG001
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,
) -> None:
    profile_amino_acid_order = sorted(amino_acids, key=lambda aa: aa.three_letter_code)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

        pssm_row = residue.get_pssm()
        profile = np.array([pssm_row.get_conservation(amino_acid) for amino_acid in profile_amino_acid_order])
        node.features[Nfeat.PSSM] = profile
        node.features[Nfeat.INFOCONTENT] = pssm_row.information_content

        if single_amino_acid_variant is not None:
            if residue == single_amino_acid_variant.residue:
                # only the variant residue can have a variant and wildtype amino acid
                conservation_wildtype = pssm_row.get_conservation(single_amino_acid_variant.wildtype_amino_acid)
                conservation_variant = pssm_row.get_conservation(single_amino_acid_variant.variant_amino_acid)
                node.features[Nfeat.CONSERVATION] = conservation_wildtype
                node.features[Nfeat.DIFFCONSERVATION] = conservation_variant - conservation_wildtype
            else:
                # all nodes must have the same features, so set them to zero here
                node.features[Nfeat.CONSERVATION] = 0.0
                node.features[Nfeat.DIFFCONSERVATION] = 0.0
