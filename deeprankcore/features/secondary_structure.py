import numpy as np
from typing import Optional, Dict
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.atom import Atom
from deeprankcore.utils.graph import Graph
from deeprankcore.domain import nodestorage as Nfeat
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from enum import Enum


class SecondarySctructure(Enum):
    "a value to express a secondary a residue's secondary structure type"

    HELIX = 0 # GHI
    STRAND = 1 # BE
    COIL = 2 # -ST

    @property
    def onehot(self):
        t = np.zeros(3)
        t[self.value] = 1.0

        return t


def _get_secstruct(pdb_path: str) -> Dict:
    """Process the DSSP output to extract secondary structure information.
    
    Args:
        pdb_path (str): The file path of the PDB file to be processed.
    
    Returns:
        dict: A dictionary containing secondary structure information for each chain and residue.
    """

    # Execute DSSP and read the output
    p = PDBParser(QUIET=True)
    model = p.get_structure("", pdb_path)[0]
    dssp = DSSP(model, pdb_path)

    chain_ids = [dssp_key[0] for dssp_key in dssp.property_keys]
    res_numbers = [dssp_key[1][1] for dssp_key in dssp.property_keys]
    sec_structs = [dssp[dssp_key][2] for dssp_key in dssp.property_keys]

    # Store output in Dictionary
    sec_structure_dict = {}
    for chain in set(chain_ids):
        sec_structure_dict[chain] = {}    
    for i, _ in enumerate(chain_ids):
        sec_structure_dict[chain_ids[i]][res_numbers[i]] = sec_structs[i]
    
    return sec_structure_dict


def add_features( # pylint: disable=unused-argument
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):    

    sec_structure_features = _get_secstruct(pdb_path)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

        chain_id = residue.chain.id
        res_num = residue.number

        if sec_structure_features[chain_id][res_num] in ('-', 'S', 'T'):
            node.features[Nfeat.SECSTRUCT] = SecondarySctructure.COIL.onehot
        elif sec_structure_features[chain_id][res_num] in ('B', 'E'):
            node.features[Nfeat.SECSTRUCT] = SecondarySctructure.STRAND.onehot
        elif sec_structure_features[chain_id][res_num] in ('G', 'H', 'I'):
            node.features[Nfeat.SECSTRUCT] = SecondarySctructure.HELIX.onehot
        else:
            raise ValueError(f'Unknown secondary structure type ({sec_structure_features[chain_id][res_num]}) \
                             detected on chain{chain_id} residues{res_num}.')
