import os
import numpy as np
from typing import Optional, Dict
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.atom import Atom
from deeprankcore.utils.graph import Graph
from deeprankcore.domain import nodestorage as Nfeat


def _get_secstruct(pdb_path: str) -> Dict:
    """Process the DSSP output to extract secondary structure information.
    
    Args:
        pdb_path (str): The file path of the PDB file to be processed.
    
    Returns:
        dict: A dictionary containing secondary structure information for each chain and residue.
    """
    
    # Execute DSSP and read the output
    # outputs residue number @ pos 5-10, chain_id @ pos 11, secondary structure @ pos 16
    dssp_output = os.popen(f'dssp -i {pdb_path}').read()
    dssp_lines = dssp_output[dssp_output.index('  #  RESIDUE'):].split('\n')[1:-1]
    residue_numbers = [int(line[5:10]) for line in dssp_lines if line[13] != '!']
    chain_ids = [line[11] for line in dssp_lines if line[13] != '!']

    #regroup secondary structures into 3 main classes
    sec_structure_features = ''.join([line[16] for line in dssp_lines if line[13] != '!'])
    sec_structure_features = (sec_structure_features.replace('B', 'E')
                                                    .replace('G', 'H')
                                                    .replace('I', 'H')
                                                    .replace('S', 'C')
                                                    .replace(' ', 'C')
                                                    .replace('T', 'C'))
    
    # Sanity check: Ensure equal lengths of chain_ids, residue_numbers, and sec_structure_features
    if not len(chain_ids) == len(residue_numbers) == len(sec_structure_features):
        raise ValueError(
            f'Unequal length of chain_ids {len(chain_ids)}, residue numbers {len(residue_numbers)}, \
                and sec_structure_features {len(sec_structure_features)} objects.\n \
                    Check DSSP output for {pdb_path}')

    # Initialize dictionary to store secondary structure information
    sec_structure_dict = {}
    for chain in set(chain_ids):
        sec_structure_dict[chain] = {}
    
    # Create one-hot encoding for secondary structure features
    one_hot_encoded_features = np.zeros((len(sec_structure_features), 3))
    for ind, _ in enumerate(sec_structure_features):
        one_hot_encoded_features[ind]['HEC'.index(sec_structure_features[ind])] = 1

    # Populate the dictionary with one-hot encoded secondary structure information
    for i, _ in enumerate(chain_ids):
        sec_structure_dict[chain_ids[i]][residue_numbers[i]] = one_hot_encoded_features[i]
    
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
        node.features[Nfeat.SECSTRUCT] = sec_structure_features[chain_id][res_num]
