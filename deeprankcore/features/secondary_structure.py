import os
import numpy as np
from typing import Optional
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.atom import Atom
from deeprankcore.utils.graph import Graph
from deeprankcore.domain import nodestorage as Nfeat


def dssp(pdb_path: str):
    """
    Process the output of the DSSP program to extract secondary structure information.
    
    Args:
        pdb_path (str): The file path of the PDB file to be processed.
    
    Returns:
        dict: A dictionary containing secondary structure information for each chain and residue.
    """
    
    # Execute DSSP and read the output
    dssp_output = os.popen('dssp -i %s' % pdb_path).read()
    
    # Extract relevant lines from the output
    dssp_lines = dssp_output[dssp_output.index('  #  RESIDUE'):].split('\n')[1:-1]
    
    # Extract secondary structure features and replace codes for readability
    sec_structure_features = ''.join([line[16] for line in dssp_lines if line[13] != '!'])
    sec_structure_features = (sec_structure_features.replace('B', 'E')
                                                    .replace('G', 'H')
                                                    .replace('I', 'H')
                                                    .replace('S', 'C')
                                                    .replace(' ', 'C')
                                                    .replace('T', 'C'))

    # Extract residue numbers and chain identifiers
    residue_numbers = [int(line[5:10]) for line in dssp_lines if line[13] != '!']
    chain_ids = [line[11] for line in dssp_lines if line[13] != '!']
    
    # Initialize dictionary to store secondary structure information
    sec_structure_dict = {}
    for chain in set(chain_ids):
        sec_structure_dict[chain] = {}
    
    # Sanity check: Ensure equal lengths of chain_ids, residue_numbers, and sec_structure_features
    if not len(chain_ids) == len(residue_numbers) == len(sec_structure_features):
        raise ValueError(
            f'Unequal length of chain_ids {len(chain_ids)}, residue numbers {len(residue_numbers)}, \
                and sec_structure_features {len(sec_structure_features)} objects')
    
    # Create one-hot encoding for secondary structure features
    one_hot_encoded_features = np.zeros((len(sec_structure_features), 3))
    for ind in range(len(sec_structure_features)):
        one_hot_encoded_features[ind]['HEC'.index(sec_structure_features[ind])] = 1

    # Populate the dictionary with one-hot encoded secondary structure information
    for i in range(len(chain_ids)):
        sec_structure_dict[chain_ids[i]][residue_numbers[i]] = one_hot_encoded_features[i]
    
    return sec_structure_dict



def add_features( # pylint: disable=unused-argument
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):    
    
    """
    Add secondary structure features to the nodes of a graph.

    Args:
        pdb_path (str): The file path of the PDB file to be processed.
        graph (Graph): A Graph object representing the protein structure.
    """

    # Get the secondary structure features from the DSSP output
    sec_structure_features = dssp(pdb_path)

    # Iterate through the nodes in the graph
    for node in graph.nodes:

        # Get the node type
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

        # Get the chain ID and residue position from the node
        chain_id = residue.chain.id
        residue_position = residue.number

        # Add the secondary structure feature to the node
        node.features[Nfeat.SECSTRUCT] = sec_structure_features[chain_id][residue_position]
