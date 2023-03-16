from typing import Optional
import numpy
import os	
from deeprankcore.models.graph import Graph
from deeprankcore.models.variant import SingleResidueVariant
from deeprankcore.models.structure import Atom, Residue
from deeprankcore.domain.feature import (FEATURENAME_AMINOACID, FEATURENAME_VARIANTAMINOACID,
										 FEATURENAME_SIZE, FEATURENAME_POLARITY,
										 FEATURENAME_SIZEDIFFERENCE, FEATURENAME_POLARITYDIFFERENCE,
										 FEATURENAME_HYDROGENBONDDONORS, FEATURENAME_HYDROGENBONDDONORSDIFFERENCE,
										 FEATURENAME_HYDROGENBONDACCEPTORS, FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE)
									 

def dssp(
	pdb_path: str,):
	"""
    Calculates DSSP features for a given PDB structure.
    
    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        str: A string representing the secondary structure features.
    """
	output = os.popen('dssp -i %s' % pdb_path).read()
	output = output[output.index('  #  RESIDUE'):].split('\n')[1:-1]
	secs   = ''.join([line[16]for line in output if line[13]!='!'])
	secs = secs.replace('B','E').replace('G', 'H').replace('I', 'H').replace('I', 'H').replace('S', 'C').replace(' ', 'C').replace('T', 'C')
	return secs



def sec_struc_id(sec_structure_features: str):
	    """
    Defines the secondary structure elements features.
    
    Args:
        sec_strucutre_features (str): A string representing the secondary structure features.

    Returns:
        list: A list of secondary structure element IDs.
    """

	element_id = 0
    sec_struc_ids = [0]
    for i in range(1, len(sec_structure_features)):
        if sec_structure_features[i] != sec_structure_features[i-1]:
            element_id += 1
        sec_struc_ids.append(element_id)
    return sec_struc_ids



def add_features_for_residues(( 
	pdb_path: str, graph: Graph,):
	"""
    Adds node features to the graph based on the protein structure.

    Args:
        pdb_path (str): The path to the PDB file for the protein structure.
        graph (Graph): The graph to which node features will be added to.
    """
	
	secs_features = dssp(pdb_path)
	secStrucids = sec_struc_id(secs_features)
    
	one_hot = numpy.zeros((len(secs_features), 3))
	for ind in range(len(secs_features)):
		one_hot[ind]['HEC'.index(secs_features[ind])] = 1
	
	for ind, node in enumerate(graph.nodes):
		if isinstance(node.id, Residue):
			residue = node.id

		elif isinstance(node.id, Atom):
			atom = node.id
			residue = atom.residue
		else:
			raise TypeError(f"Unexpected node type: {type(node.id)}") 
			
		node.features[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
		node.features['SecStruc'] = one_hot[ind]
		node.features['SecStrucId'] = secStrucids[ind]
		node.features['NodeNmb'] = ind
		
					
def add_features(pdb_path: str, graph: Graph, *args, **kwargs):
	"""
    Adds features for both residues and atoms in the graph.

    Args:
        pdb_path (str): The path to the PDB file for the protein structure.
        graph (Graph): The graph to which features will be added.
    """
	add_features_for_residues(pdb_path, graph)





