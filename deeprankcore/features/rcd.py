import pdb2sql
from deeprankcore.utils.graph import Node, Graph
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.aminoacid import Polarity
from deeprankcore.molstruct.atom import Atom
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.domain.aminoacidlist import amino_acids


def count_residue_contacts(pdb_path: str, cutoff: float = 5.5, chain1: str = 'A', chain2: str = 'B'):
    """Count total number of close contact residues and contacts of specific Polarity.

    Args:
        pdb_path (str): Path to pdb file to read molecular information from
        cutoff (float, optional): Cutoff distance (in Angstrom) to be considered a close contact. Defaults to 5.5
        chain1 (str, optional): Name of first chain from pdb file to consider. Defaults to 'A'.
        chain2 (str, optional): Name of second chain from pdb file to consider. Defaults to 'B'.

    Returns:
        dict: keys are each residue; items are _ContactDensity objects, which containing all contact density information 
    """

    sql = pdb2sql.interface(pdb_path)    
    pdb2sql_contacts = sql.get_contact_residues(
        cutoff=cutoff, 
        chain1=chain1, chain2=chain2,
        return_contact_pairs=True
    )
    
    residue_contacts = {}

    for chain1_res, chain2_residues in pdb2sql_contacts.items():
        aa1_code = chain1_res[2]
        try:
            aa1 = [amino_acid for amino_acid in amino_acids if amino_acid.three_letter_code == aa1_code][0]        
        except IndexError:
            continue  # skip keys that are not an amino acid

        # add chain1_res to residue_contact dict
        residue_contacts[chain1_res] = _ContactDensity(chain1_res)
        
        for chain2_res in chain2_residues:
            aa2_code = chain2_res[2]
            try:
                aa2 = [amino_acid for amino_acid in amino_acids if amino_acid.three_letter_code == aa2_code][0]        
            except IndexError:
                continue  # skip keys that are not an amino acid
            
            # populate densities and connections for chain1_res
            residue_contacts[chain1_res].densities['total'] += 1
            residue_contacts[chain1_res].densities[aa2.polarity] += 1
            residue_contacts[chain1_res].connections[aa2.polarity].append(chain2_res)
            
            # add chain2_res to residue_contact dict if it doesn't exist yet
            if chain2_res not in residue_contacts:
                residue_contacts[chain2_res] = _ContactDensity(chain2_res)
            # populate densities and connections for chain2_res
            residue_contacts[chain2_res].densities['total'] += 1
            residue_contacts[chain2_res].densities[aa1.polarity] += 1
            residue_contacts[chain2_res].connections[aa1.polarity].append(chain1_res)
    
    return residue_contacts


def add_features(
    pdb_path: str, 
    graph: Graph,
    *args, **kwargs): # pylint: disable=unused-argument
    
    residue_contacts = count_residue_contacts(pdb_path)
    
    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")
        # example residue will hash as: <1A0Z B 118>
        # example _ContactDensity.id will is a tuple: <('A', 27, 'GLU')>
    
    


class _ContactDensity:
    """Internal class that holds contact density information for a given residue.
    """
    def __init__(self, residue):
        self.id = residue
        self.res = residue
        self.densities = {pol: 0 for pol in Polarity}
        self.densities['total': 0]
        self.connections = {pol: [] for pol in Polarity}
    
    