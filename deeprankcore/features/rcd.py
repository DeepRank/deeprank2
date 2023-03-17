import logging
import pdb2sql
from typing import Dict, List
from deeprankcore.utils.graph import Graph
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.aminoacid import Polarity
from deeprankcore.molstruct.atom import Atom
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.domain.aminoacidlist import amino_acids


_log = logging.getLogger(__name__)


def id_from_residue(residue: tuple) -> str:
    """Create and id from pdb2sql rendered residues that is similar to the id of residue nodes

    Args:
        residue (tuple): Input residue as rendered by pdb2sql: ( str(<chain>), int(<residue_number>), str(<three_letter_code> )
            For example: ('A', 27, 'GLU').
    
    Returns:
        str: Output id in form of '<chain><residue_number>'. For example: 'A27'.
    """
    
    return residue[0] + str(residue[1])


class _ContactDensity:
    """Internal class that holds contact density information for a given residue.
    """
    
    def __init__(self, residue):
        self.res = residue
        self.id = id_from_residue(self.res)
        self.densities = {pol: 0 for pol in Polarity}
        self.densities['total'] = 0
        self.connections = {pol: [] for pol in Polarity}
        self.connections['all'] = []


def count_residue_contacts(pdb_path: str, chains: List[str], cutoff: float = 5.5) -> Dict[str, _ContactDensity]:
    """Count total number of close contact residues and contacts of specific Polarity.

    Args:
        pdb_path (str): Path to pdb file to read molecular information from.
        chains (Sequence[str]): List (or list-like object) containing strings of the chains to be considered.
        cutoff (float, optional): Cutoff distance (in Ångström) to be considered a close contact. Defaults to 10.

    Returns:
        Dict[str, _ContactDensity]: 
            keys: ids of residues in form returned by id_from_residue.
            items: _ContactDensity objects, containing all contact density information for the residue.
    """

    sql = pdb2sql.interface(pdb_path)
    pdb2sql_contacts = sql.get_contact_residues(
        cutoff=cutoff, 
        chain1=chains[0], chain2=chains[1],
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
        contact1_id = id_from_residue(chain1_res)
        residue_contacts[contact1_id] = _ContactDensity(chain1_res)
        
        for chain2_res in chain2_residues:
            aa2_code = chain2_res[2]
            try:
                aa2 = [amino_acid for amino_acid in amino_acids if amino_acid.three_letter_code == aa2_code][0]
            except IndexError:
                continue  # skip keys that are not an amino acid
            
            # populate densities and connections for chain1_res
            residue_contacts[contact1_id].densities['total'] += 1
            residue_contacts[contact1_id].densities[aa2.polarity] += 1
            residue_contacts[contact1_id].connections['all'].append(chain2_res)
            residue_contacts[contact1_id].connections[aa2.polarity].append(chain2_res)
            
            # add chain2_res to residue_contact dict if it doesn't exist yet
            contact2_id = id_from_residue(chain2_res)
            if contact2_id not in residue_contacts:
                residue_contacts[contact2_id] = _ContactDensity(chain2_res)
            # populate densities and connections for chain2_res
            residue_contacts[contact2_id].densities['total'] += 1
            residue_contacts[contact2_id].densities[aa1.polarity] += 1
            residue_contacts[contact2_id].connections['all'].append(chain1_res)
            residue_contacts[contact2_id].connections[aa1.polarity].append(chain1_res)
    
    return residue_contacts


def add_features(
    pdb_path: str, 
    graph: Graph,
    single_amino_acid_variant = None):
    
    if not single_amino_acid_variant: # VariantQueries do not use this feature
        residue_contacts = count_residue_contacts(pdb_path, graph.get_all_chains())
        
        noncontact_residues = 0
        for node in graph.nodes:
            if isinstance(node.id, Residue):
                residue = node.id
            elif isinstance(node.id, Atom):
                atom = node.id
                residue = atom.residue
            else:
                raise TypeError(f"Unexpected node type: {type(node.id)}")

            chain_name = str(residue).split()[1]  # returns the name of the chain
            res_num = str(residue).split()[2]  # returns the residue number
            contact_id = chain_name + res_num  # reformat id to be in line with residue_contacts keys
            
            try:
                node.features[Nfeat.RCDTOTAL] = residue_contacts[contact_id].densities['total']
                node.features[Nfeat.RCDNONPOLAR] = residue_contacts[contact_id].densities[Polarity.NONPOLAR]
                node.features[Nfeat.RCDPOLAR] = residue_contacts[contact_id].densities[Polarity.POLAR]
                node.features[Nfeat.RCDNEGATIVE] = residue_contacts[contact_id].densities[Polarity.NEGATIVE_CHARGE]
                node.features[Nfeat.RCDPOSITIVE] = residue_contacts[contact_id].densities[Polarity.POSITIVE_CHARGE]
            except KeyError:
                node.features[Nfeat.RCDTOTAL] = 0
                node.features[Nfeat.RCDNONPOLAR] = 0
                node.features[Nfeat.RCDPOLAR] = 0
                node.features[Nfeat.RCDNEGATIVE] = 0
                node.features[Nfeat.RCDPOSITIVE] = 0

                noncontact_residues += 1
        
        if noncontact_residues == len(graph.nodes):
            _log.warning(f"No residue contacts detected for {pdb_path}.")
