import logging
import freesasa
import numpy as np
from typing import List
from deeprankcore.utils.graph import Node, Graph
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.atom import Atom
from deeprankcore.domain import nodefeatures as Nfeat

freesasa.setVerbosity(freesasa.nowarnings) # pylint: disable=c-extension-no-member
logging.getLogger(__name__)


def add_sasa_for_residues(structure: freesasa.Structure, # pylint: disable=c-extension-no-member
                              result: freesasa.Result, # pylint: disable=c-extension-no-member
                              nodes: List[Node]):

    for node in nodes:

        residue = node.id

        selection = ("residue, (resi %s) and (chain %s)" % (residue.number_string, residue.chain.id),) # pylint: disable=consider-using-f-string

        area = freesasa.selectArea(selection, structure, result)['residue'] # pylint: disable=c-extension-no-member
        if np.isnan(area):
            raise ValueError(f"freesasa returned {area} for {residue}")

        node.features[Nfeat.SASA] = area


def add_sasa_for_atoms(structure: freesasa.Structure, # pylint: disable=c-extension-no-member
                           result: freesasa.Result, # pylint: disable=c-extension-no-member
                           nodes: List[Node]):

    for node in nodes:

        atom = node.id

        selection = ("atom, (name %s) and (resi %s) and (chain %s)" % \
            (atom.name, atom.residue.number_string, atom.residue.chain.id),) # pylint: disable=consider-using-f-string

        area = freesasa.selectArea(selection, structure, result)['atom'] # pylint: disable=c-extension-no-member
        if np.isnan(area):
            raise ValueError(f"freesasa returned {area} for {atom}")

        node.features[Nfeat.SASA] = area


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=too-many-locals, unused-argument

    """calculates the Buried Surface Area (BSA) and the Solvent Accessible Surface Area (SASA):
    BSA: the area of the protein, that only gets exposed in monomeric state"""

    # BSA
    sasa_complete_structure = freesasa.Structure() # pylint: disable=c-extension-no-member
    sasa_chain_structures = {}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure() # pylint: disable=c-extension-no-member

            for atom in residue.atoms:
                sasa_chain_structures[chain_id].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                        atom.residue.number, atom.residue.chain.id,
                                                        atom.position[0], atom.position[1], atom.position[2])
                sasa_complete_structure.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                atom.residue.number, atom.residue.chain.id,
                                                atom.position[0], atom.position[1], atom.position[2]) 

        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure() # pylint: disable=c-extension-no-member

            sasa_chain_structures[chain_id].addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                                    atom.residue.number, atom.residue.chain.id,
                                                    atom.position[0], atom.position[1], atom.position[2])
            sasa_complete_structure.addAtom(atom.name, atom.residue.amino_acid.three_letter_code,
                                            atom.residue.number, atom.residue.chain.id,
                                            atom.position[0], atom.position[1], atom.position[2])

            area_key = "atom"
            selection = (f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})")
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

    sasa_complete_result = freesasa.calc(sasa_complete_structure) # pylint: disable=c-extension-no-member
    sasa_chain_results = {chain_id: freesasa.calc(structure) # pylint: disable=c-extension-no-member
                          for chain_id, structure in sasa_chain_structures.items()}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            area_key = "residue"
            selection = ("residue, (resi %s) and (chain %s)" % (residue.number_string, residue.chain.id),) # pylint: disable=consider-using-f-string

        elif isinstance(node.id, Atom):
            atom = node.id
            chain_id = atom.residue.chain.id
            area_key = "atom"
            selection = ("atom, (name %s) and (resi %s) and (chain %s)" % \
                 (atom.name, atom.residue.number_string, atom.residue.chain.id),) # pylint: disable=consider-using-f-string

        area_monomer = freesasa.selectArea(selection, sasa_chain_structures[chain_id], \
            sasa_chain_results[chain_id])[area_key] # pylint: disable=c-extension-no-member
        area_multimer = freesasa.selectArea(selection, sasa_complete_structure, sasa_complete_result)[area_key] # pylint: disable=c-extension-no-member

        node.features[Nfeat.BSA] = area_monomer - area_multimer

    # SASA
    structure = freesasa.Structure(pdb_path) # pylint: disable=c-extension-no-member
    result = freesasa.calc(structure) # pylint: disable=c-extension-no-member

    if isinstance(graph.nodes[0].id, Residue):
        return add_sasa_for_residues(structure, result, graph.nodes)

    if isinstance(graph.nodes[0].id, Atom):
        return add_sasa_for_atoms(structure, result, graph.nodes)

    raise TypeError(f"Unexpected node type: {type(graph.nodes[0].id)}")
