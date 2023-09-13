import logging
from typing import Optional

import freesasa
import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph

# pylint: disable=c-extension-no-member

freesasa.setVerbosity(freesasa.nowarnings)
logging.getLogger(__name__)


def add_sasa(pdb_path: str, graph: Graph):
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            selection = (f"residue, (resi {residue.number_string}) and (chain {residue.chain.id})",)
            area = freesasa.selectArea(selection, structure, result)['residue']

        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
            selection = (f"atom, (name {atom.name}) and (resi {residue.number_string}) and (chain {residue.chain.id})",)
            area = freesasa.selectArea(selection, structure, result)['atom']

        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

        if np.isnan(area):
            raise ValueError(f"freesasa returned {area} for {residue}")
        node.features[Nfeat.SASA] = area



def add_bsa(graph: Graph):

    sasa_complete_structure = freesasa.Structure()
    sasa_chain_structures = {}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure()

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
                sasa_chain_structures[chain_id] = freesasa.Structure()

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

    sasa_complete_result = freesasa.calc(sasa_complete_structure)
    sasa_chain_results = {chain_id: freesasa.calc(structure)
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
            sasa_chain_results[chain_id])[area_key]
        area_multimer = freesasa.selectArea(selection, sasa_complete_structure, sasa_complete_result)[area_key]

        node.features[Nfeat.BSA] = area_monomer - area_multimer


def add_features( # pylint: disable=unused-argument
    pdb_path: str, graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):

    """calculates the Buried Surface Area (BSA) and the Solvent Accessible Surface Area (SASA):
    BSA: the area of the protein, that only gets exposed in monomeric state"""

    # BSA
    add_bsa(graph)

    # SASA
    add_sasa(pdb_path, graph)
