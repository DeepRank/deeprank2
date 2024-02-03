import logging

import freesasa
import numpy as np

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph

freesasa.setVerbosity(freesasa.nowarnings)
logging.getLogger(__name__)


def add_sasa(pdb_path: str, graph: Graph) -> None:  # noqa:D103
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            selection = (f"residue, (resi {residue.number_string}) and (chain {residue.chain.id})",)
            area = freesasa.selectArea(selection, structure, result)["residue"]

        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
            selection = (f"atom, (name {atom.name}) and (resi {residue.number_string}) and (chain {residue.chain.id})",)
            area = freesasa.selectArea(selection, structure, result)["atom"]

        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

        if np.isnan(area):
            msg = f"freesasa returned {area} for {residue}"
            raise ValueError(msg)
        node.features[Nfeat.SASA] = area


def add_bsa(graph: Graph) -> None:  # noqa:D103
    sasa_complete_structure = freesasa.Structure()
    sasa_chain_structures = {}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure()

            for atom in residue.atoms:
                sasa_chain_structures[chain_id].addAtom(
                    atom.name,
                    atom.residue.amino_acid.three_letter_code,
                    atom.residue.number,
                    atom.residue.chain.id,
                    atom.position[0],
                    atom.position[1],
                    atom.position[2],
                )
                sasa_complete_structure.addAtom(
                    atom.name,
                    atom.residue.amino_acid.three_letter_code,
                    atom.residue.number,
                    atom.residue.chain.id,
                    atom.position[0],
                    atom.position[1],
                    atom.position[2],
                )

        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
            chain_id = residue.chain.id
            if chain_id not in sasa_chain_structures:
                sasa_chain_structures[chain_id] = freesasa.Structure()

            sasa_chain_structures[chain_id].addAtom(
                atom.name,
                atom.residue.amino_acid.three_letter_code,
                atom.residue.number,
                atom.residue.chain.id,
                atom.position[0],
                atom.position[1],
                atom.position[2],
            )
            sasa_complete_structure.addAtom(
                atom.name,
                atom.residue.amino_acid.three_letter_code,
                atom.residue.number,
                atom.residue.chain.id,
                atom.position[0],
                atom.position[1],
                atom.position[2],
            )

            area_key = "atom"
            selection = f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})"
        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

    sasa_complete_result = freesasa.calc(sasa_complete_structure)
    sasa_chain_results = {chain_id: freesasa.calc(structure) for chain_id, structure in sasa_chain_structures.items()}

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
            chain_id = residue.chain.id
            area_key = "residue"
            selection = (f"residue, (resi {residue.number_string}) and (chain {residue.chain.id})",)

        elif isinstance(node.id, Atom):
            atom = node.id
            chain_id = atom.residue.chain.id
            area_key = "atom"
            selection = (f"atom, (name {atom.name}) and (resi {atom.residue.number_string}) and (chain {atom.residue.chain.id})",)

        area_monomer = freesasa.selectArea(selection, sasa_chain_structures[chain_id], sasa_chain_results[chain_id])[area_key]
        area_multimer = freesasa.selectArea(selection, sasa_complete_structure, sasa_complete_result)[area_key]

        node.features[Nfeat.BSA] = area_monomer - area_multimer


def add_features(
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,  # noqa: ARG001
) -> None:
    """Calculates the Buried Surface Area (BSA) and the Solvent Accessible Surface Area (SASA)."""
    # BSA
    add_bsa(graph)

    # SASA
    add_sasa(pdb_path, graph)
