import logging
import os
import subprocess
from typing import List, Union

import numpy as np
from deeprank2.domain.aminoacidlist import amino_acids
from deeprank2.molstruct.atom import Atom, AtomicElement
from deeprank2.molstruct.pair import Pair
from deeprank2.molstruct.residue import Residue
from deeprank2.molstruct.structure import Chain, PDBStructure
from pdb2sql import interface as get_interface
from scipy.spatial import distance_matrix

_log = logging.getLogger(__name__)


def add_hydrogens(input_pdb_path, output_pdb_path):
    """This requires reduce: https://github.com/rlabduke/reduce."""

    with open(output_pdb_path, "wt", encoding = "utf-8") as f:
        p = subprocess.run(["reduce", input_pdb_path], stdout=subprocess.PIPE, check=True)
        for line in p.stdout.decode().split("\n"):
            f.write(line.replace("   new", "").replace("   std", "") + "\n")


def _add_atom_to_residue(atom, residue):

    for other_atom in residue.atoms:
        if other_atom.name == atom.name:
            # Don't allow two atoms with the same name, pick the highest
            # occupancy
            if other_atom.occupancy < atom.occupancy:
                other_atom.change_altloc(atom)
                return

    # not there yet, add it
    residue.add_atom(atom)


_amino_acids_by_code = {
    amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids
}


_elements_by_name = {element.name: element for element in AtomicElement}


def _add_atom_data_to_structure(structure: PDBStructure,  # pylint: disable=too-many-arguments, too-many-locals
                                x: float, y: float, z: float,
                                atom_name: str,
                                altloc: str, occupancy: float,
                                element_name: str,
                                chain_id: str,
                                residue_number: int,
                                residue_name: str,
                                insertion_code: str):

    """
    This is a subroutine, to be used in other methods for converting pdb2sql atomic data into a
    deeprank structure object. It should be called for one atom.

    Args:
        structure (:class:`PDBStructure`): Where this atom should be added to.
        x (float): x-coordinate of atom.
        y (float): y-coordinate of atom.
        z (float): z-coordinate of atom.
        atom_name (str): Name of atom: 'CA', 'C', 'N', 'O', 'CB', etc.
        altloc (str): Pdb alternative location id for this atom (can be empty): 'A', 'B', 'C', etc.
        occupancy (float): Pdb occupancy of this atom, ranging from 0.0 to 1.0. Should be used with altloc.
        element_name (str): Pdb element symbol of this atom: 'C', 'O', 'H', 'N', 'S'.
        chain_id (str): Pdb chain identifier: 'A', 'B', 'C', etc.
        residue_number (int): Pdb residue number, a positive integer.
        residue_name (str): Pdb residue name: "ALA", "CYS", "ASP", etc.
        insertion_code (str): Pdb residue insertion code (can be empty) : '', 'A', 'B', 'C', etc.
    """

    # Make sure not to take the same atom twice.
    if altloc is not None and altloc != "" and altloc != "A":
        return

    # We use None to indicate that the residue has no insertion code.
    if insertion_code == "":
        insertion_code = None

    # The amino acid is only valid when we deal with protein residues.
    if residue_name in _amino_acids_by_code:
        amino_acid = _amino_acids_by_code[residue_name]
    else:
        amino_acid = None

    # Turn the x,y,z into a vector:
    atom_position = np.array([x, y, z])

    # Init chain.
    if not structure.has_chain(chain_id):

        chain = Chain(structure, chain_id)
        structure.add_chain(chain)
    else:
        chain = structure.get_chain(chain_id)

    # Init residue.
    if not chain.has_residue(residue_number, insertion_code):

        residue = Residue(chain, residue_number, amino_acid, insertion_code)
        chain.add_residue(residue)
    else:
        residue = chain.get_residue(residue_number, insertion_code)

    # Init atom.
    atom = Atom(
        residue, atom_name, _elements_by_name[element_name], atom_position, occupancy
    )
    _add_atom_to_residue(atom, residue)


def get_structure(pdb, id_: str):
    """Builds a structure from rows in a pdb file.

    Args:
        pdb (pdb2sql object): The pdb structure that we're investigating.
        id (str): Unique id for the pdb structure.

    Returns:
        PDBStructure: The structure object, giving access to chains, residues, atoms.
    """

    # We need these intermediary dicts to keep track of which residues and
    # chains have already been created.
    structure = PDBStructure(id_)

    # Iterate over the atom output from pdb2sql
    for row in pdb.get(
        "x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0
    ):

        (
            x,
            y,
            z,
            _,
            atom_name,
            altloc,
            occupancy,
            element_name,
            chain_id,
            residue_number,
            residue_name,
            insertion_code,
        ) = row

        _add_atom_data_to_structure(structure,
                                    x, y, z,
                                    atom_name,
                                    altloc, occupancy,
                                    element_name,
                                    chain_id,
                                    residue_number,
                                    residue_name,
                                    insertion_code)

    return structure


def get_contact_atoms( # pylint: disable=too-many-locals
    pdb_path: str,
    chain_id1: str,
    chain_id2: str,
    distance_cutoff: float
) -> List[Atom]:
    """Gets the contact atoms from pdb2sql and wraps them in python objects."""

    interface = get_interface(pdb_path)
    try:
        atom_indexes = interface.get_contact_atoms(cutoff=distance_cutoff, chain1=chain_id1, chain2=chain_id2)
        rows = interface.get("x,y,z,name,element,altLoc,occ,chainID,resSeq,resName,iCode",
                             rowID=atom_indexes[chain_id1] + atom_indexes[chain_id2])
    finally:
        interface._close()  # pylint: disable=protected-access

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    structure = PDBStructure(f"contact_atoms_{pdb_name}")

    for row in rows:
        (
            x,
            y,
            z,
            atom_name,
            element_name,
            altloc,
            occupancy,
            chain_id,
            residue_number,
            residue_name,
            insertion_code
        ) = row

        _add_atom_data_to_structure(structure,
                                    x, y, z,
                                    atom_name,
                                    altloc, occupancy,
                                    element_name,
                                    chain_id,
                                    residue_number,
                                    residue_name,
                                    insertion_code)

    return structure.get_atoms()

def get_residue_contact_pairs( # pylint: disable=too-many-locals
    pdb_path: str,
    structure: PDBStructure,
    chain_id1: str,
    chain_id2: str,
    distance_cutoff: float,
) -> List[Pair]:
    """Get the residues that contact each other at a protein-protein interface.

    Args:
        pdb_path (str): The path of the pdb file, that the structure was built from.
        structure (:class:`PDBStructure`): From which to take the residues.
        chain_id1 (str): First protein chain identifier.
        chain_id2 (str): Second protein chain identifier.
        distance_cutoff (float): Max distance between two interacting residues.

    Returns:
        List[Pair]: The pairs of contacting residues.
    """

    # Find out which residues are pairs
    interface = get_interface(pdb_path)
    try:
        contact_residues = interface.get_contact_residues(
            cutoff=distance_cutoff,
            chain1=chain_id1,
            chain2=chain_id2,
            return_contact_pairs=True,
        )
    finally:
        interface._close() # pylint: disable=protected-access

    # Map to residue objects
    residue_pairs = set([])
    for residue_key1, _ in contact_residues.items():
        residue_chain_id1, residue_number1, residue_name1 = residue_key1

        chain1 = structure.get_chain(residue_chain_id1)

        residue1 = None
        for residue in chain1.residues:
            if (
                residue.number == residue_number1
                and residue.amino_acid is not None
                and residue.amino_acid.three_letter_code == residue_name1
            ):
                residue1 = residue
                break
        else:
            raise ValueError(
                f"Not found: {pdb_path} {residue_chain_id1} {residue_number1} {residue_name1}"
            )

        for residue_chain_id2, residue_number2, residue_name2 in contact_residues[ # pylint: disable=unnecessary-dict-index-lookup
            residue_key1
        ]:

            chain2 = structure.get_chain(residue_chain_id2)

            residue2 = None
            for residue in chain2.residues:
                if (
                    residue.number == residue_number2
                    and residue.amino_acid is not None
                    and residue.amino_acid.three_letter_code == residue_name2
                ):
                    residue2 = residue
                    break
            else:
                raise ValueError(
                    f"Not found: {pdb_path} {residue_chain_id2} {residue_number2} {residue_name2}"
                )

            residue_pairs.add(Pair(residue1, residue2))

    return residue_pairs


def get_surrounding_residues(structure: Union[Chain, PDBStructure], residue: Residue, radius: float):
    """Get the residues that lie within a radius around a residue.

    Args:
        structure (Union[:class:`Chain`, :class:`PDBStructure`]): The structure to take residues from.
        residue (:class:`Residue`): The residue in the structure.
        radius (float): Max distance in Ångström between atoms of the residue and the other residues.

    Returns:
        (a set of deeprank residues): The surrounding residues.
    """

    structure_atoms = structure.get_atoms()
    residue_atoms = residue.atoms

    structure_atom_positions = [atom.position for atom in structure_atoms]
    residue_atom_positions = [atom.position for atom in residue_atoms]

    distances = distance_matrix(structure_atom_positions, residue_atom_positions, p=2)

    close_residues = set([])

    for structure_atom_index, structure_atom in enumerate(structure_atoms):

        shortest_distance = np.min(distances[structure_atom_index, :])

        if shortest_distance < radius:

            close_residues.add(structure_atom.residue)

    return close_residues
