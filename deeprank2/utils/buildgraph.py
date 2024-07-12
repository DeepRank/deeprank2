import logging
import os

import numpy as np
from pdb2sql import interface as pdb2sql_interface
from pdb2sql import pdb2sql as pdb2sql_object
from scipy.spatial import distance_matrix

from deeprank2.domain.aminoacidlist import amino_acids_by_code
from deeprank2.molstruct.atom import Atom, AtomicElement
from deeprank2.molstruct.pair import Pair
from deeprank2.molstruct.residue import Residue
from deeprank2.molstruct.structure import Chain, PDBStructure

_log = logging.getLogger(__name__)


def _add_atom_to_residue(atom: Atom, residue: Residue) -> None:
    """Adds an `Atom` to a `Residue` if not already there.

    If no matching atom is found, add the current atom to the residue.
    If there's another atom with the same name, choose the one with the highest occupancy.
    """
    for other_atom in residue.atoms:
        if other_atom.name == atom.name and other_atom.occupancy < atom.occupancy:
            other_atom.change_altloc(atom)
            return
    residue.add_atom(atom)


def _add_atom_data_to_structure(
    structure: PDBStructure,
    pdb_obj: pdb2sql_object,
    **kwargs,  # noqa: ANN003
) -> None:
    """This subroutine retrieves pdb2sql atomic data for `PDBStructure` objects as defined in DeepRank2.

    This function should be called for one atom at a time.

    Args:
        structure: The structure to which this atom should be added to.
        pdb_obj: The `pdb2sql` object to retrieve the data from.
        kwargs: as required by the get function for the `pdb2sql` object.
    """
    pdb2sql_columns = "x,y,z,name,altLoc,occ,element,chainID,resSeq,resName,iCode"
    data_keys = pdb2sql_columns.split(sep=",")
    for data_values in pdb_obj.get(pdb2sql_columns, **kwargs):
        atom_data = dict(zip(data_keys, data_values, strict=True))

        # exit function if this atom is already part of the structure
        if atom_data["altLoc"] not in (None, "", "A"):
            return

        atom_data["iCode"] = None if atom_data["iCode"] == "" else atom_data["iCode"]

        try:
            atom_data["aa"] = amino_acids_by_code[atom_data["resName"]]
        except KeyError:
            atom_data["aa"] = None
        atom_data["coordinates"] = np.array(data_values[:3])

        if not structure.has_chain(atom_data["chainID"]):
            structure.add_chain(Chain(structure, atom_data["chainID"]))
        chain = structure.get_chain(atom_data["chainID"])

        if not chain.has_residue(atom_data["resSeq"], atom_data["iCode"]):
            chain.add_residue(Residue(chain, atom_data["resSeq"], atom_data["aa"], atom_data["iCode"]))
        residue = chain.get_residue(atom_data["resSeq"], atom_data["iCode"])

        atom = Atom(
            residue,
            atom_data["name"],
            AtomicElement[atom_data["element"]],
            atom_data["coordinates"],
            atom_data["occ"],
        )
        _add_atom_to_residue(atom, residue)


def get_structure(pdb_obj: pdb2sql_object, id_: str) -> PDBStructure:
    """Builds a structure from rows in a pdb file.

    Args:
        pdb_obj: The pdb structure that we're investigating.
        id_: Unique id for the pdb structure.

    Returns:
        PDBStructure: The structure object, giving access to chains, residues, atoms.
    """
    structure = PDBStructure(id_)
    _add_atom_data_to_structure(structure, pdb_obj, model=0)
    return structure


def get_contact_atoms(
    pdb_path: str,
    chain_ids: list[str],
    influence_radius: float,
) -> list[Atom]:
    """Gets the contact atoms from pdb2sql and wraps them in python objects."""
    interface = pdb2sql_interface(pdb_path)
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    structure = PDBStructure(f"contact_atoms_{pdb_name}")

    try:
        atom_indexes = interface.get_contact_atoms(
            cutoff=influence_radius,
            chain1=chain_ids[0],
            chain2=chain_ids[1],
        )
        pdb_rowID = atom_indexes[chain_ids[0]] + atom_indexes[chain_ids[1]]
        _add_atom_data_to_structure(structure, interface, rowID=pdb_rowID)
    finally:
        interface._close()  # noqa: SLF001

    return structure.get_atoms()


def get_residue_contact_pairs(
    pdb_path: str,
    structure: PDBStructure,
    chain_id1: str,
    chain_id2: str,
    influence_radius: float,
) -> list[Pair]:
    """Find all residue pairs that may influence each other.

    Args:
        pdb_path: The path of the pdb file, that the structure was built from.
        structure: From which to take the residues.
        chain_id1: First protein chain identifier.
        chain_id2: Second protein chain identifier.
        influence_radius: Maximum distance between residues to consider them as interacting.

    Returns:
        list of Pair objects of contacting residues.
    """
    # Find out which residues are pairs
    interface = pdb2sql_interface(pdb_path)
    try:
        contact_residues = interface.get_contact_residues(
            cutoff=influence_radius,
            chain1=chain_id1,
            chain2=chain_id2,
            return_contact_pairs=True,
        )
    finally:
        interface._close()  # noqa: SLF001

    # Map to residue objects
    residue_pairs = set()
    for residue_key1, residue_contacts in contact_residues.items():
        residue1 = _get_residue_from_key(structure, residue_key1)
        for residue_key2 in residue_contacts:
            residue2 = _get_residue_from_key(structure, residue_key2)
            residue_pairs.add(Pair(residue1, residue2))

    return residue_pairs


def _get_residue_from_key(
    structure: PDBStructure,
    residue_key: tuple[str, int, str],
) -> Residue:
    """Returns a residue object given a pdb2sql-formatted residue key."""
    residue_chain_id, residue_number, residue_name = residue_key
    chain = structure.get_chain(residue_chain_id)

    for residue in chain.residues:
        if residue.number == residue_number and residue.amino_acid is not None and residue.amino_acid.three_letter_code == residue_name:
            return residue
    msg = f"Residue ({residue_key}) not found in {structure.id}."
    raise ValueError(msg)


def get_surrounding_residues(
    structure: Chain | PDBStructure,
    residue: Residue,
    radius: float,
) -> list[Residue]:
    """Get the residues that lie within a radius around a residue.

    Args:
        structure: The structure to take residues from.
        residue: The residue in the structure.
        radius: Max distance in Ångström between atoms of the residue and the other residues.

    Returns:
        list of surrounding Residue objects.
    """
    structure_atoms = structure.get_atoms()
    structure_atom_positions = [atom.position for atom in structure_atoms]
    residue_atom_positions = [atom.position for atom in residue.atoms]
    pairwise_distances = distance_matrix(
        structure_atom_positions,
        residue_atom_positions,
        p=2,
    )

    surrounding_residues = set()
    for structure_atom_index, structure_atom in enumerate(structure_atoms):
        shortest_distance = np.min(pairwise_distances[structure_atom_index, :])
        if shortest_distance < radius:
            surrounding_residues.add(structure_atom.residue)

    return list(surrounding_residues)
