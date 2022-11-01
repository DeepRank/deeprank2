import logging
from typing import List
import subprocess

from scipy.spatial import distance_matrix
import numpy
from pdb2sql import interface as get_interface

from deeprankcore.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprankcore.models.amino_acid import amino_acids
from deeprankcore.models.pair import Pair

_log = logging.getLogger(__name__)


def add_hydrogens(input_pdb_path, output_pdb_path):
    "this requires reduce: https://github.com/rlabduke/reduce"

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


def get_structure(pdb, id_): # pylint: disable=too-many-locals
    """Builds a structure from rows in a pdb file
    Args:
        pdb (pdb2sql object): the pdb structure that we're investigating
        id (str): unique id for the pdb structure
    Returns (Structure): the structure object, giving access to chains, residues, atoms
    """

    amino_acids_by_code = {
        amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids
    }
    elements_by_name = {element.name: element for element in AtomicElement}

    # We need these intermediary dicts to keep track of which residues and
    # chains have already been created.
    chains = {}
    residues = {}

    structure = Structure(id_)

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
            element,
            chain_id,
            residue_number,
            residue_name,
            insertion_code,
        ) = row

        # Make sure not to take the same atom twice.
        if altloc is not None and altloc != "" and altloc != "A":
            continue

        # We use None to indicate that the residue has no insertion code.
        if insertion_code == "":
            insertion_code = None

        # The amino acid is only valid when we deal with protein residues.
        if residue_name in amino_acids_by_code:
            amino_acid = amino_acids_by_code[residue_name]
        else:
            amino_acid = None

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        # Init chain.
        if chain_id not in chains:

            chain = Chain(structure, chain_id)
            chains[chain_id] = chain
            structure.add_chain(chain)
        else:
            chain = chains[chain_id]

        # Init residue.
        residue_id = (chain_id, residue_number, insertion_code)
        if residue_id not in residues:

            residue = Residue(chain, residue_number, amino_acid, insertion_code)
            residues[residue_id] = residue
            chain.add_residue(residue)
        else:
            residue = residues[residue_id]

        # Init atom.
        atom = Atom(
            residue, atom_name, elements_by_name[element], atom_position, occupancy
        )
        _add_atom_to_residue(atom, residue)

    return structure


def get_residue_contact_pairs( # pylint: disable=too-many-locals
    pdb_path: str,
    structure: Structure,
    chain_id1: str,
    chain_id2: str,
    distance_cutoff: float,
) -> List[Pair]:
    """Get the residues that contact each other at a protein-protein interface.

    Args:
        pdb_path: the path of the pdb file, that the structure was built from
        structure: from which to take the residues
        chain_id1: first protein chain identifier
        chain_id2: second protein chain identifier
        distance_cutoff: max distance between two interacting residues

    Returns: the pairs of contacting residues
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


def get_surrounding_residues(structure, residue, radius):
    """Get the residues that lie within a radius around a residue.

    Args:
        structure(deeprank structure object): the structure to take residues from
        residue(deeprank residue object): the residue in the structure
        radius(float): max distance in Ångström between atoms of the residue and the other residues

    Returns: (a set of deeprank residues): the surrounding residues
    """

    structure_atoms = structure.get_atoms()
    residue_atoms = residue.atoms

    structure_atom_positions = [atom.position for atom in structure_atoms]
    residue_atom_positions = [atom.position for atom in residue_atoms]

    distances = distance_matrix(structure_atom_positions, residue_atom_positions, p=2)

    close_residues = set([])

    for structure_atom_index, structure_atom in enumerate(structure_atoms):

        shortest_distance = numpy.min(distances[structure_atom_index, :])

        if shortest_distance < radius:

            close_residues.add(structure_atom.residue)

    return close_residues

