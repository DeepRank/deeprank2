import numpy

from deeprank_gnn.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprank_gnn.domain.amino_acid import amino_acids


def get_structure(pdb2sql, name):
    """ Builds a structure from rows in a pdb file
        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating
        Returns (Structure): the structure object, giving access to chains, residues, atoms
    """

    amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids}
    elements_by_name = {element.name: element for element in AtomicElement}

    # We need these intermediary dicts to keep track of which residues and chains have already been created.
    chains = {}
    residues = {}

    structure = Structure(name)

    # Iterate over the atom output from pdb2sql
    for row in pdb2sql.get("x,y,z,rowID,name,altLoc,element,chainID,resSeq,resName,iCode", model=0):

        x, y, z, atom_number, atom_name, altloc, element, chain_id, residue_number, residue_name, insertion_code = row

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
        atom = Atom(residue, atom_name, elements_by_name[element], atom_position)
        residue.add_atom(atom)

    return structure
