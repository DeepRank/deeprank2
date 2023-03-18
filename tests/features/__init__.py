import os
from pdb2sql import pdb2sql
from deeprankcore.utils.graph import Graph, build_atomic_graph, build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_residue_contact_pairs
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.structure import PDBStructure


def build_testgraph(pdb_path: str, detail: str, cutoff: float) -> Graph:
    """_summary_

    Args:
        pdb_path (str): _description_
        detail (str): _description_
        cutoff (float): _description_

    Raises:
        TypeError: _description_

    Returns:
        Graph: _description_
    """

    pdb = pdb2sql(pdb_path)
    structure = get_structure(pdb, os.path.splitext(pdb_path)[0])

    nodes = set([])
    for residue1, residue2 in get_residue_contact_pairs(
        pdb_path, structure, 
        structure.chains[0].id, structure.chains[1].id, 
        cutoff
    ):
        if detail == 'residue':
            nodes.add(residue1)
            nodes.add(residue2)

        elif detail == 'atom':
            for atom in residue1.atoms:
                nodes.add(atom)
            for atom in residue2.atoms:
                nodes.add(atom)

    nodes = list(nodes)


    if detail == 'residue':
        return build_residue_graph(nodes, structure.id, cutoff)

    elif detail == 'atom':
        return build_atomic_graph(nodes, structure.id, cutoff)

    else:
        raise TypeError('detail must be "atom" or "residue"')
