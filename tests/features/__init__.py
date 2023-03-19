import os
from typing import Optional
from pdb2sql import pdb2sql
from deeprankcore.utils.graph import Graph, build_atomic_graph, build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_residue_contact_pairs, get_surrounding_residues
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.structure import PDBStructure, Chain


def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue
    raise ValueError(f"Not found: {number}")



def build_testgraph(pdb_path: str, cutoff: float, detail: str, central_res: Optional[int] = None) -> Graph:


    pdb = pdb2sql(pdb_path)
    try:
        structure: PDBStructure = get_structure(pdb, os.path.splitext(pdb_path)[0])
    finally:
        pdb._close() # pylint: disable=protected-access

    if not central_res:
        nodes = set([])
        for residue1, residue2 in get_residue_contact_pairs(
            pdb_path, structure, 
            structure.chains[0].id, structure.chains[1].id, 
            cutoff
        ):
            if detail == 'residue':
                nodes.add(residue1)
                nodes.add(residue2)
                return build_residue_graph(list(nodes), structure.id, cutoff)
            
            elif detail == 'atom':
                for atom in residue1.atoms:
                    nodes.add(atom)
                for atom in residue2.atoms:
                    nodes.add(atom)
                return build_atomic_graph(list(nodes), structure.id, cutoff)

            else:
                raise TypeError('detail must be "atom" or "residue"')

    else:
        if detail == 'residue':
            residue = _get_residue(structure.chains[0], 108)
            nodes = get_surrounding_residues(structure, residue, cutoff)
            return build_residue_graph(list(nodes), structure.id, cutoff)

