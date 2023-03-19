from pathlib import Path
from typing import Optional
from pdb2sql import pdb2sql
from deeprankcore.utils.graph import Graph, build_atomic_graph, build_residue_graph
from deeprankcore.utils.buildgraph import get_structure, get_residue_contact_pairs, get_surrounding_residues
from deeprankcore.utils.parsing.pssm import parse_pssm
from deeprankcore.molstruct.aminoacid import AminoAcid
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.structure import PDBStructure, Chain
from deeprankcore.molstruct.variant import SingleResidueVariant


def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue
    raise ValueError(f"Not found: {number}")


def build_testgraph( # pylint: disable=too-many-locals
    pdb_path: str, 
    cutoff: float, 
    detail: str, 
    central_res: Optional[int] = None,
    variant: AminoAcid = None,
    ) -> Graph:


    pdb = pdb2sql(pdb_path)
    try:
        structure: PDBStructure = get_structure(pdb, Path(pdb_path).stem)
    finally:
        pdb._close() # pylint: disable=protected-access

    if not central_res: # pylint: disable=no-else-raise
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

        if detail == 'residue':
            return build_residue_graph(list(nodes), structure.id, cutoff)
        if detail == 'atom':
            return build_atomic_graph(list(nodes), structure.id, cutoff)
        raise TypeError('detail must be "atom" or "residue"')

    else:
        chain: Chain = structure.chains[0]
        residue = _get_residue(chain, central_res)
        surrounding_residues = list(get_surrounding_residues(structure, residue, cutoff))

        with open(f"tests/data/pssm/{structure.id}/{structure.id}.{chain.id}.pdb.pssm", "rt", encoding="utf-8") as f:
            chain.pssm = parse_pssm(f, chain)

        if detail == 'residue':
            return build_residue_graph(surrounding_residues, structure.id, cutoff), SingleResidueVariant(residue, variant)
        if detail == 'atom':
            atoms = set(atom for residue in surrounding_residues for atom in residue.atoms)
            return build_atomic_graph(list(atoms), structure.id, cutoff), SingleResidueVariant(residue, variant)
        raise TypeError('detail must be "atom" or "residue"')
