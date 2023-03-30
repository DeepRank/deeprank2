from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.residue import Residue
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.utils.graph import Graph


class SecondarySctructure(Enum):
    "a value to express a secondary a residue's secondary structure type"

    HELIX = 0 # 'GHI'
    STRAND = 1 # 'BE'
    COIL = 2 # ' -STP'

    @property
    def onehot(self):
        t = np.zeros(3)
        t[self.value] = 1.0

        return t


def _check_pdb(pdb_path):
    fix_pdb = False
    with open(pdb_path, encoding='utf-8') as f:
        lines = f.readlines()
        firstline = lines[0]

    # check for HEADER
    if not firstline.startswith('HEADER'):
        fix_pdb = True
        if firstline.startswith('EXPDTA'):
            lines = [f'HEADER {firstline}'] + lines[1:]
        else:
            lines = ['HEADER \n'] + lines

    # check for unnumbered REMARK lines
    for i, line in enumerate(lines):
        if line.startswith('REMARK '):
            try:
                int(line.split(' ')[1])
            except ValueError:
                fix_pdb = True
                lines[i] = f'REMARK 999 {line[7:]}'

    if fix_pdb:
        with open(pdb_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


def _classify_secstructure(subtype: str):
    if subtype in 'GHI':
        return SecondarySctructure.HELIX
    if subtype in 'BE':
        return SecondarySctructure.STRAND
    if subtype in ' -STP':
        return SecondarySctructure.COIL
    return None


def _get_secstructure(pdb_path: str) -> Dict:
    """Process the DSSP output to extract secondary structure information.
    
    Args:
        pdb_path (str): The file path of the PDB file to be processed.
    
    Returns:
        dict: A dictionary containing secondary structure information for each chain and residue.
    """
 
    # Execute DSSP and read the output
    _check_pdb(pdb_path)
    p = PDBParser(QUIET=True)
    model = p.get_structure(Path(pdb_path).stem, pdb_path)[0]
    dssp = DSSP(model, pdb_path, dssp='mkdssp')

    chain_ids = [dssp_key[0] for dssp_key in dssp.property_keys]
    res_numbers = [dssp_key[1][1] for dssp_key in dssp.property_keys]
    sec_structs = [dssp[dssp_key][2] for dssp_key in dssp.property_keys]

    # Store output in Dictionary
    sec_structure_dict = {}
    for chain in set(chain_ids):
        sec_structure_dict[chain] = {}    
    for i, _ in enumerate(chain_ids):
        sec_structure_dict[chain_ids[i]][res_numbers[i]] = sec_structs[i]
    
    return sec_structure_dict


def add_features( # pylint: disable=unused-argument
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):    

    sec_structure_features = _get_secstructure(pdb_path)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError(f"Unexpected node type: {type(node.id)}")

        chain_id = residue.chain.id
        res_num = residue.number

        # pylint: disable=raise-missing-from
        try:
            node.features[Nfeat.SECSTRUCT] = _classify_secstructure(sec_structure_features[chain_id][res_num]).onehot
        except AttributeError:
            raise ValueError(f'Unknown secondary structure type ({sec_structure_features[chain_id][res_num]}) ' +
                             f'detected on chain {chain_id} residues {res_num}.')
