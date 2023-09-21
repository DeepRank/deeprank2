from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph


class DSSPError(Exception):
    "Raised if DSSP fails to produce an output"


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


def _get_records(lines: List[str]):
    seen = set()
    seen_add = seen.add
    return [x.split()[0] for x in lines if not (x in seen or seen_add(x))]


def _check_pdb(pdb_path: str):
    fix_pdb = False
    with open(pdb_path, encoding='utf-8') as f:
        lines = f.readlines()

    # check for HEADER
    firstline = lines[0]
    if not firstline.startswith('HEADER'):
        fix_pdb = True
        if firstline.startswith('EXPDTA'):
            lines = [f'HEADER {firstline}'] + lines[1:]
        else:
            lines = ['HEADER \n'] + lines

    # check for CRYST1 record
    existing_records = _get_records(lines)
    if 'CRYST1' not in existing_records:
        fix_pdb = True
        dummy_CRYST1 = 'CRYST1   00.000   00.000   00.000  00.00  00.00  00.00 X 00 00 0    00\n'
        lines = [lines[0]] + [dummy_CRYST1] + lines[1:]

    # check for unnumbered REMARK lines
    for i, line in enumerate(lines):
        if line.startswith('REMARK'):
            try:
                int(line.split()[1])
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

    # pylint: disable=raise-missing-from
    try:
        dssp = DSSP(model, pdb_path, dssp='mkdssp')
    except Exception as e: # improperly formatted pdb files raise: `Exception: DSSP failed to produce an output`
        pdb_format_link = 'https://www.wwpdb.org/documentation/file-format-content/format33/sect1.html#Order'
        raise DSSPError(f'DSSP has raised the following exception: {e}.\
            \nThis is likely due to an improrperly formatted pdb file: {pdb_path}.\
            \nSee {pdb_format_link} for guidance on how to format your pdb files.\
            \nAlternatively, turn off secondary_structure feature module during QueryCollection.process().')

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
