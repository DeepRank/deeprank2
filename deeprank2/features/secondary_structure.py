from enum import Enum
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from numpy.typing import NDArray

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph


class DSSPError(Exception):
    """Raised if DSSP fails to produce an output."""


class SecondarySctructure(Enum):
    """Value to express a secondary a residue's secondary structure type."""

    HELIX = 0  # 'GHI'
    STRAND = 1  # 'BE'
    COIL = 2  # ' -STP'

    @property
    def onehot(self) -> NDArray:
        t = np.zeros(3)
        t[self.value] = 1.0

        return t


def _get_records(lines: list[str]) -> list[str]:
    seen = set()
    seen_add = seen.add
    return [x.split()[0] for x in lines if not (x in seen or seen_add(x))]


def _check_pdb(pdb_path: str) -> None:
    """Check whether pdb metadata required for DSSP exists and auto-fix in place where possible.

    Args:
        pdb_path: file location of pdb file
    """
    fix_pdb = False
    with open(pdb_path, encoding="utf-8") as f:
        lines = f.readlines()

    # check for HEADER
    firstline = lines[0]
    if not firstline.startswith("HEADER"):
        fix_pdb = True
        if firstline.startswith("EXPDTA"):
            lines = [f"HEADER {firstline}"] + lines[1:]
        else:
            lines = ["HEADER \n", *lines]

    # check for CRYST1 record
    existing_records = _get_records(lines)
    if "CRYST1" not in existing_records:
        fix_pdb = True
        dummy_CRYST1 = "CRYST1   00.000   00.000   00.000  00.00  00.00  00.00 X 00 00 0    00\n"
        lines = [lines[0]] + [dummy_CRYST1] + lines[1:]

    # check for unnumbered REMARK lines
    for i, line in enumerate(lines):
        if line.startswith("REMARK"):
            try:
                int(line.split()[1])
            except ValueError:
                fix_pdb = True
                lines[i] = f"REMARK 999 {line[7:]}"

    if fix_pdb:
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def _classify_secstructure(subtype: str) -> SecondarySctructure:
    if subtype in "GHI":
        return SecondarySctructure.HELIX
    if subtype in "BE":
        return SecondarySctructure.STRAND
    if subtype in " -STP":
        return SecondarySctructure.COIL
    return None


def _get_secstructure(pdb_path: str) -> dict:
    """Process the DSSP output to extract secondary structure information.

    Args:
        pdb_path: The file path of the PDB file to be processed.

    Returns:
        dict: A dictionary containing secondary structure information for each chain and residue.
    """
    # Execute DSSP and read the output
    _check_pdb(pdb_path)
    p = PDBParser(QUIET=True)
    model = p.get_structure(Path(pdb_path).stem, pdb_path)[0]

    try:
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
    except Exception as e:
        pdb_format_link = "https://www.wwpdb.org/documentation/file-format-content/format33/sect1.html#Order"
        msg = (
            f"DSSP has raised the following exception: {e}.\n\t"
            f"This is likely due to an improrperly formatted pdb file: {pdb_path}.\n\t"
            f"See {pdb_format_link} for guidance on how to format your pdb files.\n\t"
            "Alternatively, turn off secondary_structure feature module during QueryCollection.process()."
        )
        raise DSSPError(msg) from e

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


def add_features(  # noqa:D103
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,  # noqa: ARG001
) -> None:
    sec_structure_features = _get_secstructure(pdb_path)

    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

        chain_id = residue.chain.id
        res_num = residue.number

        try:
            node.features[Nfeat.SECSTRUCT] = _classify_secstructure(sec_structure_features[chain_id][res_num]).onehot
        except AttributeError as e:
            msg = f"Unknown secondary structure type ({sec_structure_features[chain_id][res_num]}) detected on chain {chain_id} residues {res_num}."
            raise ValueError(msg) from e
