from pathlib import Path
from tempfile import TemporaryFile

from pdbtools import pdb_delresname, pdb_fixinsert, pdb_keepcoord, pdb_reatom, pdb_reres, pdb_rplresname, pdb_selaltloc, pdb_sort, pdb_tidy
from Pras_Server.RunType import InitRunType as PRAS

from deeprank2.domain.aminoacidlist import amino_acids_by_code, amino_acids_by_letter


def preprocess_pdbs(
    pdb_path: str | Path,
    rename_residues: dict[str, str] | None = None,
) -> str:
    """Preprocess a pdb file for adding/fixing hydrogens.

    Args:
        pdb_path: Path of pdb file to preprocess.
        rename_residues: Dictionary mapping of non-standard residue names (keys) to their standard names.
            Defaults to:
            {
                "MSE": "MET",
                "HIP": "HIS",
                "HIE": "HIS",
                "HID": "HIS",
                "HSE": "HIS",
                "HSD": "HIS",
            }
    """
    with Path(pdb_path).open("r") as f:
        pdb_str = f.read()

    pdb_str = _run_pdb_tools(pdb_str, rename_residues)
    pdb_str = _add_missing_heavy_atoms(pdb_str)

    return pdb_str  # noqa: RET504


def _run_pdb_tools(
    pdb_str: str,
    rename_residues: dict[str, str] | None = None,
) -> str:
    """Preprocesses pdb files using pdb-tools (Bonvin lab).

    Files undergo a number of pruning steps:
        1. Scrape non-atomic records.
        2. Scrape water molecules.
        3. Replace non-standard residue names with their standard counterparts.
            A default library is used for this, which can be replaced using the `rename_residues` argument.
        4. Scrape lower occupancy atoms in case of alternate locations.
            Note that in case of equal occupancy, the first record is always used.
        5. Delete insertion codes and shift the residue numbering of downstream residues.
        6. Sort records by chain and residues.
        7. Renumber residues on each chain from 1.
        8. Renumber atoms from 1.
        9. Tidy up to somewhat adhere to pdb format specifications.

    Raises:
        ValueError: if an invalid amino acid (3-letter or 1-letter) code is given as a value to rename_residues.

    Returns:
        str: Updated pdb
    """
    if not rename_residues:
        rename_residues = {
            "MSE": "MET",
            "HIP": "HIS",
            "HIE": "HIS",
            "HID": "HIS",
            "HSE": "HIS",
            "HSD": "HIS",
        }
    else:
        for new_res in rename_residues.values():
            if new_res not in amino_acids_by_code and new_res not in amino_acids_by_letter:
                msg = f"{new_res} is not a valid amino-acid code."
                raise ValueError(msg)

    # sequentially run individual tools from pdb-tools
    new_pdb = pdb_keepcoord.run(pdb_str)  # Scrape non-atomic records
    new_pdb = pdb_delresname.run(new_pdb, ("HOH",))  # Scrape water molecules

    for old, new in rename_residues.items():
        new_pdb = pdb_rplresname.run(new_pdb, old, new)  # Replace non-standard residue names with their standard counterparts

    new_pdb = pdb_selaltloc.run(new_pdb)  # Scrape lower occupancy atoms in case of alternate locations
    new_pdb = pdb_fixinsert.run(new_pdb, [])  # Delete insertion codes and shift the residue numbering of downstream residues.
    new_pdb = pdb_sort.run(new_pdb, "CR")  # Sort records by chain and residues
    new_pdb = pdb_reres.run(new_pdb, 1)  # Renumber residues on each chain from 1
    new_pdb = pdb_reatom.run(new_pdb, 1)  # Renumber atoms from 1
    new_pdb = pdb_tidy.run(new_pdb)  # Tidy up to somewhat adhere to pdb format specifications

    return "".join(list(new_pdb))


def _add_missing_heavy_atoms(pdb_str: str) -> str:
    """Add missing heavy atoms (usually many) using PRAS.

    PRAS can only use files (no strings) as input and output, which is why this function is wrapped inside
    TemporaryFile context managers.

    Returns:
        str: Updated pdb
    """
    with TemporaryFile(mode="w", suffix="pdb", encoding="utf-8") as input_pdb, TemporaryFile(mode="r", encoding="utf-8") as output_pdb:
        input_pdb.write(pdb_str)

        fixing = PRAS(ofname=output_pdb)
        fixing.fname = input_pdb
        fixing.ProcessOther()  # write to specified filename

        return output_pdb.read()
