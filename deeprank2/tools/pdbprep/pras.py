from pathlib import Path
from tempfile import TemporaryFile

from Pras_Server.RunType import InitRunType as PRAS


def add_missing_heavy_atoms(pdb_str: str) -> str:
    """Add missing heavy atoms (usually many) using PRAS.

    PRAS can only use files (no strings) as input and output, which is why this function is wrapped inside
    TemporaryFile context managers.

    Args:
        pdb_str: string representation of pdb file.

    Returns:
        str: updated pdb
    """
    with TemporaryFile(mode="w", suffix="pdb", encoding="utf-8") as input_pdb, TemporaryFile(mode="r", encoding="utf-8") as output_pdb:
        input_pdb.write(pdb_str)

        fixing = PRAS(ofname=output_pdb)
        fixing.fname = input_pdb
        fixing.ProcessOther()  # write to specified filename

        return output_pdb.read()
