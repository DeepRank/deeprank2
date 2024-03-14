from dataclasses import dataclass
from tempfile import TemporaryFile

from pdb2pqr.config import FORCE_FIELDS
from pdb2pqr.main import main_driver as pdb2pqr
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


def calculate_protonation_state(pdb_str: str, forcefield: str = "AMBER") -> str:
    """Calculate the protonation states using PDB2PQR.

    PDB2PQR can only use files (no strings) as input and output, which is why this function is wrapped inside
    TemporaryFile context managers.

    Args:
        pdb_str: string representation of pdb file.
        forcefield: Which forcefield to use. Defaults to "AMBER".

    Returns:
        str: updated pdb
    """
    with TemporaryFile(mode="w", suffix="pdb", encoding="utf-8") as input_pdb, TemporaryFile(mode="r", encoding="utf-8") as output_pdb:
        input_pdb.write(pdb_str)

        input_args = _Pdb2pqrArgs(input_pdb, output_pdb, forcefield)
        pdb2pqr(input_args)

        return output_pdb.read()


@dataclass
class _Pdb2pqrArgs:
    """Input arguments to `main_driver` function of PDB2PQR.

    These are usually given via CLI using argparse. All arguments, including those kept as default need to be given to
    `main_driver` if called from script.
    The argument given to `main_driver` is accessed via dot notation and is iterated over, which is why this is created
    as a dataclass with an iterator.

    Args:
        input_path: path of the input file
        output_pqr: path of the output file
        ff: which forcefield to use
        all other arguments should remain untouched.

    Raises:
        ValueError: if the forcefield is not recognized
    """

    input_path: str
    output_pqr: str
    ff: str = "AMBER"

    # arguments set different from default
    debump: bool = True
    keep_chain: bool = True
    log_level: str = "CRITICAL"

    # arguments kept as default
    ph: float = 7.0
    assign_only: bool = False
    clean: bool = False
    userff: None = None
    ffout: None = None
    usernames: None = None
    ligand: None = None
    neutraln: bool = False
    neutralc: bool = False
    drop_water: bool = False
    pka_method: None = None
    opt: bool = True
    include_header: bool = False
    whitespace: bool = False
    pdb_output: None = None
    apbs_input: None = None

    def __post_init__(self):
        self._index = 0
        if self.ff.lower() not in FORCE_FIELDS:
            msg = f"Forcefield {self.ff} not recognized. Valid options: {FORCE_FIELDS}."
            raise ValueError(msg)

    def __iter__(self):
        return self

    def __next__(self):
        settings = vars(self)
        if self._index < len(settings):
            setting = list(settings)[self._index]
            self._index += 1
            return setting
        raise StopIteration
