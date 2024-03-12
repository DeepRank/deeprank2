# All code in this subpackage has been adapted from https://github.com/DeepRank/pdbprep,
# which is published under an Apache 2.0 licence

import sys
from collections.abc import Generator
from typing import TextIO


def write_pdb(new_pdb: list, pdbfh: TextIO) -> None:
    """Writes new pdb files."""
    try:
        _buffer = []
        _buffer_size = 5000  # write N lines at a time
        for lineno, line in enumerate(new_pdb):
            if not (lineno % _buffer_size):
                sys.stdout.write("".join(_buffer))
                _buffer = []
            _buffer.append(line)

        sys.stdout.write("".join(_buffer))
        sys.stdout.flush()
    except OSError:
        # This is here to catch Broken Pipes
        # for example to use 'head' or 'tail' without
        # the error message showing up
        pass

    # last line of the script
    # We can close it even if it is sys.stdin
    pdbfh.close()
    sys.exit(0)


def _prune_records(fhandle: TextIO) -> Generator[str]:
    """Prune records before processing.

    Scraps non-atomic records and records from water molecule.
    Replaces non-standard residue names by their standard counterparts.
    """
    atomic_record = ("ATOM", "HETATM")  # TODO: check if we need to keep ANISOU and TER records as well?
    water = "HOH"
    standard_resnames = {
        "MSE": "MET",
        "HIP": "HIS",
        "HIE": "HIS",
        "HID": "HIS",
        "HSE": "HIS",
        "HSD": "HIS",
    }

    for record in fhandle:
        resname = record[17:20]
        if record.startswith(atomic_record) and resname != water:
            standardized_resname = standard_resnames.get(resname, resname)
            yield record[:17] + standardized_resname + record[20:]


def pdb_prep(fhandle: TextIO) -> None:
    """Run all steps from pdb prep repo."""
    # step 1 - keep coordinates: removes non coordinate lines for simplicity
    # step 2 - delresname: remove waters
    # step 3 - rplresname: convert residue names to standard names, ex: MSE to MET
    new_pdb = _prune_records(fhandle)

    # step 4 - selaltloc: select most probable alternative location

    # step 5 - fixinsert: fix inserts
    # step 6 - sort: sort chains and resides, necessary for OpenMM
    # step 7 - reres: renumber residues from 1
    # step 8 - reatom: renumber atoms from 1
    # step 9 - tidy: tidy cleans the PDB, adds TER, etc.
