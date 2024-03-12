# All code in this subpackage has been adapted from https://github.com/DeepRank/pdbprep,
# which is published under an Apache 2.0 licence

import sys
from collections import defaultdict
from collections.abc import Generator
from typing import TextIO

# define record columns for each datum
_ATOMNAME_COLS = slice(12, 16)
_RESNAME_COLS = slice(17, 20)
_CHAIN_COLS = slice(21, 22)
_RESNUM_COLS = slice(22, 27)  # this includes both the residue number and insertion code
_OCCUPANCY_COLS = slice(54, 60)


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
        resname = record[_RESNAME_COLS]
        if record.startswith(atomic_record) and resname != water:
            standardized_resname = standard_resnames.get(resname, resname)
            yield record[:17] + standardized_resname + record[20:]


def _find_low_occ_records(pdb: list[str]) -> list[int]:
    """Helper function to identify records with lowest occupancy alternate locations.

    In case an atom is detected at more than one position (e.g. due to alternate conformations), the structure will
    contain the same atom multiple times with separate "alternate location indicators" (col 17 of the pdb record).
    Each location will have a certain occupancy, i.e. proportion of structures where this particular location is found
    (and thus all occupancies for a given atom sum to 1).

    This function first identifies atoms that are listed more than once in a pdb file, based on their chain identifier
    (col 22), residue sequence number (col 23-26), and atom name (col 13-16). It then identifies the record with the
    highest occupancy for each atom (in case of equal occupancy, the first entry is considered higher). From this, a
    list of indices is returned representing the records that do not contain the highest occupancy for the atom in that
    record.

    Args:
        pdb: list of records (lines) from a pdb file

    Returns:
        list of indices of records that do not contain the highest occupancy location
    """
    # define record columns for each datum

    atom_indentiers = [record[_CHAIN_COLS] + record[_RESNUM_COLS] + record[_ATOMNAME_COLS] for record in pdb]

    # create a dictionary containing only duplicated atom_indentiers (keys) and their indices in pdb (values)
    # from: https://stackoverflow.com/a/11236042/5170442
    duplicates = defaultdict(list)
    for i, atom in enumerate(atom_indentiers):
        duplicates[atom].append(i)
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}

    highest_occupancies = {}
    for atom, record_indices in duplicates.items():
        highest_occ = 0
        for i in record_indices:
            occupancy = pdb[i][_OCCUPANCY_COLS]
            if occupancy > highest_occ:
                # only keep the record with the highest occupancy; in case of tie keep the first
                highest_occ = occupancy
                highest_occupancies[atom] = i
    return [x for xs in duplicates.values() for x in xs if x not in highest_occupancies.values()]


def pdb_prep(fhandle: TextIO) -> None:
    """Run all steps from pdb prep repo."""
    # step 1 - keep coordinates: removes non coordinate lines for simplicity
    # step 2 - delresname: remove waters
    # step 3 - rplresname: convert residue names to standard names, ex: MSE to MET
    _new_pdb = _prune_records(fhandle)

    # step 4 - selaltloc: select most probable alternative location

    # step 5 - fixinsert: fix inserts
    # step 6 - sort: sort chains and resides, necessary for OpenMM
    # step 7 - reres: renumber residues from 1
    # step 8 - reatom: renumber atoms from 1
    # step 9 - tidy: tidy cleans the PDB, adds TER, etc.
