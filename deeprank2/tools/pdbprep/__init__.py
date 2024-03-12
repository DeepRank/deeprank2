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


def _select_alt_location(pdb: list[str]) -> Generator[str]:
    """Select alternate location."""
    register = {}  # register atom information
    prev_chain = None  # register previous chain

    # This loop will collect information on the different atoms throughout the PDB file until a new chain or any terminal line is
    # found. At that point, the collected information is processed because all altlocs for that block have been defined.
    for nline, record in enumerate(pdb):  # line number will be used to sort lines after selecting the desired alternative location
        atomname = record[12:16]
        altloc = record[16]
        chain = record[21:22]
        resnum = record[22:27]  # resnum (22-25) + insertion code (26) is taken to identify different residues

        # process lines because we enter a new chain
        if chain != prev_chain:
            yield from _process_altloc(register)
            register = {}

        # add info to dictionary in a hierarchically organized manner
        resnum_d: dict = register.setdefault(resnum, {})
        atomname_d: dict = resnum_d.setdefault(atomname, {})
        altloc_d: list = atomname_d.setdefault(altloc, [])
        altloc_d.append((nline, record))

        prev_chain = chain

    # at the end of the PDB, process the remaining lines
    yield from _process_altloc(register)


def _process_altloc(register: dict[str, dict[str, dict[str, list[tuple[int, str]]]]]) -> Generator[str]:
    # TODO: Reduce complexity of `register` if possible
    """Processes the collected atoms according to the selaltloc option."""
    lines_to_yield = []

    anisou_record = ("ANISOU",)  # anisou lines are treated specially and always follow atom records

    for atomnames in register.values():
        for altlocs in atomnames.values():
            all_lines: list[tuple[int, str]] = list(*altlocs.values())  # all alternative locations for the atom

            # identify the highest occupancy combining dictionary and sorting
            occ_line_dict = {}  # TODO: rename
            for line_number, line in all_lines:
                occupancy = line[54:60]
                occ_line_dict[occupancy] = [(line_number, line)]

            # sort keys by occupancy
            keys_ = sorted(occ_line_dict.keys(), key=lambda x: float(x.strip()), reverse=True)  # TODO: rename once I know what this is used for

            these_atom_lines = occ_line_dict[keys_[0]]
            if len(keys_) == 1 and len(these_atom_lines) > 1:
                # address "take first if occ is the same"
                # see: https://github.com/haddocking/pdb-tools/issues/153#issuecomment-1488627668
                lines_to_yield.extend(_remove_altloc(these_atom_lines[0:1]))

                # if there's ANISOU, add it
                if these_atom_lines[1][1].startswith(anisou_record):
                    lines_to_yield.extend(_remove_altloc(these_atom_lines[1:2]))

            # this should run when there are more than one key or
            # the key has only one atom line. Keys are the occ
            # value.
            else:
                # when occs are different, select the highest one
                lines_to_yield.extend(_remove_altloc(these_atom_lines))

            del all_lines, occ_line_dict

    # lines are sorted to the line number so that the output is sorted
    # the same way as in the input PDB
    lines_to_yield.sort(key=lambda x: x[0])

    # the line number is ignored, only the line is yield
    for _, line in lines_to_yield:
        yield line


def _remove_altloc(lines: str) -> Generator[str]:
    # the altloc ID is removed in processed altloc lines
    for line_num, line in lines:
        yield (line_num, line[:16] + " " + line[17:])


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
