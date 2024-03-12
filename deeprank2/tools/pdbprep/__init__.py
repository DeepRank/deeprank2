# All code in this subpackage has been adapted from https://github.com/DeepRank/pdbprep,
# which is published under an Apache 2.0 licence

import sys


def write_pdb(new_pdb, pdbfh):
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
