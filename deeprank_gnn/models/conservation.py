class ConservationRow:
    "holds data for one position-specific scoring matrix row"

    def __init__(self, conservations, information_content):
        self._conservations = conservations
        self._information_content = information_content

    @property
    def conservations(self):
        return self._conservations

    @property
    def information_content(self):
        return self._information_content

    def get_conservation(self, amino_acid):
        return self._conservations[amino_acid]


class ConservationTable:
    "holds data for one position-specific scoring table"

    def __init__(self, rows=None):
        if rows is None:
            rows = {}
        else:
            self._rows = rows

    def __contains__(self, residue):
        return residue in self._rows

    def __getitem__(self, residue):
        return self._rows[residue]

    def update(self, other):
        "can be used to merge two non-overlapping scoring tables"

        self._rows.update(other._rows) # pylint: disable = protected-access
