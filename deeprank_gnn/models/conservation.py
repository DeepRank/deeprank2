

class ConservationRow:
    "holds data for one residue"

    def __init__(self, conservations, information_content):
        self._conservations = conservations
        self._information_content = information_content

    @property
    def conservations(self):
        return self._conservations

    @property
    def information_content(self):
        return self._information_content


class ConservationTable:
    "holds data for one pssm file"

    def __init__(self, rows={}):
        self._rows = rows

    def __contains__(self, residue):
        return residue in self._rows

    def __getitem__(self, residue):
        return self._rows[residue]

    def update(self, other):
        self._rows.update(other._rows)

