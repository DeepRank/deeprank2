

class TopRowObject:
    def __init__(self, residue_name, atom_name, kwargs):
        self.residue_name = residue_name
        self.atom_name = atom_name
        self.kwargs = kwargs

    def __getitem__(self, key):
        return self.kwargs[key]
