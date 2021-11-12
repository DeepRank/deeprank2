

class AminoAcid:
    def __init__(self, name, three_letter_code, one_letter_code):
        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

    @property
    def name(self):
        return self._name

    @property
    def three_letter_code(self):
        return self._three_letter_code

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(other) == type(self) and other.name == self.name
