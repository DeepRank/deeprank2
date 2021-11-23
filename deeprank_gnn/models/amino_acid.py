import numpy


class AminoAcid:
    def __init__(self, name, three_letter_code, one_letter_code, charge=None, polarity=None, index=None):
        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        self._charge = charge
        self._polarity = polarity
        self._index = index

    @property
    def name(self):
        return self._name

    @property
    def three_letter_code(self):
        return self._three_letter_code

    @property
    def one_letter_code(self):
        return self._one_letter_code

    @property
    def onehot(self):
        if self._index is None:
            raise ValueError("amino acid {} index is not set, thus no onehot can be computed".format(self._name))

        a = numpy.zeros(20)
        a[self._index] = 1.0

        return self._index

    @property
    def charge(self):
        return self._charge

    @property
    def polarity(self):
        return self._polarity

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(other) == type(self) and other.name == self.name

    def __repr__(self):
        return self._name
