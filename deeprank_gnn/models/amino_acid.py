import numpy


class AminoAcid:
    "a value to represent one of the amino acids"

    def __init__(self, name, three_letter_code, one_letter_code, charge=None, polarity=None, size=None, index=None):
        """
            Args:
                name(str): unique name for the amino acid
                three_letter_code(str): code of the amino acid, as in PDB
                one_letter_code(str): letter of the amino acid, as in fasta
                charge(float, optional): the charge property of the amino acid
                polarity(deeprank polarity enum, optional): the polarity property of the amino acid
                size(int, optional): the number of heavy atoms in the side chain
                index(int, optional): the rank of the amino acid, used for computing one-hot encoding
        """

        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        self._size = size
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

        a = numpy.zeros(20)  # assumed that there are only 20 different amino acids
        a[self._index] = 1.0

        return a

    @property
    def charge(self):
        return self._charge

    @property
    def polarity(self):
        return self._polarity

    @property
    def size(self):
        return self._size

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(other) == type(self) and other.name == self.name

    def __repr__(self):
        return self._name
