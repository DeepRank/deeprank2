import numpy as np
from enum import Enum


class Polarity(Enum):
    """A value to express a residue's polarity."""

    NONPOLAR = 0
    POLAR = 1
    NEGATIVE_CHARGE = 2
    POSITIVE_CHARGE = 3

    @property
    def onehot(self):
        t = np.zeros(4)
        t[self.value] = 1.0

        return t


class AminoAcid:
    """A value to represent one of the amino acids."""

    def __init__( # pylint: disable=too-many-arguments
        self,
        name: str,
        three_letter_code: str,
        one_letter_code: str,
        charge: int,
        polarity: Polarity,
        size: int,
        mass: float,
        pI: float, 
        hydrogen_bond_donors: int,
        hydrogen_bond_acceptors: int,
        index: int,
    ):
        """
        Args:
            name (str): Full name of the amino acid.
            three_letter_code (str): Three-letter code of the amino acid (as in PDB).
            one_letter_code (str): One-letter of the amino acid (as in fasta).
            charge (int): Charge of the amino acid.
            polarity (:class:`Polarity`): The polarity of the amino acid.
            size (int): The number of non-hydrogen atoms in the side chain.
            mass (float): Average residue mass (i.e. mass of amino acid - H20) in Daltons.
            pI (float): Isolectric point; pH at which the molecule has no net electric charge.
            hydrogen_bond_donors (int): Number of hydrogen bond donors.
            hydrogen_bond_acceptors (int): Number of hydrogen bond acceptors.
            index (int): The rank of the amino acid, used for computing one-hot encoding.
        """

        # amino acid nomenclature
        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        # side chain properties
        self._charge = charge
        self._polarity = polarity
        self._size = size
        self._mass = mass
        self._pI = pI
        self._hydrogen_bond_donors = hydrogen_bond_donors
        self._hydrogen_bond_acceptors = hydrogen_bond_acceptors

        # one hot encoding
        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @property
    def three_letter_code(self) -> str:
        return self._three_letter_code

    @property
    def one_letter_code(self) -> str:
        return self._one_letter_code

    @property
    def charge(self) -> int:
        return self._charge

    @property
    def polarity(self) -> Polarity:
        return self._polarity

    @property
    def size(self) -> int:
        return self._size

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def pI(self) -> float:
        return self._pI

    @property
    def hydrogen_bond_donors(self) -> int:
        return self._hydrogen_bond_donors

    @property
    def hydrogen_bond_acceptors(self) -> int:
        return self._hydrogen_bond_acceptors

    @property
    def onehot(self) -> np.ndarray:
        if self._index is None:
            raise ValueError(
                "amino acid {self._name} index is not set, thus no onehot can be computed"
            )
        # 20 canonical amino acids
        # selenocysteine and pyrrolysine are indexed as cysteine and lysine, respectively
        a = np.zeros(20)
        a[self._index] = 1.0

        return a

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.name == self.name

    def __repr__(self):
        return self._name
