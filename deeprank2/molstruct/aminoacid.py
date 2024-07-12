from enum import Enum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


class Polarity(Enum):
    """One-hot encoding of the amino acid polarity."""

    NONPOLAR = 0
    POLAR = 1
    NEGATIVE = 2
    POSITIVE = 3

    @property
    def onehot(self) -> NDArray:
        t = np.zeros(4)
        t[self.value] = 1.0
        return t


class AminoAcid:
    """An amino acid represents the type of `Residue` in a `PDBStructure`.

    Args:
        name: Full name of the amino acid.
        three_letter_code: Three-letter code of the amino acid (as in PDB).
        one_letter_code: One-letter of the amino acid (as in fasta).
        charge: Charge of the amino acid.
        polarity: The polarity of the amino acid.
        size: The number of non-hydrogen atoms in the side chain.
        mass: Average residue mass (i.e. mass of amino acid - H20) in Daltons.
        pI: Isolectric point; pH at which the molecule has no net electric charge.
        hydrogen_bond_donors: Number of hydrogen bond donors.
        hydrogen_bond_acceptors: Number of hydrogen bond acceptors.
        index: The rank of the amino acid, used for computing one-hot encoding.
    """

    def __init__(
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
    def onehot(self) -> NDArray:
        if self._index is None:
            msg = f"Amino acid {self._name} index is not set, thus no onehot can be computed."
            raise ValueError(msg)
        # 20 canonical amino acids
        # selenocysteine and pyrrolysine are indexed as cysteine and lysine, respectively
        a = np.zeros(20)
        a[self._index] = 1.0
        return a

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self) -> hash:
        return hash(self.name)

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, AminoAcid):
            return other.name == self.name
        return NotImplemented

    def __repr__(self) -> str:
        return self._three_letter_code
