from typing import Optional
import numpy as np
from deeprankcore.molstruct.structure import Chain
from deeprankcore.molstruct.aminoacid import AminoAcid
from deeprankcore.operations.pssm import PssmRow


class Residue:
    "represents a pdb residue"

    def __init__(
        self,
        chain: Chain,
        number: int,
        amino_acid: Optional[AminoAcid] = None,
        insertion_code: Optional[str] = None,
    ):
        """
        Args:
            chain(deeprank chain object): the chain that this residue belongs to
            number(int): the residue number
            amino_acid(deeprank amino acid, optional): the residue's amino acid (if it's part of a protein)
            insertion_code(str, optional): the pdb insertion code, if any
        """

        self._chain = chain
        self._number = number
        self._amino_acid = amino_acid
        self._insertion_code = insertion_code
        self._atoms = []

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self._chain == other._chain
            and self._number == other._number
            and self._insertion_code == other._insertion_code
        )

    def __hash__(self) -> hash:
        return hash((self._chain, self._number, self._insertion_code))

    def get_pssm(self) -> PssmRow:
        """ if the residue's chain has pssm info linked to it,
            then return the part that belongs to this residue
        """

        pssm = self._chain.pssm
        if pssm is None:
            raise ValueError(f"pssm not set on {self._chain}")

        return pssm[self]

    @property
    def number(self) -> int:
        return self._number

    @property
    def chain(self):
        return self._chain

    @property
    def amino_acid(self) -> AminoAcid:
        return self._amino_acid

    @property
    def atoms(self):
        return self._atoms

    @property
    def number_string(self) -> str:
        "contains both the number and the insertion code (if any)"

        if self._insertion_code is not None:
            return f"{self._number}{self._insertion_code}"

        return str(self._number)

    @property
    def insertion_code(self) -> str:
        return self._insertion_code

    def add_atom(self, atom):
        self._atoms.append(atom)

    def __repr__(self) -> str:
        return f"{self._chain} {self.number_string}"

    @property
    def position(self) -> np.array:
        return np.mean([atom.position for atom in self._atoms], axis=0)