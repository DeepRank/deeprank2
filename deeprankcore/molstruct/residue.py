from typing import Optional

import numpy as np

from deeprankcore.molstruct.aminoacid import AminoAcid
from deeprankcore.molstruct.structure import Chain
from deeprankcore.utils.pssmdata import PssmRow


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
            chain (:class:`Chain`): The chain that this residue belongs to.
            number (int): the residue number
            amino_acid (:class:`AminoAcid`, optional): The residue's amino acid (if it's part of a protein).
                Defaults to None.
            insertion_code (str, optional): The pdb insertion code, if any. Defaults to None.
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
        return hash((self._number, self._insertion_code))

    def get_pssm(self) -> PssmRow:
        """ 
        If the residue's chain has pssm info linked to it,
        then return the part that belongs to this residue.
        """

        pssm = self._chain.pssm
        if pssm is None:
            raise FileNotFoundError(f'No pssm file found for Chain {self._chain}.')

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


def get_residue_center(residue: Residue) -> np.ndarray:
    """Chooses a center position for a residue. 
    
    Based on the atoms it has:
    1. find beta carbon, if present
    2. find alpha carbon, if present
    3. else take the mean of the atom positions
    """

    betas = [atom for atom in residue.atoms if atom.name == "CB"]
    if len(betas) > 0:
        return betas[0].position

    alphas = [atom for atom in residue.atoms if atom.name == "CA"]
    if len(alphas) > 0:
        return alphas[0].position

    if len(residue.atoms) == 0:
        raise ValueError(f"cannot get the center position from {residue}, because it has no atoms")

    return np.mean([atom.position for atom in residue.atoms], axis=0)
