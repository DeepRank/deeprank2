from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from deeprank2.molstruct.aminoacid import AminoAcid
from deeprank2.molstruct.structure import Chain
from deeprank2.utils.pssmdata import PssmRow

if TYPE_CHECKING:
    from deeprank2.molstruct.atom import Atom


class Residue:
    """One protein residue in a `PDBStructure`.

    A `Residue` is the basic building block of proteins and protein complex,
    here represented by `PDBStructures`.
    Each residue is of a certain `AminoAcid` type and consists of multiple
    `Atom`s.
    """

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
        if isinstance(other, Residue):
            return (self._chain == other._chain
                    and self._number == other._number
                    and self._insertion_code == other._insertion_code
            )
        return NotImplemented

    def __hash__(self) -> hash:
        return hash((self._number, self._insertion_code))

    def get_pssm(self) -> PssmRow:
        """Load pssm info linked to the residue."""
        pssm = self._chain.pssm
        if pssm is None:
            raise FileNotFoundError(f'No pssm file found for Chain {self._chain}.')
        return pssm[self]

    @property
    def number(self) -> int:
        return self._number

    @property
    def chain(self) -> Chain:
        return self._chain

    @property
    def amino_acid(self) -> AminoAcid:
        return self._amino_acid

    @property
    def atoms(self) -> list[Atom]:
        return self._atoms

    @property
    def number_string(self) -> str:
        """Contains both the number and the insertion code (if any)."""
        if self._insertion_code is not None:
            return f"{self._number}{self._insertion_code}"
        return str(self._number)

    @property
    def insertion_code(self) -> str:
        return self._insertion_code

    def add_atom(self, atom: Atom):
        self._atoms.append(atom)

    def __repr__(self) -> str:
        return f"{self._chain} {self.number_string}"

    @property
    def position(self) -> np.array:
        return np.mean([atom.position for atom in self._atoms], axis=0)

    def get_center(self) -> np.ndarray:
        """Find the center position of a `Residue`.

        Center position is found as follows:
        1. find beta carbon
        2. if no beta carbon is found: find alpha carbon
        3. if no alpha carbon is found: take the mean of the atom positions
        """
        betas = [atom for atom in self.atoms if atom.name == "CB"]
        if len(betas) > 0:
            return betas[0].position

        alphas = [atom for atom in self.atoms if atom.name == "CA"]
        if len(alphas) > 0:
            return alphas[0].position

        if len(self.atoms) == 0:
            raise ValueError(f"cannot get the center position from {self}, because it has no atoms")

        return np.mean([atom.position for atom in self.atoms], axis=0)


class SingleResidueVariant:
    """A single residue mutation of a PDBStrcture."""

    def __init__(self, residue: Residue, variant_amino_acid: AminoAcid):
        """
        Args:
            residue (Residue): the `Residue` object from the PDBStructure that is mutated.
            variant_amino_acid (AminoAcid): the amino acid that the `Residue` is mutated into.
        """
        self._residue = residue
        self._variant_amino_acid = variant_amino_acid

    @property
    def residue(self) -> Residue:
        return self._residue

    @property
    def variant_amino_acid(self) -> AminoAcid:
        return self._variant_amino_acid

    @property
    def wildtype_amino_acid(self) -> AminoAcid:
        return self._residue.amino_acid
