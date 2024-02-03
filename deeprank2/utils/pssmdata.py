from typing_extensions import Self

from deeprank2.molstruct.aminoacid import AminoAcid
from deeprank2.molstruct.residue import Residue


class PssmRow:
    """Holds data for one position-specific scoring matrix row."""

    def __init__(
        self,
        conservations: dict[AminoAcid, float],
        information_content: float,
    ):
        self._conservations = conservations
        self._information_content = information_content

    @property
    def conservations(self) -> dict[AminoAcid, float]:
        return self._conservations

    @property
    def information_content(self) -> float:
        return self._information_content

    def get_conservation(self, amino_acid: AminoAcid) -> float:
        return self._conservations[amino_acid]


class PssmTable:
    """Holds data for one position-specific scoring table."""

    def __init__(self, rows: list[PssmRow] | None = None):
        if rows is None:
            self._rows = {}
        else:
            self._rows = rows

    def __contains__(self, residue: Residue) -> bool:
        return residue in self._rows

    def __getitem__(self, residue: Residue) -> PssmRow:
        return self._rows[residue]

    def update(self, other: Self) -> None:
        """Can be used to merge two non-overlapping scoring tables."""
        self._rows.update(other._rows)  # noqa: SLF001
