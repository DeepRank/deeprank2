from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from deeprank2.molstruct.residue import Residue


class AtomicElement(Enum):
    """One-hot encoding of the atomic element (or atom type)."""

    C = 1
    O = 2  # noqa: E741
    N = 3
    S = 4
    P = 5
    H = 6

    @property
    def onehot(self) -> np.array:
        value = np.zeros(max(el.value for el in AtomicElement))
        value[self.value - 1] = 1.0
        return value


class Atom:
    """One atom in a PDBStructure.

    Args:
        residue: The residue that this atom belongs to.
        name: Pdb atom name.
        element: The chemical element.
        position: Pdb position xyz of this atom.
        occupancy: Pdb occupancy value.
            This represents the proportion of structures where the atom is detected at a given position.
            Sometimes a single atom can be detected at multiple positions. In that case separate structures exist where sum(occupancy) == 1.
            Note that only the highest occupancy atom is used by deeprank2 (see tools.pdb._add_atom_to_residue).
    """

    def __init__(
        self,
        residue: Residue,
        name: str,
        element: AtomicElement,
        position: NDArray,
        occupancy: float,
    ):
        self._residue = residue
        self._name = name
        self._element = element
        self._position = position
        self._occupancy = occupancy

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Atom):
            return self._residue == other._residue and self._name == other._name
        return NotImplemented

    def __hash__(self) -> hash:
        return hash((tuple(self._position), self._element, self._name))

    def __repr__(self) -> str:
        return f"{self._residue} {self._name}"

    def change_altloc(self, alternative_atom: Atom) -> None:
        """Replace the atom's location by another atom's location."""
        self._position = alternative_atom.position
        self._occupancy = alternative_atom.occupancy

    @property
    def name(self) -> str:
        return self._name

    @property
    def element(self) -> AtomicElement:
        return self._element

    @property
    def occupancy(self) -> float:
        return self._occupancy

    @property
    def position(self) -> NDArray:
        return self._position

    @property
    def residue(self) -> Residue:
        return self._residue
