from __future__ import annotations

from enum import Enum

import numpy as np

from deeprank2.molstruct.residue import Residue


class AtomicElement(Enum):
    """One-hot encoding of the atomic element (or atom type)."""
    C = 1
    O = 2 # noqa: pycodestyle
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
    """One atom in a PDBStructure."""

    def __init__( # pylint: disable=too-many-arguments
        self,
        residue: Residue,
        name: str,
        element: AtomicElement,
        position: np.array,
        occupancy: float,
    ):
        """
        Args:
            residue (:class:`Residue`): The residue that this atom belongs to.
            name (str): Pdb atom name.
            element (:class:`AtomicElement`): The chemical element.
            position (np.array): Pdb position xyz of this atom.
            occupancy (float): Pdb occupancy value.
                This represents the proportion of structures where the atom is detected at a given position.
                Sometimes a single atom can be detected at multiple positions. In that case separate structures exist where sum(occupancy) == 1.
                Note that only the highest occupancy atom is used by deeprank2 (see tools.pdb._add_atom_to_residue)
        """
        self._residue = residue
        self._name = name
        self._element = element
        self._position = position
        self._occupancy = occupancy

    def __eq__(self, other) -> bool:
        if isinstance (other, Atom):
            return (self._residue == other._residue
                    and self._name == other._name)
        return NotImplemented

    def __hash__(self) -> hash:
        return hash((tuple(self._position), self._element, self._name))

    def __repr__(self) -> str:
        return f"{self._residue} {self._name}"

    def change_altloc(self, alternative_atom: Atom):
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
    def position(self) -> np.array:
        return self._position

    @property
    def residue(self) -> Residue:
        return self._residue
