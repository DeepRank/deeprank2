from abc import ABC
from typing import Any

from typing_extensions import Self

from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue


class Pair:
    """A hashable, comparable object for any set of two inputs where order doesn't matter.

    Args:
        item1: The pair's first object, must be convertable to string.
        item2: The pair's second object, must be convertable to string.
    """

    def __init__(self, item1: Any, item2: Any):  # noqa: ANN401
        self.item1 = item1
        self.item2 = item2

    def __hash__(self) -> hash:
        """The hash should be solely based on the two paired items, not on their order."""
        s1 = str(self.item1)
        s2 = str(self.item2)
        if s1 < s2:
            return hash(s1 + s2)
        return hash(s2 + s1)

    def __eq__(self, other: Self) -> bool:
        """Compare the pairs as sets, so the order doesn't matter."""
        if isinstance(other, Pair):
            return self.item1 == other.item1 and self.item2 == other.item2 or self.item1 == other.item2 and self.item2 == other.item1
        return NotImplemented

    def __iter__(self):
        # Iterate over the two items in the pair.
        return iter([self.item1, self.item2])

    def __repr__(self) -> str:
        return str(self.item1) + str(self.item2)


class Contact(Pair, ABC):
    """Parent class to bind `ResidueContact` and `AtomicContact` objects."""


class ResidueContact(Contact):
    """A contact between two residues from a structure."""

    def __init__(self, residue1: Residue, residue2: Residue):
        self._residue1 = residue1
        self._residue2 = residue2
        super().__init__(residue1, residue2)

    @property
    def residue1(self) -> Residue:
        return self.item1

    @property
    def residue2(self) -> Residue:
        return self.item2


class AtomicContact(Contact):
    """A contact between two atoms from a structure."""

    def __init__(self, atom1: Atom, atom2: Atom):
        self._atom1 = atom1
        self._atom2 = atom2
        super().__init__(atom1, atom2)

    @property
    def atom1(self) -> Atom:
        return self.item1

    @property
    def atom2(self) -> Atom:
        return self.item2
