from abc import ABC, abstractmethod

import numpy

from deeprank_gnn.models.pair import Pair
from deeprank_gnn.models.structure import Residue, Atom


class Contact(Pair, ABC):
    @property
    @abstractmethod
    def distance(self) -> float:
        pass

    @property
    @abstractmethod
    def electrostatic_potential(self) -> float:
        pass

    @property
    def vanderwaals_potential(self) -> float:
        pass


class ResidueContact(Contact):
    "A contact between two residues from a structure"

    def __init__(self, residue1: Residue, residue2: Residue,
                 distance: float,
                 electrostatic_potential: float,
                 vanderwaals_potential: float):

        self._residue1 = residue1
        self._residue2 = residue2

        Pair.__init__(self, residue1, residue2)

        self._distance = distance
        self._electrostatic_potential = electrostatic_potential
        self._vanderwaals_potential = vanderwaals_potential

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def electrostatic_potential(self) -> float:
        return self._electrostatic_potential

    @property
    def vanderwaals_potential(self) -> float:
        return self._vanderwaals_potential

    @property
    def residue1(self) -> Residue:
        return self.item1

    @property
    def residue2(self) -> Residue:
        return self.item2


class AtomicContact(Contact):
    "A contact between two atoms from a structure"

    def __init__(self, atom1: Atom, atom2: Atom,
                 distance: float,
                 electrostatic_potential: float,
                 vanderwaals_potential: float):

        self._atom1 = atom1
        self._atom2 = atom2

        Pair.__init__(self, atom1, atom2)

        self._distance = distance
        self._electrostatic_potential = electrostatic_potential
        self._vanderwaals_potential = vanderwaals_potential

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def electrostatic_potential(self) -> float:
        return self._electrostatic_potential

    @property
    def vanderwaals_potential(self) -> float:
        return self._vanderwaals_potential

    @property
    def atom1(self) -> Atom:
        return self.item1

    @property
    def atom2(self) -> Atom:
        return self.item2
