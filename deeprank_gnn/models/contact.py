import numpy

from deeprank_gnn.models.pair import Pair
from deeprank_gnn.models.structure import Residue, Atom


class Contact(Pair):
    @property
    def distance(self) -> float:
        raise TypeError("called unimplemented method on interface class: Contact")

    @property
    def electrostatic_potential(self) -> float:
        raise TypeError("called unimplemented method on interface class: Contact")

    @property
    def vanderwaals_potential(self) -> float:
        raise TypeError("called unimplemented method on interface class: Contact")


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
    def residue1(self):
        return self.item1

    @property
    def residue2(self):
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
    def atom1(self):
        return self.item1

    @property
    def atom2(self):
        return self.item2
