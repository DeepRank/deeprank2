import numpy

from deeprank_gnn.models.pair import Pair
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.domain.forcefield import get_vanderwaals_potential, get_electrostatic_potential
from deeprank_gnn.tools.pdb import get_residue_distance, get_atom_distance


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
    def __init__(self, residue1: Residue, residue2: Residue):
        self._residue1 = residue1
        self._residue2 = residue2

        Pair.__init__(self, residue1, residue2)

    @property
    def distance(self) -> float:
        return get_residue_distance(self._residue1, self._residue2)

    @property
    def electrostatic_potential(self) -> float:
        potential = 0.0
        for atom1 in self._residue1.atoms:
            for atom2 in self._residue2.atoms:
                potential += get_electrostatic_potential(atom1, atom2)

        return potential

    @property
    def vanderwaals_potential(self) -> float:
        potential = 0.0
        for atom1 in self._residue1.atoms:
            for atom2 in self._residue2.atoms:
                potential += get_vanderwaals_potential(atom1, atom2)

        return potential


class AtomicContact(Contact):
    def __init__(self, atom1: Atom, atom2: Atom):
        self._atom1 = atom1
        self._atom2 = atom2

        Pair.__init__(self, atom1, atom2)

    @property
    def distance(self) -> float:
        return get_atom_distance(self._atom1, self._atom2)

    @property
    def electrostatic_potential(self) -> float:
        return get_electrostatic_potential(self._atom1, self._atom2)

    @property
    def vanderwaals_potential(self) -> float:
        return get_vanderwaals_potential(self._atom1, self._atom2)
