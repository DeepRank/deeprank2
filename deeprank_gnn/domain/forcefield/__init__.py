import os
import logging

import numpy

from deeprank_gnn.models.structure import Atom
from deeprank_gnn.tools.pdb import get_atom_distance
from deeprank_gnn.models.forcefield.patch import PatchActionType
from deeprank_gnn.tools.forcefield.top import TopParser
from deeprank_gnn.tools.forcefield.patch import PatchParser
from deeprank_gnn.tools.forcefield.residue import ResidueClassParser
from deeprank_gnn.tools.forcefield.param import ParamParser
from deeprank_gnn.models.error import UnknownAtomError


_log = logging.getLogger(__name__)


_forcefield_directory_path = os.path.dirname(os.path.abspath(__file__))


VANDERWAALS_DISTANCE_OFF = 20.0
VANDERWAALS_DISTANCE_ON = 1.0

SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)

EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636

class AtomicForcefield:
    def __init__(self):
        top_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-5_new.top")
        with open(top_path, 'rt') as f:
            self._top_rows = {(row.residue_name, row.atom_name): row for row in TopParser.parse(f)}

        patch_path = os.path.join(_forcefield_directory_path, "patch.top")
        with open(patch_path, 'rt') as f:
            self._patch_actions = PatchParser.parse(f)

        residue_class_path = os.path.join(_forcefield_directory_path, "residue-classes")
        with open(residue_class_path, 'rt') as f:
            self._residue_class_criteria = ResidueClassParser.parse(f)

        param_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-4_new.param")
        with open(param_path, 'rt') as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _find_matching_residue_class(self, residue):
        for criterium in self._residue_class_criteria:
            if criterium.matches(residue.amino_acid.three_letter_code, [atom.name for atom in residue.atoms]):
                return criterium.class_name

        return None

    def get_vanderwaals_parameters(self, atom):
        type_ = self._get_type(atom)

        return self._vanderwaals_parameters[type_]

    def _get_type(self, atom):
        atom_name = atom.name

        if atom.residue.amino_acid is None:
            raise UnknownAtomError("no amino acid for {}".format(atom))

        residue_name = atom.residue.amino_acid.three_letter_code

        type_ = None

        # check top
        top_key = (residue_name, atom_name)
        if top_key in self._top_rows:
            type_ = self._top_rows[top_key]["type"]

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and \
                        residue_class == action.selection.residue_type and "TYPE" in action:

                    type_ = action["TYPE"]

        if type_ is None:
            raise UnknownAtomError("not mentioned in top or patch: {}".format(top_key))

        return type_

    def get_charge(self, atom):
        """
            Args:
                atom(Atom): the atom to get the charge for
            Returns(float): the charge of the given atom
        """

        atom_name = atom.name
        amino_acid_code = atom.residue.amino_acid.three_letter_code

        charge = None

        # check top
        top_key = (amino_acid_code, atom_name)
        if top_key in self._top_rows:
            charge = float(self._top_rows[top_key]["charge"])

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and \
                        residue_class == action.selection.residue_type:

                    charge = float(action["CHARGE"])

        if charge is None:
            raise UnknownAtomError("not mentioned in top or patch: {}".format(top_key))

        return charge


atomic_forcefield = AtomicForcefield()


def get_electrostatic_potential(atom1: Atom, atom2: Atom) -> float:
    "Calculates the Coulomb electrostatic potential between two atoms"

    charge1 = atomic_forcefield.get_charge(atom1)
    charge2 = atomic_forcefield.get_charge(atom2)
    distance = get_atom_distance(atom1, atom2)

    return charge1 * charge2 * COULOMB_CONSTANT / (EPSILON0 * numpy.square(distance))


def get_vanderwaals_potential(atom1: Atom, atom2: Atom) -> float:
    "Caluclated the vanderwaals potential between two atoms"

    parameters1 = atomic_forcefield.get_vanderwaals_parameters(atom1)
    parameters2 = atomic_forcefield.get_vanderwaals_parameters(atom2)
    distance = get_atom_distance(atom1, atom2)
    squared_distance = numpy.square(distance)

    if atom1.residue.chain == atom2.residue.chain:
        epsilon = numpy.sqrt(parameters1.intra_epsilon * parameters2.intra_epsilon)
        sigma = 0.5 * (parameters1.intra_sigma + parameters2.intra_sigma)
    else:
        epsilon = numpy.sqrt(parameters1.inter_epsilon * parameters2.inter_epsilon)
        sigma = 0.5 * (parameters1.inter_sigma + parameters2.inter_sigma)

    if distance < VANDERWAALS_DISTANCE_ON:
        prefactor = 0.0
    elif distance > VANDERWAALS_DISTANCE_OFF:
        prefactor = 1.0
    else:
        vanderwaals_constant_factor = (SQUARED_VANDERWAALS_DISTANCE_OFF - SQUARED_VANDERWAALS_DISTANCE_ON) ** 3
        prefactor = (((SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distance) ** 2) *
                     (SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distances - 3 *
                      (SQUARED_VANDERWAALS_DISTANCE_ON - squared_distances)) / vanderwaals_constant_factor)

    return 4.0 * epsilon * (((sigma / distance) ** 12) - ((sigma / distance) ** 6)) * prefactor

