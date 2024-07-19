import logging
import os

from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue
from deeprank2.utils.parsing.patch import PatchActionType, PatchParser
from deeprank2.utils.parsing.residue import ResidueClassParser
from deeprank2.utils.parsing.top import TopParser
from deeprank2.utils.parsing.vdwparam import ParamParser, VanderwaalsParam

_log = logging.getLogger(__name__)

_forcefield_directory_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../domain/forcefield"))


class AtomicForcefield:  # noqa: D101
    def __init__(self):
        top_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-5_new.top")
        with open(top_path, encoding="utf-8") as f:
            self._top_rows = {(row.residue_name, row.atom_name): row for row in TopParser.parse(f)}

        patch_path = os.path.join(_forcefield_directory_path, "patch.top")
        with open(patch_path, encoding="utf-8") as f:
            self._patch_actions = PatchParser.parse(f)

        residue_class_path = os.path.join(_forcefield_directory_path, "residue-classes")
        with open(residue_class_path, encoding="utf-8") as f:
            self._residue_class_criteria = ResidueClassParser.parse(f)

        param_path = os.path.join(_forcefield_directory_path, "protein-allhdg5-4_new.param")
        with open(param_path, encoding="utf-8") as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _find_matching_residue_class(self, residue: Residue) -> str | None:
        for criterium in self._residue_class_criteria:
            if criterium.matches(
                residue.amino_acid.three_letter_code,
                [atom.name for atom in residue.atoms],
            ):
                return criterium.class_name

        return None

    def get_vanderwaals_parameters(self, atom: Atom) -> VanderwaalsParam:
        atom_name = atom.name

        if atom.residue.amino_acid is None:
            _log.warning(f"no amino acid for {atom}; three letter code set to XXX")
            residue_name = "XXX"
        else:
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
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and residue_class == action.selection.residue_type and "TYPE" in action:
                    type_ = action["TYPE"]

        if type_ is None:
            _log.warning(
                f"Atom {atom} is unknown to the forcefield, vanderwaals_parameters set to (0.0, 0.0, 0.0, 0.0).\n"
                "   This will affect `vanderwaals` feature.\n"
                "       Check https://deeprank2.readthedocs.io/en/latest/features.html#nonbond-energies for more details.",
            )
            return VanderwaalsParam(0.0, 0.0, 0.0, 0.0)
        return self._vanderwaals_parameters[type_]

    def get_charge(self, atom: Atom) -> float:
        """Get the charge of a given `Atom`.

        Args:
            atom: the atom to get the charge for

        Returns:
            the charge of the given atom.
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
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and residue_class == action.selection.residue_type:
                    charge = float(action["CHARGE"])

        if charge is None:
            _log.warning(
                f"Atom {atom} is unknown to the forcefield, `electrostatic` and `atom_charge` charge is set to 0.0.\n"
                "   This will affect `electrostatic` and `atom_charge` features.\n"
                "       Check https://deeprank2.readthedocs.io/en/latest/features.html#nonbond-energies for more details.",
            )
            return 0.0
        return charge


atomic_forcefield = AtomicForcefield()
