import re


class ResidueClassCriterium:  # noqa: D101
    def __init__(
        self,
        class_name: str,
        amino_acid_names: str | list[str],
        present_atom_names: list[str],
        absent_atom_names: list[str],
    ):
        self.class_name = class_name
        self.amino_acid_names = amino_acid_names
        self.present_atom_names = present_atom_names
        self.absent_atom_names = absent_atom_names

    def matches(self, amino_acid_name: str, atom_names: list[str]) -> bool:
        # check the amino acid name
        if self.amino_acid_names != "all" and not any(amino_acid_name == crit_amino_acid_name for crit_amino_acid_name in self.amino_acid_names):
            return False

        # check the atom names that should be absent
        if any(atom_name in self.absent_atom_names for atom_name in atom_names):
            return False

        # check the atom names that should be present
        return all(atom_name in atom_names for atom_name in self.present_atom_names)


class ResidueClassParser:  # noqa: D101
    _RESIDUE_CLASS_PATTERN = re.compile(r"([A-Z]{3,4}) *\: *name *\= *(all|[A-Z]{3})")
    _RESIDUE_ATOMS_PATTERN = re.compile(r"(present|absent)\(([A-Z0-9\, ]+)\)")

    @staticmethod
    def parse(file_: str) -> list[ResidueClassCriterium]:
        result = []
        for line in file_:
            match = ResidueClassParser._RESIDUE_CLASS_PATTERN.match(line)
            if not match:
                msg = f"Unparsable line: '{line}'"
                raise ValueError(msg)

            class_name = match.group(1)
            amino_acid_names = ResidueClassParser._parse_amino_acids(match.group(2))

            present_atom_names = []
            absent_atom_names = []
            for match in ResidueClassParser._RESIDUE_ATOMS_PATTERN.finditer(line[match.end() :]):  # noqa: B020
                atom_names = [name.strip() for name in match.group(2).split(",")]
                if match.group(1) == "present":
                    present_atom_names.extend(atom_names)

                elif match.group(1) == "absent":
                    absent_atom_names.extend(atom_names)

            result.append(ResidueClassCriterium(class_name, amino_acid_names, present_atom_names, absent_atom_names))
        return result

    @staticmethod
    def _parse_amino_acids(string: str) -> str | list[str]:
        if string.strip() == "all":
            return string.strip()
        return [name.strip() for name in string.split(",")]
