import re
from enum import Enum
from typing import Any

# ruff: noqa: D101


class PatchActionType(Enum):
    MODIFY = 1
    ADD = 2


class PatchSelection:
    def __init__(self, residue_type: str, atom_name: str):
        self.residue_type = residue_type
        self.atom_name = atom_name


class PatchAction:
    def __init__(self, type_: str, selection: PatchSelection, kwargs: dict[str, Any]):
        self.type = type_
        self.selection = selection
        self.kwargs = kwargs

    def __contains__(self, key: str):
        return key in self.kwargs

    def __getitem__(self, key: str):
        return self.kwargs[key]


class PatchParser:
    STRING_VAR_PATTERN = re.compile(r"([A-Z]+)=([A-Z0-9]+)")
    NUMBER_VAR_PATTERN = re.compile(r"([A-Z]+)=(\-?[0-9]+\.[0-9]+)")
    ACTION_PATTERN = re.compile(r"^([A-Z]{3,4})\s+([A-Z]+)\s+ATOM\s+([A-Z0-9]{1,3})\s+(.*)$")

    @staticmethod
    def _parse_action_type(s: str) -> PatchActionType:
        for type_ in PatchActionType:
            if type_.name == s:
                return type_

        msg = f"Unmatched residue action: {s!r}"
        raise ValueError(msg)

    @staticmethod
    def parse(file_: str) -> list[PatchAction]:
        result = []
        for line in file_:
            if line.startswith(("#", "!")) or len(line.strip()) == 0:
                continue

            m = PatchParser.ACTION_PATTERN.match(line)
            if not m:
                msg = f"Unmatched patch action: {line!r}"
                raise ValueError(msg)

            residue_type = m.group(1)
            action_type = PatchParser._parse_action_type(m.group(2))
            atom_name = m.group(3)

            kwargs = {}
            for w in PatchParser.STRING_VAR_PATTERN.finditer(m.group(4)):
                kwargs[w.group(1)] = w.group(2)
            for w in PatchParser.NUMBER_VAR_PATTERN.finditer(m.group(4)):
                kwargs[w.group(1)] = float(w.group(2))

            result.append(PatchAction(action_type, PatchSelection(residue_type, atom_name), kwargs))
        return result
