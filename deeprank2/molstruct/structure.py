from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from deeprank2.molstruct.atom import Atom
    from deeprank2.molstruct.residue import Residue
    from deeprank2.utils.pssmdata import PssmRow


class PDBStructure:
    """."""

    def __init__(self, id_: str | None = None):
        """A proitein or protein complex structure.

        A `PDBStructure` can contain one or multiple `Chains`, i.e. separate molecular entities (individual proteins).
        One PDBStructure consists of a number of `Residue`s, each of which is of a particular `AminoAcid` type and in turn consists of a number of `Atom`s.

        Args:
            id_: An unique identifier for this structure, can be the pdb accession code. Defaults to None.
        """
        self._id = id_
        self._chains = {}

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, PDBStructure):
            return self._id == other._id
        return NotImplemented

    def __hash__(self) -> hash:
        return hash(self._id)

    def __repr__(self) -> str:
        return self._id

    def has_chain(self, chain_id: str) -> bool:
        return chain_id in self._chains

    def get_chain(self, chain_id: str) -> Chain:
        return self._chains[chain_id]

    def add_chain(self, chain: Chain) -> None:
        if chain.id in self._chains:
            msg = f"Duplicate chain: {chain.id}"
            raise ValueError(msg)
        self._chains[chain.id] = chain

    @property
    def chains(self) -> list[Chain]:
        return list(self._chains.values())

    def get_atoms(self) -> list[Atom]:
        """List all atoms in the structure."""
        atoms = []
        for chain in self._chains.values():
            for residue in chain.residues:
                atoms.extend(residue.atoms)
        return atoms

    @property
    def id(self) -> str:
        return self._id


class Chain:
    """One independent molecular entity of a `PDBStructure`.

    In other words: each `Chain` in a `PDBStructure` is a separate molecule.
    """

    def __init__(self, model: PDBStructure, id_: str | None):
        """One chain of a PDBStructure.

        Args:
        model: The model that this chain is part of.
        id_: The pdb identifier of this chain.
        """
        self._model = model
        self._id = id_
        self._residues = {}
        self._pssm = None  # pssm is per chain

    @property
    def model(self) -> PDBStructure:
        return self._model

    @property
    def pssm(self) -> PssmRow:
        return self._pssm

    @pssm.setter
    def pssm(self, pssm: PssmRow) -> None:
        self._pssm = pssm

    def add_residue(self, residue: Residue) -> None:
        self._residues[(residue.number, residue.insertion_code)] = residue

    def has_residue(self, residue_number: int, insertion_code: str | None = None) -> bool:
        return (residue_number, insertion_code) in self._residues

    def get_residue(self, residue_number: int, insertion_code: str | None = None) -> Residue:
        return self._residues[(residue_number, insertion_code)]

    @property
    def id(self) -> str:
        return self._id

    @property
    def residues(self) -> list[Residue]:
        return list(self._residues.values())

    def get_atoms(self) -> list[Atom]:
        """Shortcut to list all atoms in this chain."""
        atoms = []
        for residue in self.residues:
            atoms.extend(residue.atoms)

        return atoms

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Chain):
            return self._model == other._model and self._id == other._id
        return NotImplemented

    def __hash__(self) -> hash:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"{self._model} {self._id}"
