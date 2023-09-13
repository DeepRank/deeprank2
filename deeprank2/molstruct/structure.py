from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from deeprank2.utils.pssmdata import PssmRow

if TYPE_CHECKING:
    from deeprank2.molstruct.atom import Atom
    from deeprank2.molstruct.residue import Residue


class PDBStructure:
    """A proitein or protein complex structure..

    A `PDBStructure` can contain one or multiple `Chains`, i.e. separate
    molecular entities (individual proteins).
    One PDBStructure consists of a number of `Residue`s, each of which is of a
    particular `AminoAcid` type and in turn consists of a number of `Atom`s.
    """

    def __init__(self, id_: Optional[str] = None):
        """
        Args:
            id_ (str, optional): An unique identifier for this structure, can be the pdb accession code.
                Defaults to None.
        """
        self._id = id_
        self._chains = {}

    def __eq__(self, other) -> bool:
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

    def add_chain(self, chain: Chain):
        if chain.id in self._chains:
            raise ValueError(f"duplicate chain: {chain.id}")
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

    def __init__(self, model: PDBStructure, id_: Optional[str]):
        """One chain of a PDBStructure.

            Args:
            model (:class:`PDBStructure`): The model that this chain is part of.
            id_ (str): The pdb identifier of this chain.
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
    def pssm(self, pssm: PssmRow):
        self._pssm = pssm

    def add_residue(self, residue: Residue):
        self._residues[(residue.number, residue.insertion_code)] = residue

    def has_residue(self, residue_number: int, insertion_code: Optional[str] = None) -> bool:
        return (residue_number, insertion_code) in self._residues

    def get_residue(self, residue_number: int, insertion_code: Optional[str] = None) -> Residue:
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

    def __eq__(self, other) -> bool:
        if isinstance(other, Chain):
            return (self._model == other._model
                    and self._id == other._id)
        return NotImplemented

    def __hash__(self) -> hash:
        return hash(self._id)

    def __repr__(self) -> str:
        return f"{self._model} {self._id}"
