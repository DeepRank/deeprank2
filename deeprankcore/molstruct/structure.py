from typing import Optional
from deeprankcore.utils.pssmdata import PssmRow


class PDBStructure:
    "represents one entire pdb structure"

    def __init__(self, id_: Optional[str] = None):
        """
        Args:
            id_(str): an unique identifier for this structure, can be the pdb accession code.
        """
        self._id = id_
        self._chains = {}

    def __eq__(self, other) -> bool:
        return isinstance(self, type(other)) and self._id == other._id

    def __hash__(self) -> hash:
        return hash(self._id)

    def __repr__(self) -> str:
        return self._id

    def has_chain(self, chain_id: str) -> bool:
        return chain_id in self._chains

    def get_chain(self, chain_id: str):
        return self._chains[chain_id]

    def add_chain(self, chain):
        if chain.id in self._chains:
            raise ValueError(f"duplicate chain: {chain.id}")

        self._chains[chain.id] = chain

    @property
    def chains(self):
        return list(self._chains.values())

    def get_atoms(self):
        "shortcut to list all atoms in this structure"
        atoms = []
        for chain in self._chains.values():
            for residue in chain.residues:
                atoms.extend(residue.atoms)

        return atoms

    @property
    def id(self) -> str:
        return self._id


class Chain:
    "represents one pdb chain"

    def __init__(self, model: PDBStructure, id_: Optional[str]):
        """
        Args:
            model(deeprank structure object): the model that this chain is part of
            id_(str): the pdb identifier of this chain
        """

        self._model = model
        self._id = id_
        self._residues = {}
        self._pssm = None  # pssm is per chain

    @property
    def model(self):
        return self._model

    @property
    def pssm(self) -> PssmRow:
        return self._pssm

    @pssm.setter
    def pssm(self, pssm: PssmRow):
        self._pssm = pssm

    def add_residue(self, residue):
        self._residues[(residue.number, residue.insertion_code)] = residue

    def has_residue(self, residue_number: int, insertion_code: str) -> bool:
        return (residue_number, insertion_code) in self._residues

    def get_residue(self, residue_number: int, insertion_code: str):
        return self._residues[(residue_number, insertion_code)]

    @property
    def id(self) -> str:
        return self._id

    @property
    def residues(self):
        return list(self._residues.values())

    def get_atoms(self):
        "shortcut to list all atoms in this chain"
        atoms = []
        for residue in self._residues.values():
            atoms.extend(residue.atoms)

        return atoms

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self._model == other._model
            and self._id == other._id
        )

    def __hash__(self) -> hash:
        return hash((self._model, self._id))

    def __repr__(self) -> str:
        return f"{self._model} {self._id}"
