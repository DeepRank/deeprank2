import logging
import os
from typing import Dict, List, Optional, Iterator, Union
import tempfile
import pdb2sql
from deeprankcore.models.graph import Graph
from deeprankcore.models.amino_acid import AminoAcid
from deeprankcore.tools.pdb import (
    get_residue_contact_pairs,
    get_surrounding_residues,
    get_structure,
    add_hydrogens,
)
from deeprankcore.tools.pssm import parse_pssm
from deeprankcore.tools.graph import build_residue_graph, build_atomic_graph
from deeprankcore.models.variant import SingleResidueVariant
import pickle

_log = logging.getLogger(__name__)


class Query:
    """Represents one entity of interest, like a single residue variant or a protein-protein interface.

    Query objects are used to generate graphs from structures.
    objects of this class should be created before any model is loaded

    Query objects can have target values associated with them, these will be stored with the resulting graph.
    The get_all_scores function under deeprankcore.tools.score is a nice way to get started. It will output a directory that can serve
    as input for the targets argument.

    Currently, the NeuralNet class under deeprankcore.NeuralNet can work with target values, that have one of the following names:

      for classification:
       - bin_class (scalar value is expected to be either 0 or 1)
       - capri_classes (scalar integer values are expected)

      for regression (expects one scalar per graph per target):
       - irmsd
       - lrmsd
       - fnat
       - dockQ

    Other target names are also allowed, but require additional settings to the NeuralNet object.
    """

    def __init__(self, model_id: str, targets: Optional[Dict[str, Union[float, int]]] = None):
        """
        Args:
            model_id: the id of the model to load, usually a pdb accession code
            targets: target values associated with this query
            pssm_paths: the paths of the pssm files, per protein(chain) id
        """

        self._model_id = model_id

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    def _set_graph_targets(self, graph: Graph):
        "simply copies target data from query to graph"

        for target_name, target_data in self._targets.items():
            graph.targets[target_name] = target_data

    def _load_structure(
        self, pdb_path: str, pssm_paths: Optional[Dict[str, str]],
        include_hydrogens: bool
    ):
        "A helper function, to build the structure from pdb and pssm files."

        # make a copy of the pdb, with hydrogens
        pdb_name = os.path.basename(pdb_path)
        hydrogen_pdb_file, hydrogen_pdb_path = tempfile.mkstemp(
            prefix="hydrogenated-", suffix=pdb_name
        )
        os.close(hydrogen_pdb_file)

        if include_hydrogens:
            add_hydrogens(pdb_path, hydrogen_pdb_path)

            # read the pdb copy
            try:
                pdb = pdb2sql.pdb2sql(hydrogen_pdb_path)
            finally:
                os.remove(hydrogen_pdb_path)
        else:
            pdb = pdb2sql.pdb2sql(pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close() # pylint: disable=protected-access

        # read the pssm
        if pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in pssm_paths:
                    pssm_path = pssm_paths[chain.id]

                    with open(pssm_path, "rt", encoding="utf-8") as f:
                        chain.pssm = parse_pssm(f, chain)

        return structure

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def targets(self) -> Dict[str, float]:
        return self._targets

    def __repr__(self) -> str:
        return f"{type(self)}({self.get_query_id()})"


class SingleResidueVariantResidueQuery(Query):
    "creates a residue graph from a single residue variant in a pdb file"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id: str,
        residue_number: int,
        insertion_code: str,
        wildtype_amino_acid: AminoAcid,
        variant_amino_acid: AminoAcid,
        pssm_paths: Optional[Dict[str, str]] = None,
        radius: Optional[float] = 10.0,
        external_distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id(str): the pdb chain identifier of the variant residue
            residue_number(int): the number of the variant residue
            insertion_code(str): the insertion code of the variant residue, set to None if not applicable
            wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid
            variant_amino_acid(deeprank amino acid object): the variant amino acid
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            radius(float): in Ångström, determines how many residues will be included in the graph
            external_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
            targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius
        self._external_distance_cutoff = external_distance_cutoff

    @property
    def residue_id(self) -> str:
        "residue identifier within chain"

        if self._insertion_code is not None:

            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        return f"residue-graph-{self.model_id}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

    def build_graph(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the pdb structure.
        Args:
            feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens)

        # find the variant residue
        variant_residue = None
        for residue in structure.get_chain(self._chain_id).residues:
            if (
                residue.number == self._residue_number
                and residue.insertion_code == self._insertion_code
            ):
                variant_residue = residue
                break

        if variant_residue is None:
            raise ValueError(
                "Residue not found in {self._pdb_path}: {self._chain_id} {self.residue_id}"
            )

        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)

        # select which residues will be the graph
        residues = get_surrounding_residues(structure, residue, self._radius) # pylint: disable=undefined-loop-variable

        # build the graph
        graph = build_residue_graph(
            residues, self.get_query_id(), self._external_distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)

        return graph


class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id: str,
        residue_number: int,
        insertion_code: str,
        wildtype_amino_acid: AminoAcid,
        variant_amino_acid: AminoAcid,
        pssm_paths: Optional[Dict[str, str]] = None,
        radius: Optional[float] = 10.0,
        external_distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id(str): the pdb chain identifier of the variant residue
            residue_number(int): the number of the variant residue
            insertion_code(str): the insertion code of the variant residue, set to None if not applicable
            wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid
            variant_amino_acid(deeprank amino acid object): the variant amino acid
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            radius(float): in Ångström, determines how many residues will be included in the graph
            external_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
            internal_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an internal edge in the graph
            (must be shorter than external)
            targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius

        self._external_distance_cutoff = external_distance_cutoff

    @property
    def residue_id(self) -> str:
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        return "{self.model_id,}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and self._chain_id == other._chain_id
            and self.residue_id == other.residue_id
            and self._wildtype_amino_acid == other._wildtype_amino_acid
            and self._variant_amino_acid == other._variant_amino_acid
        )

    def __hash__(self) -> hash:
        return hash(
            (
                self.model_id,
                self._chain_id,
                self.residue_id,
                self._wildtype_amino_acid,
                self._variant_amino_acid,
            )
        )

    @staticmethod
    def _get_atom_node_key(atom) -> str:
        """Pickle has problems serializing the graph when the nodes are atoms,
        so use this function to generate an unique key for the atom"""

        # This should include the model, chain, residue and atom
        return str(atom)

    def build_graph(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the pdb structure.
        Args:
            feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens)

        # find the variant residue
        variant_residue = None
        for residue in structure.get_chain(self._chain_id).residues:
            if (
                residue.number == self._residue_number
                and residue.insertion_code == self._insertion_code
            ):
                variant_residue = residue
                break

        if variant_residue is None:
            raise ValueError(
                "Residue not found in {self._pdb_path}: {self._chain_id} {self.residue_id}"
            )

        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = set([])
        for residue in residues:
            if residue.amino_acid is not None:
                for atom in residue.atoms:
                    atoms.add(atom)
        atoms = list(atoms)

        # build the graph
        graph = build_atomic_graph(
            atoms, self.get_query_id(), self._external_distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)

        return graph


class ProteinProteinInterfaceAtomicQuery(Query):
    "a query that builds atom-based graphs, using the residues at a protein-protein interface"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        interface_distance_cutoff: Optional[float] = 8.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id1(str): the pdb chain identifier of the first protein of interest
            chain_id2(str): the pdb chain identifier of the second protein of interest
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff

    def get_query_id(self) -> str:
        return f"atom-ppi-{self.model_id}:{self._chain_id1}-{self._chain_id2}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the pdb structure.
        Args:
            feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(
            self._pdb_path,
            structure,
            self._chain_id1,
            self._chain_id2,
            self._interface_distance_cutoff,
        )
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        atoms_selected = set([])
        for residue1, residue2 in interface_pairs:
            for atom in residue1.atoms + residue2.atoms:
                atoms_selected.add(atom)
        atoms_selected = list(atoms_selected)

        # build the graph
        graph = build_atomic_graph(
            atoms_selected, self.get_query_id(), self._interface_distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        interface_distance_cutoff: float = 8.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id1(str): the pdb chain identifier of the first protein of interest
            chain_id2(str): the pdb chain identifier of the second protein of interest
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff

    def get_query_id(self) -> str:
        return f"residue-ppi-{self.model_id}:{self._chain_id1}-{self._chain_id2}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the pdb structure.
        Args:
            feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(
            self._pdb_path,
            structure,
            self._chain_id1,
            self._chain_id2,
            self._interface_distance_cutoff,
        )

        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        residues_selected = set([])
        for residue1, residue2 in interface_pairs:
            residues_selected.add(residue1)
            residues_selected.add(residue2)
        residues_selected = list(residues_selected)

        # build the graph
        graph = build_residue_graph(
            residues_selected, self.get_query_id(), self._interface_distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph


class QueryDataset:
    """
    Represents the collection of data queries. Queries can be saved as a dictionary to easily navigate through their data 
    
    """

    def __init__(self):
        self._queries = []

    def add(self, query: Query):
        """ Adds new query to the colection of all generated queries.
            Args:
                query (Query): must be a Query object, either ProteinProteinInterfaceResidueQuery or SingleResidueVariantAtomicQuery.
        """
        self._queries.append(query)

    def export_dict(self, dataset_path: str):
        """ Exports the colection of all queries to a dictionary file
            Args:
                dataset_path (str): the new path where the list of queries be saved to.
        """
        with open(dataset_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file)    
            
    @property
    def queries(self) -> List[Query]:
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)
