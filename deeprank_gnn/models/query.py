import logging
import os
from typing import Dict, List, Iterator, Optional
import tempfile

import freesasa
import numpy
import pdb2sql
from scipy.spatial import distance_matrix

from deeprank_gnn.domain.amino_acid import *
from deeprank_gnn.domain.feature import *
from deeprank_gnn.models.graph import Graph, Edge, Node
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.tools.pdb import (get_residue_contact_pairs, get_residue_distance, get_surrounding_residues,
                                    get_structure, get_atomic_contacts, get_residue_contacts, add_hydrogens)
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.models.variant import SingleResidueVariant

_log = logging.getLogger(__name__)


class Query:
    """ Represents one entity of interest, like a single residue variant or a protein-protein interface.

        Query objects are used to generate graphs from structures.
        objects of this class should be created before any model is loaded
    """

    def __init__(self, model_id: str, targets: Dict[str, float] = None):
        """
            Args:
                model_id(str): the id of the model to load, usually a pdb accession code
                targets(dict, optional): target values associated with this query
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

    def _load_structure(self, pdb_path: str, pssm_paths: Optional[Dict[str, str]] = None):
        "A helper function, to build the structure from pdb and pssm files."

        # make a copy of the pdb, with hydrogens
        pdb_name = os.path.basename(pdb_path)
        hydrogen_pdb_file, hydrogen_pdb_path = tempfile.mkstemp(prefix="hydrogenated-", suffix=pdb_name)
        os.close(hydrogen_pdb_file)

        add_hydrogens(pdb_path, hydrogen_pdb_path)

        # read the pdb copy
        try:
            pdb = pdb2sql.pdb2sql(hydrogen_pdb_path)
        finally:
            os.remove(hydrogen_pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in pssm_paths:
                    pssm_path = pssm_paths[chain.id]

                    with open(pssm_path, 'rt') as f:
                        chain.pssm = parse_pssm(f, chain)

        return structure

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def targets(self) -> Dict[str, float]:
        return self._targets

    def __repr__(self) -> str:
        return "{}({})".format(type(self), self.get_query_id())


class SingleResidueVariantResidueQuery(Query):
    "creates a residue graph from a single residue variant in a pdb file"

    def __init__(self, pdb_path: str, chain_id: str, residue_number: int, insertion_code: str,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 pssm_paths: Optional[Dict[str, str]] = None, wildtype_conservation: Optional[float] = None,
                 variant_conservation: Optional[float] = None, radius: Optional[float] = 10.0,
                 external_distance_cutoff: Optional[float] = 4.5, targets: Optional[Dict[str, float]] = None):

        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id(str): the pdb chain identifier of the variant residue
                residue_number(int): the number of the variant residue
                insertion_code(str): the insertion code of the variant residue, set to None if not applicable
                wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid
                variant_amino_acid(deeprank amino acid object): the variant amino acid
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                wildtype_conservation(float): conservation value for the wildtype
                variant_conservation(float): conservation value for the variant
                radius(float): in Ångström, determines how many residues will be included in the graph
                external_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
                targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths
        self._wildtype_conservation = wildtype_conservation
        self._variant_conservation = variant_conservation

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

            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self) -> str:
        return "residue-graph-{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

    def build_graph(self, feature_modules: List) -> Graph:

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # find the variant residue
        variant_residue = None
        for residue in structure.get_chain(self._chain_id).residues:
            if residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break

        if variant_residue is None:
            raise ValueError("Residue not found in {}: {} {}"
                             .format(self._pdb_path, self._chain_id, self.residue_id))

        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)

        # select which residues will be the graph
        residues = get_surrounding_residues(structure, residue, self._radius)

        # build the graph
        graph = build_residue_graph(residues, self.get_query_id(), self._external_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)

        return graph


class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

    def __init__(self, pdb_path: str, chain_id: str, residue_number: int, insertion_code: str,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 pssm_paths: Optional[Dict[str, str]] = None,
                 radius: Optional[float] = 10.0,
                 external_distance_cutoff: Optional[float] = 4.5,
                 targets: Optional[Dict[str, float]] = None):
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
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self) -> str:
        return "{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            self._chain_id == other._chain_id and self.residue_id == other.residue_id and \
            self._wildtype_amino_acid == other._wildtype_amino_acid and self._variant_amino_acid == other._variant_amino_acid

    def __hash__(self) -> hash:
        return hash((self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid, self._variant_amino_acid))

    @staticmethod
    def _get_atom_node_key(atom) -> str:
        """ Pickle has problems serializing the graph when the nodes are atoms,
            so use this function to generate an unique key for the atom"""

        # This should include the model, chain, residue and atom
        return str(atom)

    def build_graph(self, feature_modules: List) -> Graph:

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # find the variant residue
        variant_residue = None
        for residue in structure.get_chain(self._chain_id).residues:
            if residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break

        if variant_residue is None:
            raise ValueError("Residue not found in {}: {} {}"
                             .format(self._pdb_path, self._chain_id, self.residue_id))

        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = []
        for residue in residues:
            if residue.amino_acid is not None:
                atoms.extend(residue.atoms)

        # build the graph
        graph = build_atomic_graph(atoms, self.get_query_id(), self._external_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)

        return graph


class ProteinProteinInterfaceAtomicQuery(Query):
    "a query that builds atom-based graphs, using the residues at a protein-protein interface"

    def __init__(self, pdb_path: str, chain_id1: str, chain_id2: str, pssm_paths: Optional[Dict[str, str]] = None,
                 interface_distance_cutoff: Optional[float] = 8.5,
                 targets: Optional[Dict[str, float]] = None):
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
        return "atom-ppi-{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the residue graph.

            Returns(deeprank graph object): the resulting graph
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(self._pdb_path, structure,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        # get all atoms in the selection
        atoms_selected = set([])
        for residue1, residue2 in interface_pairs:
            atoms_selected |= set(residue1.atoms)
            atoms_selected |= set(residue2.atoms)
        atoms_selected = list(atoms_selected)

        # find the two chains
        model = atoms_selected[0].residue.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # build the graph
        graph = build_atomic_graph(atoms_selected, self.get_query_id(), self._interface_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface"

    def __init__(self, pdb_path: str, chain_id1: str, chain_id2: str, pssm_paths: Optional[Dict[str, str]] = None,
                 interface_distance_cutoff: float = 8.5,
                 use_biopython: bool = False, targets: Optional[Dict[str, float]] = None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id1(str): the pdb chain identifier of the first protein of interest
                chain_id2(str): the pdb chain identifier of the second protein of interest
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
                use_biopython(bool): whether or not to use biopython tools
                targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff

        self._use_biopython = use_biopython

    def get_query_id(self) -> str:
        return "residue-ppi-{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the residue graph.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(self._pdb_path, structure,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)

        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        residues_selected = set([])
        for residue1, residue2 in interface_pairs:
            residues_selected.add(residue1)
            residues_selected.add(residue2)
        residues_selected = list(residues_selected)

        # find the two chains
        model = residues_selected[0].chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # build the graph
        graph = build_residue_graph(residues_selected, self.get_query_id(), self._interface_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph


class QueryDataset:
    "represents a collection of data queries"

    def __init__(self):
        self._queries = []

    def add(self, query: Query):
        self._queries.append(query)

    @property
    def queries(self) -> List[Query]:
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)
