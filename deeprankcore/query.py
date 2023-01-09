import logging
import os
from typing import Dict, List, Optional, Iterator, Union
import tempfile
import pdb2sql
import pickle
from glob import glob
from types import ModuleType
from functools import partial
from multiprocessing import Pool
import importlib
from os.path import basename
import h5py
import pkgutil
from deeprankcore.utils.graph import Graph
from deeprankcore.molstruct.aminoacid import AminoAcid
from deeprankcore.utils.buildgraph import (
    get_residue_contact_pairs,
    get_surrounding_residues,
    get_structure,
    add_hydrogens,
)
from deeprankcore.utils.parsing.pssm import parse_pssm
from deeprankcore.utils.graph import build_residue_graph, build_atomic_graph
from deeprankcore.molstruct.variant import SingleResidueVariant
import deeprankcore.features


_log = logging.getLogger(__name__)


class Query():

    def __init__(self, model_id: str, targets: Optional[Dict[str, Union[float, int]]] = None):
        """
        Represents one entity of interest, like a single residue variant or a protein-protein interface.
        :class:`Query` objects are used to generate graphs from structures, and they should be created before any model is loaded. They can have target values associated with them, which will be stored with the resulting graph.

        Args:
            model_id(str): the id of the model to load, usually a pdb accession code.
            targets(Dict[str, Union[float, int]], optional): target values associated with the query, defaults to None.
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

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    @property
    def targets(self) -> Dict[str, float]:
        return self._targets

    def __repr__(self) -> str:
        return f"{type(self)}({self.get_query_id()})"


class QueryCollection:
    """
    Represents the collection of data queries. Queries can be saved as a dictionary to easily navigate through their data.
    
    """

    def __init__(self):
        self._queries = []
        self.cpu_count = None
        self.ids_count = {}

    def add(self, query: Query, verbose: bool = False):

        """
        Adds a new query to the collection.

        Args:
            query(:class:`Query`): must be a :class:`Query` object, either :class:`ProteinProteinInterfaceResidueQuery` or :class:`SingleResidueVariantAtomicQuery`.
            verbose(bool, optional): for logging query ids added, defaults to False.
        """
        query_id = query.get_query_id()

        if verbose:
            _log.info(f'Adding query with ID {query_id}.')

        if query_id not in self.ids_count:
            self.ids_count[query_id] = 1
        else:
            self.ids_count[query_id] += 1
            new_id = query.model_id + "_" + str(self.ids_count[query_id])
            query.model_id = new_id
            _log.warning(f'Query with id {query_id} has already been added to the collection. Renaming it as {query.get_query_id()}')

        self._queries.append(query)

    def export_dict(self, dataset_path: str):
        """ Exports the colection of all queries to a dictionary file.

            Args:
                dataset_path(str): the path where to save the list of queries.
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

    def _process_one_query(
        self,
        prefix: str,
        feature_names: List[str],
        verbose: bool,
        query: Query):

        if verbose:
            _log.info(f'\nProcess query with process ID {os.getpid()}.')

        # because only one process may access an hdf5 file at the time:
        output_path = f"{prefix}-{os.getpid()}.hdf5"

        feature_modules = [
            importlib.import_module('deeprankcore.features.' + name) for name in feature_names]

        try:
            graph = query.build(feature_modules)
            graph.write_to_hdf5(output_path)
        except ValueError as e:
            _log.error(e)
            _log.warning(f'Query {query.get_query_id()}\'s graph was not saved in the hdf5 file; check the query\'s files')

    def process( # pylint: disable=too-many-arguments
        self, 
        prefix: Optional[str] = None,
        feature_modules: List[ModuleType] = None,
        cpu_count: Optional[int] = None,
        combine_output: bool = True,
        verbose: bool = False
        ) -> List[str]:

        """
        Args:
            prefix(str, optional): prefix for the output files. Defaults to None, which sets ./processed-queries- prefix.

            feature_modules(List[ModuleType], optional): list of features' modules used to generate features.
            Each feature's module must implement the add_features function, and features' modules can be found (or should be placed in case of a custom made feature) in `deeprankcore.features` folder. Defaults to None, which means that all available modules in `deeprankcore.features` are used to generate the features. 
            
            cpu_count(int, optional): how many processes to be run simultaneously. Defaults to None, which takes all available cpu cores.

            combine_output(bool, optional): for combining the hdf5 files generated by the processes, defaults to True.

            verbose(bool, optional): for logging query ids processed, defaults to False.
        
        Returns:
            output_paths(List(str)): the list of paths of the generated .hdf5 files.
        """

        if cpu_count is None:
            # returns the number of CPUs in the system
            cpu_count = os.cpu_count()
        else:
            cpu_count_system = os.cpu_count()
            if cpu_count > cpu_count_system:
                _log.warning(f'\nTried to set {cpu_count} CPUs, but only {cpu_count_system} are present in the system.')
                cpu_count = cpu_count_system
        
        self.cpu_count = cpu_count

        _log.info(f'\nNumber of CPUs for processing the queries set to: {self.cpu_count}.')

        if prefix is None:
            prefix = "processed-queries"
        
        if feature_modules is None:
            feature_names = [modname for _, modname, _ in pkgutil.iter_modules(deeprankcore.features.__path__)]
        else:
            feature_names = [basename(m.__file__)[:-3] for m in feature_modules]

        _log.info('Creating pool function to process the queries...')
        pool_function = partial(self._process_one_query, prefix,
                                feature_names, verbose)

        with Pool(self.cpu_count) as pool:
            _log.info('Starting pooling...\n')
            pool.map(pool_function, self.queries)

        output_paths = glob(f"{prefix}-*.hdf5")

        if combine_output:
            for output_path in output_paths:
                with h5py.File(f"{prefix}.hdf5",'a') as f_dest, h5py.File(output_path,'r') as f_src:
                    for _, value in f_src.items():
                        f_src.copy(value, f_dest)
                os.remove(output_path)
            return glob(f"{prefix}.hdf5")

        return output_paths


class SingleResidueVariantResidueQuery(Query):

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
        distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        "Creates a residue graph from a single residue variant in a .pdb file."

        Args:
            pdb_path(str): the path to the pdb file.
            chain_id(str): the pdb chain identifier of the variant residue.
            residue_number(int): the number of the variant residue.
            insertion_code(str): the insertion code of the variant residue, set to None if not applicable.
            wildtype_amino_acid(:class:`AminoAcid`): the wildtype amino acid.
            variant_amino_acid(:class:`AminoAcid`): the variant amino acid.
            pssm_paths(Dict(str,str), optional): the paths to the pssm files, per chain identifier.
            radius(float, optional): in Ångström, determines how many residues will be included in the graph.
            distance_cutoff(float, optional): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph.
            targets(Dict(str,float), optional): named target values associated with this query.
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
        self._distance_cutoff = distance_cutoff

    @property
    def residue_id(self) -> str:
        "Residue identifier within the chain."

        if self._insertion_code is not None:

            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        "Returns the complete query id."

        return f"residue-graph-{self.model_id}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

    def build(self, feature_modules: List[ModuleType], include_hydrogens: bool = False) -> Graph:
        """
        Builds the graph from the pdb structure.

        Args:
            feature_modules(List[ModuleType]): each must implement the add_features function.
        
        Returns:
            graph(:class:`Graph`): The resulting :class:`Graph` object with all the features and targets. 
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
        residues = list(get_surrounding_residues(structure, residue, self._radius)) # pylint: disable=undefined-loop-variable

        # build the graph
        graph = build_residue_graph(
            residues, self.get_query_id(), self._distance_cutoff
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
        distance_cutoff: Optional[float] = 4.5,
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
            distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
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

        self._distance_cutoff = distance_cutoff

    @property
    def residue_id(self) -> str:
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        return f"{self.model_id,}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

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

    def build(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
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
            atoms, self.get_query_id(), self._distance_cutoff
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
        distance_cutoff: Optional[float] = 5.5,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id1(str): the pdb chain identifier of the first protein of interest
            chain_id2(str): the pdb chain identifier of the second protein of interest
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            distance_cutoff(float): max distance in Ångström between two interacting atoms of the two proteins
            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._distance_cutoff = distance_cutoff

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

    def build(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
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
            self._distance_cutoff,
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
            atoms_selected, self.get_query_id(), self._distance_cutoff
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
        distance_cutoff: float = 10,
        targets: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            pdb_path(str): the path to the pdb file
            chain_id1(str): the pdb chain identifier of the first protein of interest
            chain_id2(str): the pdb chain identifier of the second protein of interest
            pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
            distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
            targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._distance_cutoff = distance_cutoff

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

    def build(self, feature_modules: List, include_hydrogens: bool = False) -> Graph:
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
            self._distance_cutoff,
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
            residues_selected, self.get_query_id(), self._distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph
