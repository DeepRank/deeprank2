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
from os.path import basename, isfile, join
import h5py
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


_log = logging.getLogger(__name__)


class Query:
    """Represents one entity of interest, like a single residue variant or a protein-protein interface.

    Query objects are used to generate graphs from structures.
    objects of this class should be created before any model is loaded

    Query objects can have target values associated with them, these will be stored with the resulting graph.
    The compute_targets function under deeprankcore.tools.target is a nice way to get started. It will output a directory that can serve
    as input for the targets argument.

    Currently, the Trainer class under deeprankcore.Trainer can work with target values, that have one of the following names:

      for classification:
       - bin_class (scalar value is expected to be either 0 or 1)
       - capri_classes (scalar integer values are expected)

      for regression (expects one scalar per graph per target):
       - irmsd
       - lrmsd
       - fnat
       - dockq

    Other target names are also allowed, but require additional settings to the Trainer object.
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


class QueryCollection:
    """
    Represents the collection of data queries. Queries can be saved as a dictionary to easily navigate through their data 
    
    """

    def __init__(self):
        self._queries = []

    def add(self, query: Query):
        """ Adds new query to the collection of all generated queries.
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

    def _process_one_query(
        self,
        prefix: str,
        feature_names: List[str],
        query: Query):

        _log.info(f'\nProcess query with process ID {os.getpid()}.')

        # because only one process may access an hdf5 file at the time:
        output_path = f"{prefix}-{os.getpid()}.hdf5"

        feature_modules = [
            importlib.import_module('deeprankcore.features.' + name) for name in feature_names]

        graph = query.build(feature_modules)

        graph.write_to_hdf5(output_path)

    def process( # pylint: disable=too-many-locals
        self, 
        prefix: Optional[str] = None,
        feature_modules: List[ModuleType] = None,
        processes: Optional[int] = None,
        combine_output: bool = True,
        ) -> List[str]:

        """
        Args:
            prefix: prefix for the output files. ./processed-queries- by default.

            feature_modules: list of features' modules used to generate features.
                Each feature's module must implement the add_features function, and
                features' modules can be found (or should be placed in case of a custom made feature)
                in deeprankcore.features folder.
                If None, all available modules in deeprankcore.features are used to generate the features. 
                Defaults to None.
            
            processes: how many processes to be run simultaneously.
                By default takes all available cpu cores.

            combine_output: boolean for combining the hdf5 files generated by the processes.
                By default, the hdf5 files generated are combined into one, and then deleted.
        """

        if processes is None:
            # returns the number of CPUs in the system
            processes = os.cpu_count()
        else:
            processes_system = os.cpu_count()
            if processes > processes_system:
                _log.warning(f'\nTried to set {processes} CPUs, but only {processes_system} are present in the system.')
                processes = processes_system

        _log.info(f'\nNumber of CPUs for processing the queries set to: {processes}.')

        if prefix is None:
            prefix = "processed-queries"
        
        if feature_modules is None:
            feature_modules = glob(join('./deeprankcore/features/', "*.py"))
            feature_names = [basename(f)[:-3] for f in feature_modules if isfile(f) and not f.endswith('__init__.py')]
        else:
            feature_names = [basename(m.__file__)[:-3] for m in feature_modules]

        _log.info('Creating pool function to process the queries...')
        pool_function = partial(self._process_one_query, prefix,
                                feature_names)

        with Pool(processes) as pool:
            _log.info('Starting pooling...\n')
            pool.map(pool_function, self.queries)

        output_paths = glob(f"{prefix}-*.hdf5")

        if combine_output:
            dupl_ids = {}
            for output_path in output_paths:
                with h5py.File(f"{prefix}.hdf5",'a') as f_dest, h5py.File(output_path,'r') as f_src:
                    for key, value in f_src.items():
                        try:
                            f_src.copy(value, f_dest)
                        except RuntimeError as e:
                            if key not in dupl_ids:
                                dupl_ids[key] = 2
                            f_src.copy(value, f_dest,name = key + "_" + str(dupl_ids[key]))
                            _log.error(e)
                            _log.info(f'{key} Group id has already been added to the file. Renaming Group as {key+"_"+str(dupl_ids)}')
                            dupl_ids[key] += 1
                os.remove(output_path)
            return glob(f"{prefix}.hdf5")

        return output_paths


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
        "residue identifier within chain"

        if self._insertion_code is not None:

            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        return f"residue-graph-{self.model_id}:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}"

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

        # select which residues will be the graph
        residues = get_surrounding_residues(structure, residue, self._radius) # pylint: disable=undefined-loop-variable

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
