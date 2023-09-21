import importlib
import logging
import os
import pickle
import pkgutil
import tempfile
import warnings
from functools import partial
from glob import glob
from multiprocessing import Pool
from random import randrange
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Union

import h5py
import numpy as np
import pdb2sql

import deeprank2.features
from deeprank2.domain.aminoacidlist import convert_aa_nomenclature
from deeprank2.features import components, conservation, contact
from deeprank2.molstruct.aminoacid import AminoAcid
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import SingleResidueVariant
from deeprank2.molstruct.structure import PDBStructure
from deeprank2.utils.buildgraph import (add_hydrogens, get_contact_atoms,
                                        get_structure,
                                        get_surrounding_residues)
from deeprank2.utils.graph import (Graph, build_atomic_graph,
                                   build_residue_graph)
from deeprank2.utils.grid import Augmentation, GridSettings, MapMethod
from deeprank2.utils.parsing.pssm import parse_pssm

_log = logging.getLogger(__name__)


def _check_pssm(pdb_path: str, pssm_paths: Dict[str, str], suppress: bool, verbosity: int = 0):
    """Checks whether information stored in pssm file matches the corresponding pdb file.

    Args:
        pdb_path (str): Path to the PDB file.
        pssm_paths (Dict[str, str]): The paths to the PSSM files, per chain identifier.
        suppress (bool): Suppress errors and throw warnings instead.
        verbosity (int): Level of verbosity of error/warning. Defaults to 0.
            0 (low): Only state file name where error occurred;
            1 (medium): Also state number of incorrect and missing residues;
            2 (high): Also list the incorrect residues

    Raises:
        ValueError: Raised if info between pdb file and pssm file doesn't match or if no pssms were provided
    """

    if not pssm_paths:
        raise ValueError('No pssm paths provided for conservation feature module.')

    pssm_data = {}
    for chain in pssm_paths:
        with open(pssm_paths[chain], encoding='utf-8') as f:
            lines = f.readlines()[1:]
        for line in lines:
            pssm_data[chain + line.split()[0].zfill(4)] = convert_aa_nomenclature(line.split()[1], 3)

    # load ground truth from pdb file
    pdb_truth = pdb2sql.pdb2sql(pdb_path).get_residues()
    pdb_truth = {res[0] + str(res[2]).zfill(4): res[1] for res in pdb_truth if res[0] in pssm_paths}

    wrong_list = []
    missing_list = []

    for residue in pdb_truth:
        try:
            if pdb_truth[residue] != pssm_data[residue]:
                wrong_list.append(residue)
        except KeyError:
            missing_list.append(residue)

    if len(wrong_list) + len(missing_list) > 0:
        error_message = f'Amino acids in PSSM files do not match pdb file for {os.path.split(pdb_path)[1]}.'
        if verbosity:
            if len(wrong_list) > 0:
                error_message = error_message + f'\n\t{len(wrong_list)} entries are incorrect.'
                if verbosity == 2:
                    error_message = error_message[-1] + f':\n\t{missing_list}'
            if len(missing_list) > 0:
                error_message = error_message + f'\n\t{len(missing_list)} entries are missing.'
                if verbosity == 2:
                    error_message = error_message[-1] + f':\n\t{missing_list}'

        if not suppress:
            raise ValueError(error_message)

        warnings.warn(error_message)
        _log.warning(error_message)


class Query:

    def __init__(self, model_id: str, targets: Optional[Dict[str, Union[float, int]]] = None, suppress_pssm_errors: bool = False):
        """Represents one entity of interest, like a single-residue variant or a protein-protein interface.

        :class:`Query` objects are used to generate graphs from structures, and they should be created before any model is loaded.
        They can have target values associated with them, which will be stored with the resulting graph.

        Args:
            model_id (str): The ID of the model to load, usually a .PDB accession code.
            targets (Optional[Dict[str, Union[float, int]]], optional): Target values associated with the query. Defaults to None.
            suppress_pssm_errors (bool, optional): Suppress error raised if .pssm files do not match .pdb files and throw warning instead.
                Defaults to False.
        """

        self._model_id = model_id
        self._suppress = suppress_pssm_errors

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    def _set_graph_targets(self, graph: Graph):
        "Simply copies target data from query to graph."

        for target_name, target_data in self._targets.items():
            graph.targets[target_name] = target_data

    def _load_structure(
        self, pdb_path: str, pssm_paths: Optional[Dict[str, str]],
        include_hydrogens: bool,
        load_pssms: bool,
    ):
        "A helper function, to build the structure from .PDB and .PSSM files."

        # make a copy of the pdb, with hydrogens
        pdb_name = os.path.basename(pdb_path)
        hydrogen_pdb_file, hydrogen_pdb_path = tempfile.mkstemp(
            prefix="hydrogenated-", suffix=pdb_name
        )
        os.close(hydrogen_pdb_file)

        if include_hydrogens:
            add_hydrogens(pdb_path, hydrogen_pdb_path)

            # read the .PDB copy
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
        if load_pssms:
            _check_pssm(pdb_path, pssm_paths, suppress = self._suppress)
            for chain in structure.chains:
                if chain.id in pssm_paths:
                    pssm_path = pssm_paths[chain.id]

                    with open(pssm_path, "rt", encoding="utf-8") as f:
                        chain.pssm = parse_pssm(f, chain)

        return structure

    @property
    def model_id(self) -> str:
        "The ID of the model, usually a .PDB accession code."
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    @property
    def targets(self) -> Dict[str, float]:
        "The target values associated with the query."
        return self._targets

    def __repr__(self) -> str:
        return f"{type(self)}({self.get_query_id()})"

    def build(self, feature_modules: List[ModuleType], include_hydrogens: bool = False) -> Graph:
        raise NotImplementedError("Must be defined in child classes.")
    def get_query_id(self) -> str:
        raise NotImplementedError("Must be defined in child classes.")


class QueryCollection:
    """
    Represents the collection of data queries.
        Queries can be saved as a dictionary to easily navigate through their data.

    """

    def __init__(self):

        self._queries = []
        self.cpu_count = None
        self.ids_count = {}

    def add(self, query: Query, verbose: bool = False, warn_duplicate: bool = True):
        """
        Adds a new query to the collection.

        Args:
            query(:class:`Query`): Must be a :class:`Query` object, either :class:`ProteinProteinInterfaceResidueQuery` or
                :class:`SingleResidueVariantAtomicQuery`.
            verbose(bool, optional): For logging query IDs added, defaults to False.
            warn_duplicate (bool): Log a warning before renaming if a duplicate query is identified.

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

            if warn_duplicate:
                _log.warning(f'Query with ID {query_id} has already been added to the collection. Renaming it as {query.get_query_id()}')

        self._queries.append(query)

    def export_dict(self, dataset_path: str):
        """Exports the colection of all queries to a dictionary file.

        Args:
            dataset_path (str): The path where to save the list of queries.
        """
        with open(dataset_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file)

    @property
    def queries(self) -> List[Query]:
        "The list of queries added to the collection."
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)

    def __len__(self) -> int:
        return len(self._queries)

    def _process_one_query(  # pylint: disable=too-many-arguments
        self,
        prefix: str,
        feature_names: List[str],
        grid_settings: Optional[GridSettings],
        grid_map_method: Optional[MapMethod],
        grid_augmentation_count: int,
        query: Query
    ):

        try:
            # because only one process may access an hdf5 file at a time:
            output_path = f"{prefix}-{os.getpid()}.hdf5"

            feature_modules = [
                importlib.import_module('deeprank2.features.' + name) for name in feature_names]

            graph = query.build(feature_modules)
            graph.write_to_hdf5(output_path)

            if grid_settings is not None and grid_map_method is not None:
                graph.write_as_grid_to_hdf5(output_path, grid_settings, grid_map_method)

                for _ in range(grid_augmentation_count):
                    # repeat with random augmentation
                    axis, angle = pdb2sql.transform.get_rot_axis_angle(randrange(100))
                    augmentation = Augmentation(axis, angle)
                    graph.write_as_grid_to_hdf5(output_path, grid_settings, grid_map_method, augmentation)

            return None

        except (ValueError, AttributeError, KeyError, TimeoutError) as e:
            _log.warning(f'\nGraph/Query with ID {query.get_query_id()} ran into an Exception ({e.__class__.__name__}: {e}),'
            ' and it has not been written to the hdf5 file. More details below:')
            _log.exception(e)
            return None

    def process( # pylint: disable=too-many-arguments, too-many-locals, dangerous-default-value
        self,
        prefix: Optional[str] = None,
        feature_modules: Union[ModuleType, List[ModuleType], str, List[str]] = [components, contact],
        cpu_count: Optional[int] = None,
        combine_output: bool = True,
        grid_settings: Optional[GridSettings] = None,
        grid_map_method: Optional[MapMethod] = None,
        grid_augmentation_count: int = 0
    ) -> List[str]:
        """
        Args:
            prefix (Optional[str], optional): Prefix for the output files. Defaults to None, which sets ./processed-queries- prefix.
            feature_modules (Union[ModuleType, List[ModuleType], str, List[str]], optional): Features' module or list of features' modules
                used to generate features (given as string or as an imported module). Each module must implement the :py:func:`add_features` function,
                and features' modules can be found (or should be placed in case of a custom made feature) in `deeprank2.features` folder.
                If set to 'all', all available modules in `deeprank2.features` are used to generate the features.
                Defaults to only the basic feature modules `deeprank2.features.components` and `deeprank2.features.contact`.
            cpu_count (Optional[int], optional): How many processes to be run simultaneously. Defaults to None, which takes all available cpu cores.
            combine_output (bool, optional): For combining the HDF5 files generated by the processes. Defaults to True.
            grid_settings (Optional[:class:`GridSettings`], optional): If valid together with `grid_map_method`, the grid data will be stored as well.
                Defaults to None.
            grid_map_method (Optional[:class:`MapMethod`], optional): If valid together with `grid_settings`, the grid data will be stored as well.
                Defaults to None.
            grid_augmentation_count (int, optional): Number of grid data augmentations. May not be negative be zero or a positive number.
                Defaults to 0.

        Returns:
            List[str]: The list of paths of the generated HDF5 files.
        """

        # set defaults
        if prefix is None:
            prefix = "processed-queries"
        elif prefix.endswith('.hdf5'):
            prefix = prefix[:-5]
        if cpu_count is None:
            cpu_count = os.cpu_count()  # returns the number of CPUs in the system
        else:
            cpu_count_system = os.cpu_count()
            if cpu_count > cpu_count_system:
                _log.warning(f'\nTried to set {cpu_count} CPUs, but only {cpu_count_system} are present in the system.')
                cpu_count = cpu_count_system
        self.cpu_count = cpu_count
        _log.info(f'\nNumber of CPUs for processing the queries set to: {self.cpu_count}.')


        if feature_modules == 'all':
            feature_names = [modname for _, modname, _ in pkgutil.iter_modules(deeprank2.features.__path__)]
        elif isinstance(feature_modules, list):
            feature_names = [os.path.basename(m.__file__)[:-3] if isinstance(m,ModuleType)
                             else m.replace('.py','') for m in feature_modules]
        elif isinstance(feature_modules, ModuleType):
            feature_names = [os.path.basename(feature_modules.__file__)[:-3]]
        elif isinstance(feature_modules, str):
            feature_names = [feature_modules.replace('.py','')]
        else:
            raise ValueError(f'Feature_modules has received an invalid input type: {type(feature_modules)}.')
        _log.info(f'\nSelected feature modules: {feature_names}.')

        _log.info(f'Creating pool function to process {len(self.queries)} queries...')
        pool_function = partial(self._process_one_query, prefix,
                                feature_names,
                                grid_settings, grid_map_method, grid_augmentation_count)

        with Pool(self.cpu_count) as pool:
            _log.info('Starting pooling...\n')
            pool.map(pool_function, self.queries)

        output_paths = glob(f"{prefix}-*.hdf5")

        if combine_output:
            for output_path in output_paths:
                with h5py.File(f"{prefix}.hdf5",'a') as f_dest, h5py.File(output_path,'r') as f_src:
                    for key, value in f_src.items():
                        _log.debug(f"copy {key} from {output_path} to {prefix}.hdf5")
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
        radius: float = 10.0,
        distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
        suppress_pssm_errors: bool = False,
    ):
        """
        Creates a residue graph from a single-residue variant in a .PDB file.

        Args:
            pdb_path (str): The path to the PDB file.
            chain_id (str): The .PDB chain identifier of the variant residue.
            residue_number (int): The number of the variant residue.
            insertion_code (str): The insertion code of the variant residue, set to None if not applicable.
            wildtype_amino_acid (:class:`AminoAcid`): The wildtype amino acid.
            variant_amino_acid (:class:`AminoAcid`): The variant amino acid.
            pssm_paths (Optional[Dict(str,str)], optional): The paths to the PSSM files, per chain identifier. Defaults to None.
            radius (float, optional): In Ångström, determines how many residues will be included in the graph. Defaults to 10.0.
            distance_cutoff (Optional[float], optional): Max distance in Ångström between a pair of atoms to consider them as an external edge in the graph.
                Defaults to 4.5.
            targets (Optional[Dict(str,float)], optional): Named target values associated with this query. Defaults to None.
            suppress_pssm_errors (bool, optional): Suppress error raised if .pssm files do not match .pdb files and throw warning instead.
                Defaults to False.
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets, suppress_pssm_errors)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius
        self._distance_cutoff = distance_cutoff

    @property
    def residue_id(self) -> str:
        "String representation of the residue number and insertion code."

        if self._insertion_code is not None:

            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        "Returns the string representing the complete query ID."
        return f"residue-graph:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}:{self.model_id}"

    def build(self, feature_modules: List[ModuleType], include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the .PDB structure.

        Args:
            feature_modules (List[ModuleType]): Each must implement the :py:func:`add_features` function.
            include_hydrogens (bool, optional): Whether to include hydrogens in the :class:`Graph`. Defaults to False.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """

        # load .PDB structure
        if isinstance(feature_modules, List):
            load_pssms = conservation in feature_modules
        else:
            load_pssms = conservation == feature_modules
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens, load_pssms)

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
                f"Residue not found in {self._pdb_path}: {self._chain_id} {self.residue_id}"
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

        graph.center = variant_residue.get_center()
        return graph


class SingleResidueVariantAtomicQuery(Query):

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id: str,
        residue_number: int,
        insertion_code: str,
        wildtype_amino_acid: AminoAcid,
        variant_amino_acid: AminoAcid,
        pssm_paths: Optional[Dict[str, str]] = None,
        radius: float = 10.0,
        distance_cutoff: Optional[float] = 4.5,
        targets: Optional[Dict[str, float]] = None,
        suppress_pssm_errors: bool = False,
    ):
        """
        Creates an atomic graph for a single-residue variant in a .PDB file.

        Args:
            pdb_path (str): The path to the .PDB file.
            chain_id (str): The .PDB chain identifier of the variant residue.
            residue_number (int): The number of the variant residue.
            insertion_code (str): The insertion code of the variant residue, set to None if not applicable.
            wildtype_amino_acid (:class:`AminoAcid`): The wildtype amino acid.
            variant_amino_acid (:class:`AminoAcid`): The variant amino acid.
            pssm_paths (Optional[Dict(str,str)], optional): The paths to the .PSSM files, per chain identifier. Defaults to None.
            radius (float, optional): In Ångström, determines how many residues will be included in the graph. Defaults to 10.0.
            distance_cutoff (Optional[float], optional): Max distance in Ångström between a pair of atoms to consider them as an external edge in the graph.
                Defaults to 4.5.
            targets (Optional[Dict(str,float)], optional): Named target values associated with this query. Defaults to None.
            suppress_pssm_errors (bool, optional): Suppress error raised if .pssm files do not match .pdb files and throw warning instead.
                Defaults to False.
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets, suppress_pssm_errors)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius

        self._distance_cutoff = distance_cutoff

    @property
    def residue_id(self) -> str:
        "String representation of the residue number and insertion code."

        if self._insertion_code is not None:
            return f"{self._residue_number}{self._insertion_code}"

        return str(self._residue_number)

    def get_query_id(self) -> str:
        "Returns the string representing the complete query ID."
        return f"atomic-graph:{self._chain_id}:{self.residue_id}:{self._wildtype_amino_acid.name}->{self._variant_amino_acid.name}:{self.model_id}"

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
        """
        Since pickle has problems serializing the graph when the nodes are atoms,
        this function can be used to generate a unique key for the atom.
        """

        # This should include the model, chain, residue and atom
        return str(atom)

    def build(self, feature_modules: Union[ModuleType, List[ModuleType]], include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the .PDB structure.

        Args:
            feature_modules (Union[ModuleType, List[ModuleType]]): Each must implement the :py:func:`add_features` function.
            include_hydrogens (bool, optional): Whether to include hydrogens in the :class:`Graph`. Defaults to False.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """

        # load .PDB structure
        if isinstance(feature_modules, List):
            load_pssms = conservation in feature_modules
        else:
            load_pssms = conservation == feature_modules
            feature_modules = [feature_modules]
        structure = self._load_structure(self._pdb_path, self._pssm_paths, include_hydrogens, load_pssms)

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
                f"Residue not found in {self._pdb_path}: {self._chain_id} {self.residue_id}"
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

        graph.center = variant_residue.get_center()
        return graph


def _load_ppi_atoms(pdb_path: str,
                    chain_id1: str, chain_id2: str,
                    distance_cutoff: float,
                    include_hydrogens: bool) -> List[Atom]:

    # get the contact atoms
    if include_hydrogens:

        pdb_name = os.path.basename(pdb_path)
        hydrogen_pdb_file, hydrogen_pdb_path = tempfile.mkstemp(
            prefix="hydrogenated-", suffix=pdb_name
        )
        os.close(hydrogen_pdb_file)

        add_hydrogens(pdb_path, hydrogen_pdb_path)

        try:
            contact_atoms = get_contact_atoms(hydrogen_pdb_path,
                                              chain_id1, chain_id2,
                                              distance_cutoff)
        finally:
            os.remove(hydrogen_pdb_path)
    else:
        contact_atoms = get_contact_atoms(pdb_path,
                                          chain_id1, chain_id2,
                                          distance_cutoff)

    if len(contact_atoms) == 0:
        raise ValueError("no contact atoms found")

    return contact_atoms


def _load_ppi_pssms(pssm_paths: Optional[Dict[str, str]],
                    chains: List[str],
                    structure: PDBStructure,
                    pdb_path,
                    suppress_error):

    _check_pssm(pdb_path, pssm_paths, suppress_error, verbosity = 0)
    for chain_id in chains:
        if chain_id in pssm_paths:

            chain = structure.get_chain(chain_id)

            pssm_path = pssm_paths[chain_id]

            with open(pssm_path, "rt", encoding="utf-8") as f:
                chain.pssm = parse_pssm(f, chain)


class ProteinProteinInterfaceAtomicQuery(Query):

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        distance_cutoff: Optional[float] = 5.5,
        targets: Optional[Dict[str, float]] = None,
        suppress_pssm_errors: bool = False,
    ):
        """
        A query that builds atom-based graphs, using the residues at a protein-protein interface.

        Args:
            pdb_path (str): The path to the .PDB file.
            chain_id1 (str): The .PDB chain identifier of the first protein of interest.
            chain_id2 (str): The .PDB chain identifier of the second protein of interest.
            pssm_paths (Optional[Dict(str,str)], optional): The paths to the .PSSM files, per chain identifier. Defaults to None.
            distance_cutoff (Optional[float], optional): Max distance in Ångström between two interacting atoms of the two proteins.
                Defaults to 5.5.
            targets (Optional[Dict(str,float)], optional): Named target values associated with this query. Defaults to None.
            suppress_pssm_errors (bool, optional): Suppress error raised if .pssm files do not match .pdb files and throw warning instead.
                Defaults to False.
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets, suppress_pssm_errors)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._distance_cutoff = distance_cutoff

    def get_query_id(self) -> str:
        "Returns the string representing the complete query ID."
        return f"atom-ppi:{self._chain_id1}-{self._chain_id2}:{self.model_id}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build(self, feature_modules: List[ModuleType], include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the .PDB structure.

        Args:
            feature_modules (List[ModuleType]): Each must implement the :py:func:`add_features` function.
            include_hydrogens (bool, optional): Whether to include hydrogens in the :class:`Graph`. Defaults to False.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """

        contact_atoms = _load_ppi_atoms(self._pdb_path,
                                        self._chain_id1, self._chain_id2,
                                        self._distance_cutoff,
                                        include_hydrogens)

        # build the graph
        graph = build_atomic_graph(
            contact_atoms, self.get_query_id(), self._distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        # read the pssm
        structure = contact_atoms[0].residue.chain.model

        if not isinstance(feature_modules, List):
            feature_modules = [feature_modules]
        if conservation in feature_modules:
            _load_ppi_pssms(self._pssm_paths,
                            [self._chain_id1, self._chain_id2],
                            structure, self._pdb_path,
                            suppress_error=self._suppress)

        # add the features
        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        graph.center = np.mean([atom.position for atom in contact_atoms], axis=0)
        return graph


class ProteinProteinInterfaceResidueQuery(Query):

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pdb_path: str,
        chain_id1: str,
        chain_id2: str,
        pssm_paths: Optional[Dict[str, str]] = None,
        distance_cutoff: Optional[float] = 10,
        targets: Optional[Dict[str, float]] = None,
        suppress_pssm_errors: bool = False,
    ):
        """
        A query that builds residue-based graphs, using the residues at a protein-protein interface.

        Args:
            pdb_path (str): The path to the .PDB file.
            chain_id1 (str): The .PDB chain identifier of the first protein of interest.
            chain_id2 (str): The .PDB chain identifier of the second protein of interest.
            pssm_paths (Optional[Dict(str,str)], optional): The paths to the .PSSM files, per chain identifier. Defaults to None.
            distance_cutoff (Optional[float], optional): Max distance in Ångström between two interacting residues of the two proteins.
                Defaults to 10.
            targets (Optional[Dict(str,float)], optional): Named target values associated with this query. Defaults to None.
            suppress_pssm_errors (bool, optional): Suppress error raised if .pssm files do not match .pdb files and throw warning instead.
                Defaults to False.
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets, suppress_pssm_errors)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._distance_cutoff = distance_cutoff

    def get_query_id(self) -> str:
        "Returns the string representing the complete query ID."
        return f"residue-ppi:{self._chain_id1}-{self._chain_id2}:{self.model_id}"

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, type(other))
            and self.model_id == other.model_id
            and {self._chain_id1, self._chain_id2}
            == {other._chain_id1, other._chain_id2}
        )

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build(self, feature_modules: List[ModuleType], include_hydrogens: bool = False) -> Graph:
        """Builds the graph from the .PDB structure.

        Args:
            feature_modules (List[ModuleType]): Each must implement the :py:func:`add_features` function.
            include_hydrogens (bool, optional): Whether to include hydrogens in the :class:`Graph`. Defaults to False.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """

        contact_atoms = _load_ppi_atoms(self._pdb_path,
                                        self._chain_id1, self._chain_id2,
                                        self._distance_cutoff,
                                        include_hydrogens)

        atom_positions = []
        residues_selected = set([])
        for atom in contact_atoms:
            atom_positions.append(atom.position)
            residues_selected.add(atom.residue)
        residues_selected = list(residues_selected)

        # build the graph
        graph = build_residue_graph(
            residues_selected, self.get_query_id(), self._distance_cutoff
        )

        # add data to the graph
        self._set_graph_targets(graph)

        # read the pssm
        structure = contact_atoms[0].residue.chain.model

        if not isinstance(feature_modules, List):
            feature_modules = [feature_modules]
        if conservation in feature_modules:
            _load_ppi_pssms(self._pssm_paths,
                            [self._chain_id1, self._chain_id2],
                            structure, self._pdb_path,
                            suppress_error=self._suppress)

        # add the features
        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        graph.center = np.mean(atom_positions, axis=0)
        return graph
