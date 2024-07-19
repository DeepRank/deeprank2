from __future__ import annotations

import importlib
import logging
import os
import pickle
import pkgutil
import re
import warnings
from dataclasses import MISSING, dataclass, field, fields
from functools import partial
from glob import glob
from multiprocessing import Pool
from random import randrange
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pdb2sql

import deeprank2.features
from deeprank2.domain.aminoacidlist import convert_aa_nomenclature
from deeprank2.features import components, conservation, contact
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.buildgraph import get_contact_atoms, get_structure, get_surrounding_residues
from deeprank2.utils.graph import Graph
from deeprank2.utils.grid import Augmentation, GridSettings, MapMethod
from deeprank2.utils.parsing.pssm import parse_pssm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deeprank2.molstruct.aminoacid import AminoAcid
    from deeprank2.molstruct.structure import PDBStructure

_log = logging.getLogger(__name__)

VALID_RESOLUTIONS = ["atom", "residue"]


@dataclass(repr=False, kw_only=True)
class Query:
    """Parent class of :class:`SingleResidueVariantQuery` and :class:`ProteinProteinInterfaceQuery`.

    More detailed information about the parameters can be found in :class:`SingleResidueVariantQuery` and :class:`ProteinProteinInterfaceQuery`.
    """

    pdb_path: str
    resolution: Literal["residue", "atom"]
    chain_ids: list[str] | str
    pssm_paths: dict[str, str] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    influence_radius: float | None = None
    max_edge_length: float | None = None
    suppress_pssm_errors: bool = False

    def __post_init__(self):
        self._model_id = os.path.splitext(os.path.basename(self.pdb_path))[0]
        self.variant = None  # not used for PPI, overwritten for SRV

        if self.resolution == "residue":
            self.max_edge_length = 10 if not self.max_edge_length else self.max_edge_length
            self.influence_radius = 10 if not self.influence_radius else self.influence_radius
        elif self.resolution == "atom":
            self.max_edge_length = 4.5 if not self.max_edge_length else self.max_edge_length
            self.influence_radius = 4.5 if not self.influence_radius else self.influence_radius
        else:
            msg = f"Invalid resolution given ({self.resolution}). Must be one of {VALID_RESOLUTIONS}"
            raise ValueError(msg)

        if not isinstance(self.chain_ids, list):
            self.chain_ids = [self.chain_ids]

        # convert None to empty type (e.g. list, dict) for arguments where this is expected
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())

    def _set_graph_targets(self, graph: Graph) -> None:
        """Copy target data from query to graph."""
        for target_name, target_data in self.targets.items():
            graph.targets[target_name] = target_data

    def _load_structure(self) -> PDBStructure:
        """Build PDBStructure objects from pdb and pssm data."""
        pdb = pdb2sql.pdb2sql(self.pdb_path)
        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()  # noqa: SLF001
        # read the pssm
        if self._pssm_required:
            self._load_pssm_data(structure)

        return structure

    def _load_pssm_data(self, structure: PDBStructure) -> None:
        self._check_pssm()
        for chain in structure.chains:
            if chain.id in self.pssm_paths:
                pssm_path = self.pssm_paths[chain.id]
                with open(pssm_path, encoding="utf-8") as f:
                    chain.pssm = parse_pssm(f, chain)

    def _check_pssm(self, verbosity: Literal[0, 1, 2] = 0) -> None:  # noqa: C901
        """Checks whether information stored in pssm file matches the corresponding pdb file.

        Args:
            pdb_path: Path to the PDB file.
            pssm_paths: The paths to the PSSM files, per chain identifier.
            suppress: Suppress errors and throw warnings instead.
            verbosity: Level of verbosity of error/warning. Defaults to 0.
                0 (low): Only state file name where error occurred;
                1 (medium): Also state number of incorrect and missing residues;
                2 (high): Also list the incorrect residues

        Raises:
            ValueError: Raised if info between pdb file and pssm file doesn't match or if no pssms were provided.
        """
        if not self.pssm_paths:
            msg = "No pssm paths provided for conservation feature module."
            raise ValueError(msg)

        # load residues from pssm and pdb files
        pssm_file_residues = {}
        for chain, pssm_path in self.pssm_paths.items():
            with open(pssm_path, encoding="utf-8") as f:
                lines = f.readlines()[1:]
            for line in lines:
                pssm_file_residues[chain + line.split()[0].zfill(4)] = convert_aa_nomenclature(line.split()[1], 3)
        pdb_file_residues = {res[0] + str(res[2]).zfill(4): res[1] for res in pdb2sql.pdb2sql(self.pdb_path).get_residues() if res[0] in self.pssm_paths}

        # list errors
        mismatches = []
        missing_entries = []
        for residue in pdb_file_residues:
            try:
                if pdb_file_residues[residue] != pssm_file_residues[residue]:
                    mismatches.append(residue)
            except KeyError:  # noqa: PERF203
                missing_entries.append(residue)

        # generate error message
        if len(mismatches) + len(missing_entries) > 0:
            error_message = f"Amino acids in PSSM files do not match pdb file for {os.path.split(self.pdb_path)[1]}."
            if verbosity:
                if len(mismatches) > 0:
                    error_message = error_message + f"\n\t{len(mismatches)} entries are incorrect."
                    if verbosity == 2:  # noqa: PLR2004
                        error_message = error_message[-1] + f":\n\t{missing_entries}"
                if len(missing_entries) > 0:
                    error_message = error_message + f"\n\t{len(missing_entries)} entries are missing."
                    if verbosity == 2:  # noqa: PLR2004
                        error_message = error_message[-1] + f":\n\t{missing_entries}"

            # raise exception (or warning)
            if not self.suppress_pssm_errors:
                raise ValueError(error_message)
            warnings.warn(error_message)
            _log.warning(error_message)

    @property
    def model_id(self) -> str:
        """The ID of the model, usually a .PDB accession code."""
        return self._model_id

    @model_id.setter
    def model_id(self, value: str) -> None:
        self._model_id = value

    def __repr__(self) -> str:
        return f"{type(self)}({self.get_query_id()})"

    def build(
        self,
        feature_modules: list[str | ModuleType],
    ) -> Graph:
        """Builds the graph from the .PDB structure.

        Args:
            feature_modules: the feature modules used to build the graph. These must be filenames existing inside `deeprank2.features` subpackage.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """
        if not isinstance(feature_modules, list):
            feature_modules = [feature_modules]
        feature_modules = [importlib.import_module("deeprank2.features." + module) if isinstance(module, str) else module for module in feature_modules]
        self._pssm_required = conservation in feature_modules
        graph = self._build_helper()

        # add target and feature data to the graph
        self._set_graph_targets(graph)
        for feature_module in feature_modules:
            feature_module.add_features(self.pdb_path, graph, self.variant)

        return graph

    def _build_helper(self) -> Graph:
        msg = "Must be defined in child classes."
        raise NotImplementedError(msg)

    def get_query_id(self) -> str:
        msg = "Must be defined in child classes."
        raise NotImplementedError(msg)


@dataclass(kw_only=True)
class SingleResidueVariantQuery(Query):
    """A query that builds a single residue variant graph.

    Args:
        pdb_path: the path to the PDB file to query.
        resolution: sets whether each node is a residue or atom.
        chain_ids: the chain identifier of the variant residue (generally a single capital letter).
            Note that this does not limit the structure to residues from this chain.
        pssm_paths: the name of the chain(s) (key) and path to the pssm file(s) (value).
        targets: Name(s) (key) and target value(s) (value) associated with this query.
        influence_radius: all residues within this radius from the variant residue will be included in the graph, irrespective of the chain they are on.
        max_edge_length: the maximum distance between two nodes to generate an edge connecting them.
        suppress_pssm_errors: Whether to suppress the error raised if the .pssm files do not match the .pdb files. If True, a warning is returned instead.
        variant_residue_number: the residue number of the variant residue.
        insertion_code: the insertion code of the variant residue.
        wildtype_amino_acid: the amino acid at above position in the wildtype protein.
        variant_amino_acid: the amino acid at above position in the variant protein.
        radius: all Residues within this radius (in Å) from the variant residue will be included in the graph.
    """

    variant_residue_number: int
    insertion_code: str | None
    wildtype_amino_acid: AminoAcid
    variant_amino_acid: AminoAcid

    def __post_init__(self):
        super().__post_init__()  # calls __post_init__ of parents

        if len(self.chain_ids) != 1:
            raise ValueError("`chain_ids` must contain exactly 1 chain for `SingleResidueVariantQuery` objects, " + f"but {len(self.chain_ids)} were given.")
        self.variant_chain_id = self.chain_ids[0]

    @property
    def residue_id(self) -> str:
        """String representation of the residue number and insertion code."""
        if self.insertion_code is not None:
            return f"{self.variant_residue_number}{self.insertion_code}"
        return str(self.variant_residue_number)

    def get_query_id(self) -> str:
        """Returns the string representing the complete query ID."""
        return (
            f"{self.resolution}-srv:"
            f"{self.variant_chain_id}:{self.residue_id}:"
            f"{self.wildtype_amino_acid.name}->{self.variant_amino_acid.name}:{self.model_id}"
        )

    def _build_helper(self) -> Graph:
        """Helper function to build a graph for SRV queries.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """
        # load .PDB structure
        structure = self._load_structure()

        # find the variant residue and its surroundings
        variant_residue: Residue = None
        for residue in structure.get_chain(self.variant_chain_id).residues:
            if residue.number == self.variant_residue_number and residue.insertion_code == self.insertion_code:
                variant_residue = residue
                break
        if variant_residue is None:
            msg = f"Residue not found in {self.pdb_path}: {self.variant_chain_id} {self.residue_id}"
            raise ValueError(msg)
        self.variant = SingleResidueVariant(variant_residue, self.variant_amino_acid)
        residues = get_surrounding_residues(
            structure,
            variant_residue,
            self.influence_radius,
        )

        # build the graph
        if self.resolution == "residue":
            graph = Graph.build_graph(
                residues,
                self.get_query_id(),
                self.max_edge_length,
            )
        elif self.resolution == "atom":
            residues.append(variant_residue)
            atoms = set()
            for residue in residues:
                if residue.amino_acid is not None:
                    for atom in residue.atoms:
                        atoms.add(atom)
            atoms = list(atoms)

            graph = Graph.build_graph(atoms, self.get_query_id(), self.max_edge_length)

        else:
            msg = f"No function exists to build graphs with resolution of {self.resolution}."
            raise NotImplementedError(msg)
        graph.center = variant_residue.get_center()

        return graph


@dataclass(kw_only=True)
class ProteinProteinInterfaceQuery(Query):
    """A query that builds a protein-protein interface graph.

    Args:
        pdb_path: the path to the PDB file to query.
        resolution: sets whether each node is a residue or atom.
        chain_ids: the chain identifiers of the interacting interfaces (generally a single capital letter each).
            Note that this does not limit the structure to residues from these chains.
        pssm_paths: the name of the chain(s) (key) and path to the pssm file(s) (value).
        targets: Name(s) (key) and target value(s) (value) associated with this query.
        influence_radius: all residues within this radius from the interacting interface will be included in the graph, irrespective of the chain they are on.
        max_edge_length: the maximum distance between two nodes to generate an edge connecting them.
        suppress_pssm_errors: Whether to suppress the error raised if the .pssm files do not match the .pdb files. If True, a warning is returned instead.
    """

    def __post_init__(self):
        super().__post_init__()

        if len(self.chain_ids) != 2:  # noqa: PLR2004
            raise ValueError(
                "`chain_ids` must contain exactly 2 chains for `ProteinProteinInterfaceQuery` objects, " + f"but {len(self.chain_ids)} was/were given.",
            )

    def get_query_id(self) -> str:
        """Returns the string representing the complete query ID."""
        return (
            f"{self.resolution}-ppi:"  # resolution and query type (ppi for protein protein interface)
            f"{self.chain_ids[0]}-{self.chain_ids[1]}:{self.model_id}"
        )

    def _build_helper(self) -> Graph:
        """Helper function to build a graph for PPI queries.

        Returns:
            :class:`Graph`: The resulting :class:`Graph` object with all the features and targets.
        """
        # find the atoms near the contact interface
        contact_atoms = get_contact_atoms(
            self.pdb_path,
            self.chain_ids,
            self.influence_radius,
        )
        if len(contact_atoms) == 0:
            msg = "No contact atoms found"
            raise ValueError(msg)

        # build the graph
        if self.resolution == "atom":
            graph = Graph.build_graph(
                contact_atoms,
                self.get_query_id(),
                self.max_edge_length,
            )
        elif self.resolution == "residue":
            residues_selected = list({atom.residue for atom in contact_atoms})
            graph = Graph.build_graph(
                residues_selected,
                self.get_query_id(),
                self.max_edge_length,
            )

        graph.center = np.mean([atom.position for atom in contact_atoms], axis=0)
        structure = contact_atoms[0].residue.chain.model
        if self._pssm_required:
            self._load_pssm_data(structure)

        return graph


class QueryCollection:
    """Represents the collection of data queries that will be processed.

    The class attributes are set either while adding queries to the collection (`_queries`
    and `_ids_count`), or when processing the collection (other attributes).

    Attributes:
        _queries (list[:class:`Query`]): The `Query` objects in the collection.
        _ids_count (dict[str, int]): The original `query_id` and the repeat number for this id.
            This is used to rename the `query_id` to ensure that there are no duplicate ids.
        _prefix, _cpu_count, _grid_settings, etc.: See docstring for `QueryCollection.process`.

    Notes:
        Queries can be saved as a dictionary to easily navigate through their data,
        using `QueryCollection.export_dict()`.
    """

    def __init__(self):
        self._queries: list[Query] = []
        self._ids_count: dict[str, int] = {}
        self._prefix: str | None = None
        self._cpu_count: int | None = None
        self._grid_settings: GridSettings | None = None
        self._grid_map_method: MapMethod | None = None
        self._grid_augmentation_count: int = 0

    def add(
        self,
        query: Query,
        verbose: bool = False,
        warn_duplicate: bool = True,
    ) -> None:
        """Add a new query to the collection.

        Args:
            query: The `Query` to add to the collection.
            verbose: For logging query IDs added. Defaults to `False`.
            warn_duplicate: Log a warning before renaming if a duplicate query is identified. Defaults to `True`.
        """
        query_id = query.get_query_id()
        if verbose:
            _log.info(f"Adding query with ID {query_id}.")

        if query_id not in self._ids_count:
            self._ids_count[query_id] = 1
        else:
            self._ids_count[query_id] += 1
            new_id = query.model_id + "_" + str(self._ids_count[query_id])
            query.model_id = new_id
            if warn_duplicate:
                _log.warning(f"Query with ID {query_id} has already been added to the collection. Renaming it as {query.get_query_id()}")

        self._queries.append(query)

    def export_dict(self, dataset_path: str) -> None:
        """Exports the colection of all queries to a dictionary file.

        Args:
            dataset_path: The path where to save the list of queries.
        """
        with open(dataset_path, "wb") as pkl_file:
            pickle.dump(self, pkl_file)

    @property
    def queries(self) -> list[Query]:
        """The list of queries added to the collection."""
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)

    def __len__(self) -> int:
        return len(self._queries)

    def _process_one_query(self, query: Query) -> None:
        """Only one process may access an hdf5 file at a time."""
        try:
            output_path = f"{self._prefix}-{os.getpid()}.hdf5"
            graph = query.build(self._feature_modules)
            graph.write_to_hdf5(output_path)

            if self._grid_settings is not None and self._grid_map_method is not None:
                graph.write_as_grid_to_hdf5(
                    output_path,
                    self._grid_settings,
                    self._grid_map_method,
                )
                for _ in range(self._grid_augmentation_count):
                    # repeat with random augmentation
                    axis, angle = pdb2sql.transform.get_rot_axis_angle(randrange(100))
                    augmentation = Augmentation(axis, angle)
                    graph.write_as_grid_to_hdf5(
                        output_path,
                        self._grid_settings,
                        self._grid_map_method,
                        augmentation,
                    )

        except (ValueError, AttributeError, KeyError, TimeoutError) as e:
            _log.warning(
                f"\nGraph/Query with ID {query.get_query_id()} ran into an Exception ({e.__class__.__name__}: {e}),"
                " and it has not been written to the hdf5 file. More details below:",
            )
            _log.exception(e)

    def process(
        self,
        prefix: str = "processed-queries",
        feature_modules: list[ModuleType, str] | ModuleType | str | None = None,
        cpu_count: int | None = None,
        combine_output: bool = True,
        grid_settings: GridSettings | None = None,
        grid_map_method: MapMethod | None = None,
        grid_augmentation_count: int = 0,
    ) -> list[str]:
        """Render queries into graphs (and optionally grids).

        Args:
            prefix: Prefix for naming the output files. Defaults to "processed-queries".
            feature_modules: Feature module or list of feature modules used to generate features (given as string or as an imported module).
                Each module must implement the :py:func:`add_features` function, and all feature modules must exist inside `deeprank2.features` folder.
                If set to 'all', all available modules in `deeprank2.features` are used to generate the features.
                Defaults to the two primary feature modules `deeprank2.features.components` and `deeprank2.features.contact`.
            cpu_count: The number of processes to be run in parallel (i.e. number of CPUs used), capped by the number of CPUs available to the system.
                Defaults to None, which takes all available cpu cores.
            combine_output:
                If `True` (default): all processes are combined into a single HDF5 file.
                If `False`: separate HDF5 files are created for each process (i.e. for each CPU used).
            grid_settings: If valid together with `grid_map_method`, the grid data will be stored as well. Defaults to None.
            grid_map_method: If valid together with `grid_settings`, the grid data will be stored as well. Defaults to None.
            grid_augmentation_count: Number of grid data augmentations (must be >= 0). Defaults to 0.

        Returns:
            The list of paths of the generated HDF5 files.
        """
        # set defaults
        feature_modules = feature_modules or [components, contact]
        self._prefix = "processed-queries" if not prefix else re.sub(".hdf5$", "", prefix)  # scrape extension if present

        max_cpus = os.cpu_count()
        self._cpu_count = max_cpus if cpu_count is None else min(cpu_count, max_cpus)
        if cpu_count and self._cpu_count < cpu_count:
            _log.warning(f"\nTried to set {cpu_count} CPUs, but only {max_cpus} are present in the system.")
        _log.info(f"\nNumber of CPUs for processing the queries set to: {self._cpu_count}.")

        self._feature_modules = self._set_feature_modules(feature_modules)
        _log.info(f"\nSelected feature modules: {self._feature_modules}.")

        self._grid_settings = grid_settings
        self._grid_map_method = grid_map_method

        if grid_augmentation_count < 0:
            msg = f"`grid_augmentation_count` cannot be negative, but was given as {grid_augmentation_count}"
            raise ValueError(msg)
        self._grid_augmentation_count = grid_augmentation_count

        _log.info(f"Creating pool function to process {len(self)} queries...")
        pool_function = partial(self._process_one_query)
        with Pool(self._cpu_count) as pool:
            _log.info("Starting pooling...\n")
            pool.map(pool_function, self.queries)

        output_paths = glob(f"{prefix}-*.hdf5")
        if combine_output:
            for output_path in output_paths:
                with h5py.File(f"{prefix}.hdf5", "a") as f_dest, h5py.File(output_path, "r") as f_src:
                    for key, value in f_src.items():
                        _log.debug(f"copy {key} from {output_path} to {prefix}.hdf5")
                        f_src.copy(value, f_dest)
                os.remove(output_path)
            return glob(f"{prefix}.hdf5")

        return output_paths

    def _set_feature_modules(self, feature_modules: list[ModuleType, str] | ModuleType | str) -> list[str]:
        """Convert `feature_modules` to list[str] irrespective of input type.

        Raises:
            TypeError: if an invalid input type is passed.
        """
        if feature_modules == "all":
            return [modname for _, modname, _ in pkgutil.iter_modules(deeprank2.features.__path__)]
        if isinstance(feature_modules, ModuleType):
            return [os.path.basename(feature_modules.__file__)[:-3]]
        if isinstance(feature_modules, str):
            return [re.sub(".py$", "", feature_modules)]  # scrape extension if present
        if isinstance(feature_modules, list):
            invalid_inputs = [type(el) for el in feature_modules if not isinstance(el, str | ModuleType)]
            if invalid_inputs:
                msg = f"`feature_modules` contains invalid input ({invalid_inputs}). Only `str` and `ModuleType` are accepted."
                raise TypeError(msg)
            return [
                re.sub(".py$", "", m) if isinstance(m, str) else os.path.basename(m.__file__)[:-3]  # for ModuleTypes
                for m in feature_modules
            ]
        msg = f"`feature_modules` has received an invalid input type: {type(feature_modules)}. Only `str` and `ModuleType` are accepted."
        raise TypeError(msg)
