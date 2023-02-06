from tempfile import mkstemp
import os
from typing import Tuple
import logging

import h5py
import numpy as np

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery, ProteinProteinInterfaceResidueQuery
from deeprankcore.utils.grid import MapMethod, GridSettings, Grid
import deeprankcore.features.contact


_log = logging.getLogger(__name__)


def test_residue_grid_orientation():

    coord_error_margin = 1.0  # Angstrom

    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]

    # Extract data from original deeprank's preprocessed file.
    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

        target_center = grid_points_group["center"][()]

    # Build the atomic graph, according to this repository's code.
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    chain_id1 = "C"
    chain_id2 = "D"
    distance_cutoff = 8.5

    query = ProteinProteinInterfaceResidueQuery(pdb_path, chain_id1, chain_id2,
                                                distance_cutoff=distance_cutoff)

    graph = query.build([])

    # Make a grid from the graph.
    map_method = MapMethod.FAST_GAUSSIAN
    grid_settings = GridSettings(points_counts, grid_sizes)
    grid = Grid("test_grid", graph.center, grid_settings)
    graph.map_to_grid(grid, map_method)

    assert np.all(np.abs(target_center - grid.center) < coord_error_margin), f"\n{grid.center} != \n{target_center}"

    # Orientation must be the same as in the original deeprank.
    # Check that the grid point coordinates are the same.
    assert grid.xs.shape == target_xs.shape
    assert np.all(np.abs(grid.xs - target_xs) < coord_error_margin), f"\n{grid.xs} != \n{target_xs}"

    assert grid.ys.shape == target_ys.shape
    assert np.all(np.abs(grid.ys - target_ys) < coord_error_margin), f"\n{grid.ys} != \n{target_ys}"

    assert grid.zs.shape == target_zs.shape
    assert np.all(np.abs(grid.zs - target_zs) < coord_error_margin), f"\n{grid.zs} != \n{target_zs}"


def test_atomic_grid_orientation():

    coord_error_margin = 1.0  # Angstrom

    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]

    # Extract data from original deeprank's preprocessed file.
    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

        target_center = grid_points_group["center"][()]

    # Build the atomic graph, according to this repository's code.
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    chain_id1 = "C"
    chain_id2 = "D"
    distance_cutoff = 8.5

    query = ProteinProteinInterfaceAtomicQuery(pdb_path, chain_id1, chain_id2,
                                               distance_cutoff=distance_cutoff)

    graph = query.build([])

    # Make a grid from the graph.
    map_method = MapMethod.FAST_GAUSSIAN
    grid_settings = GridSettings(points_counts, grid_sizes)
    grid = Grid("test_grid", graph.center, grid_settings)
    graph.map_to_grid(grid, map_method)

    assert np.all(np.abs(target_center - grid.center) < coord_error_margin), f"\n{grid.center} != \n{target_center}"

    # Orientation must be the same as in the original deeprank.
    # Check that the grid point coordinates are the same.
    assert grid.xs.shape == target_xs.shape
    assert np.all(np.abs(grid.xs - target_xs) < coord_error_margin), f"\n{grid.xs} != \n{target_xs}"

    assert grid.ys.shape == target_ys.shape
    assert np.all(np.abs(grid.ys - target_ys) < coord_error_margin), f"\n{grid.ys} != \n{target_ys}"

    assert grid.zs.shape == target_zs.shape
    assert np.all(np.abs(grid.zs - target_zs) < coord_error_margin), f"\n{grid.zs} != \n{target_zs}"


def _inflate(index: np.ndarray, value: np.ndarray, shape: Tuple[int]):

    size = 1
    for dim in shape:
        size *= dim

    data = np.zeros(size)
    data[index] = value[:, 0]

    return data.reshape(shape)


def test_grid_features():
    "Check that grid mapped features produce output that makes sense"

    error_margin = 0.001

    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:

        original_chain1_electrostatic = _inflate(data_file["1AK4/mapped_features/Feature_ind/coulomb_chain1/index"][:],
                                                 data_file["1AK4/mapped_features/Feature_ind/coulomb_chain1/value"][:],
                                                 (10, 10, 10))

        original_chain2_electrostatic = _inflate(data_file["1AK4/mapped_features/Feature_ind/coulomb_chain2/index"][:],
                                                 data_file["1AK4/mapped_features/Feature_ind/coulomb_chain2/value"][:],
                                                 (10, 10, 10))

        original_chain1_vanderwaals = _inflate(data_file["1AK4/mapped_features/Feature_ind/vdwaals_chain1/index"][:],
                                               data_file["1AK4/mapped_features/Feature_ind/vdwaals_chain1/value"][:],
                                               (10, 10, 10))

        original_chain2_vanderwaals = _inflate(data_file["1AK4/mapped_features/Feature_ind/vdwaals_chain2/index"][:],
                                               data_file["1AK4/mapped_features/Feature_ind/vdwaals_chain2/value"][:],
                                               (10, 10, 10))

    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"

    query = ProteinProteinInterfaceAtomicQuery(pdb_path, "C", "D",
                                               distance_cutoff=8.5)

    graph = query.build([deeprankcore.features.contact])

    map_method = MapMethod.FAST_GAUSSIAN
    grid_settings = GridSettings([10, 10, 10], [30.0, 30.0, 30.0])

    hdf5_file, hdf5_path = mkstemp()
    os.close(hdf5_file)

    try:
        graph.write_as_grid_to_hdf5(hdf5_path, grid_settings, map_method)

        with h5py.File(hdf5_path, 'r') as data_file:

            entry_name = list(data_file.keys())[0]

            _log.debug(f"entry is {entry_name}")

            electrostatic_data = data_file[entry_name]["mapped_features/electrostatic/value"][:]
            vanderwaals_data = data_file[entry_name]["mapped_features/vanderwaals/value"][:]
    finally:
        os.remove(hdf5_path)

    assert np.all(np.abs(original_chain1_electrostatic + original_chain2_electrostatic - electrostatic_data) < error_margin), \
            f"difference is {np.max(np.abs(original_chain1_electrostatic + original_chain2_electrostatic - electrostatic_data))}"
    assert np.all(np.abs(original_chain1_vanderwaals + original_chain2_vanderwaals - vanderwaals_data) < error_margin), \
            f"difference is {np.max(np.abs(original_chain1_vanderwaals + original_chain2_vanderwaals - vanderwaals_data))}"

