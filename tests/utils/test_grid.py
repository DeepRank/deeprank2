import h5py
import numpy as np

from deeprank2.query import (
    ProteinProteinInterfaceAtomicQuery,
    ProteinProteinInterfaceResidueQuery,
)
from deeprank2.utils.grid import Grid, GridSettings, MapMethod


def test_residue_grid_orientation(): # pylint: disable=too-many-locals

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


def test_atomic_grid_orientation(): # pylint: disable=too-many-locals

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
