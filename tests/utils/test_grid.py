from tempfile import mkstemp
import os

import h5py
import numpy as np

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery
import deeprankcore.features.contact
from deeprankcore.utils.grid import MapMethod, GridSettings


def test_grid_orientation():

    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

    query = ProteinProteinInterfaceAtomicQuery("tests/data/pdb/1ak4/1ak4.pdb",
                                               "C", "D",
                                               distance_cutoff=8.5)

    graph = query.build([deeprankcore.features.contact])

    grid_file, grid_path = mkstemp(suffix=".hdf5")
    os.close(grid_file)

    points_counts = [target_xs.shape[0], target_ys.shape[0], target_zs.shape[0]]
    grid_sizes = [np.max(target_xs) - np.min(target_xs),
                  np.max(target_ys) - np.min(target_ys),
                  np.max(target_zs) - np.min(target_zs)]

    grid_settings = GridSettings(points_counts, grid_sizes)

    graph.write_as_grid_to_hdf5(grid_path, grid_settings, MapMethod.GAUSSIAN)

    try:
        with h5py.File(grid_path, 'r') as data_file:
            entry_group = data_file[list(data_file.keys())[0]]
            grid_points_group = entry_group["grid_points"]

            xs = grid_points_group["x"][()]
            ys = grid_points_group["y"][()]
            zs = grid_points_group["z"][()]
    finally:
        os.remove(grid_path)

    assert xs.shape == target_xs.shape
    assert np.all(xs == target_xs)

    assert ys.shape == target_ys.shape
    assert np.all(ys == target_ys)

    assert zs.shape == target_zs.shape
    assert np.all(zs == target_zs)
