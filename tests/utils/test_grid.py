from tempfile import mkstemp
import os

import h5py

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery
import deeprankcore.features.contact
from deeprankcore.utils.grid import MapMethod, Grid, GridSettings


def test_grid_orientation():

    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

    query = ProteinProteinInterfaceAtomicQuery("tests/data/pdb/1ak5/1ak5.pdb",
                                               "C", "D",
                                               distance_cutoff=8.5)

    graph = query.build([contact])

    grid_path, grid_file = mkstemp(suffix=".hdf5")
    os.close(grid_file)

    graph.write_as_grid_to_hdf5
