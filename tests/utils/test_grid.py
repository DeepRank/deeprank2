from tempfile import mkstemp
import os
from typing import List

import h5py
import numpy as np
from pdb2sql import interface

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery
import deeprankcore.features.contact
from deeprankcore.utils.grid import MapMethod, GridSettings


def _inflate(index: np.array, value: np.array, shape: List[int]):

    data = np.zeros(shape[0] * shape[1] * shape[2])

    data[index] = value[:,0]

    return data.reshape(shape)


def test_grid_orientation():

    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]

    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

        target_center = grid_points_group["center"][()]

        c_group = data_file["1AK4/mapped_features/AtomicDensities_ind/C_chain1"]
        chain1_c_index = c_group["index"][()]
        chain1_c_value = c_group["value"][()]
        target_chain1_c = _inflate(chain1_c_index, chain1_c_value, points_counts)

    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    chain1 = "C"
    chain2 = "D"
    distance_cutoff = 8.5

    query = ProteinProteinInterfaceAtomicQuery(pdb_path, chain1, chain2,
                                               distance_cutoff=distance_cutoff)

    graph = query.build([deeprankcore.features.contact])

    grid_file, grid_path = mkstemp(suffix=".hdf5")
    os.close(grid_file)

    pdb = interface(pdb_path)
    try:
        contact_atoms = pdb.get_contact_atoms(cutoff=distance_cutoff, chain1=chain1, chain2=chain2)
        contact_atom_indices = []
        for atom_indices in contact_atoms.values():
            contact_atom_indices.extend(atom_indices)

        center = np.mean(pdb.get("x,y,z", rowID=list(set(contact_atom_indices))), axis=0)
    finally:
        pdb._close()

    assert np.all(np.abs(target_center - center) < 0.000000001), f"\n{center} != \n{target_center}"

    grid_settings = GridSettings(center, points_counts, grid_sizes)

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
    assert np.all(np.abs(xs - target_xs) < 0.000000001), f"\n{xs} != \n{target_xs}"

    assert ys.shape == target_ys.shape
    assert np.all(np.abs(ys - target_ys) < 0.000000001), f"\n{ys} != \n{target_ys}"

    assert zs.shape == target_zs.shape
    assert np.all(np.abs(zs - target_zs) < 0.000000001), f"\n{zs} != \n{target_zs}"

