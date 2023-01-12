from tempfile import mkstemp
import os
from typing import List

import h5py
import numpy as np
from pdb2sql import interface

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery
import deeprankcore.features.contact
from deeprankcore.utils.grid import MapMethod, GridSettings, Grid
from deeprankcore.molstruct.atom import AtomicElement
from deeprankcore.utils.buildgraph import get_structure


def _inflate(index: np.array, value: np.array, shape: List[int]):

    data = np.zeros(shape[0] * shape[1] * shape[2])

    data[index] = value[:,0]

    return data.reshape(shape)


def test_grid_orientation():

    error_margin = 0.001

    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]
    carbon_vanderwaals_radius = 1.7

    # Extract data from original deeprank's preprocessed file.
    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

        target_center = grid_points_group["center"][()]

        c_group = data_file["1AK4/mapped_features/AtomicDensities_ind/C_chain1"]
        chain1_c_index = c_group["index"][()]
        chain1_c_value = c_group["value"][()]

        target_chain1_densities_carbon = _inflate(chain1_c_index, chain1_c_value, points_counts)

    # Build the atomic graph, according to this repository's code.
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    chain_id1 = "C"
    chain_id2 = "D"
    distance_cutoff = 8.5

    query = ProteinProteinInterfaceAtomicQuery(pdb_path, chain_id1, chain_id2,
                                               distance_cutoff=distance_cutoff)

    graph = query.build([deeprankcore.features.contact])

    # Get atomic positions.
    pdb = interface(pdb_path)
    try:
        contact_atoms_by_chain = pdb.get_contact_atoms(cutoff=distance_cutoff, chain1=chain_id1, chain2=chain_id2)
        contact_atom_indices = []
        for atom_indices in contact_atoms_by_chain.values():
            contact_atom_indices.extend(atom_indices)

        center = np.mean(pdb.get("x,y,z", rowID=list(set(contact_atom_indices))), axis=0)

        chain1_carbon_positions = pdb.get("x,y,z", rowID=contact_atoms_by_chain[chain_id1], element="C")
    finally:
        pdb._close()

    assert np.all(np.abs(target_center - center) < error_margin), f"\n{center} != \n{target_center}"

    # Make a grid from the graph.
    map_method = MapMethod.GAUSSIAN
    grid_settings = GridSettings(center, points_counts, grid_sizes)
    grid = Grid("test_grid", grid_settings)
    graph.map_to_grid(grid, map_method)

    # Orientation must be the same as in the original deeprank.
    # Check that the grid point coordinates are the same.
    assert grid.xs.shape == target_xs.shape
    assert np.all(np.abs(grid.xs - target_xs) < error_margin), f"\n{grid.xs} != \n{target_xs}"

    assert grid.ys.shape == target_ys.shape
    assert np.all(np.abs(grid.ys - target_ys) < error_margin), f"\n{grid.ys} != \n{target_ys}"

    assert grid.zs.shape == target_zs.shape
    assert np.all(np.abs(grid.zs - target_zs) < error_margin), f"\n{grid.zs} != \n{target_zs}"

    # Map the atomic densities for carbon.
    chain1_densities_carbon = np.zeros(target_chain1_densities_carbon.shape)
    for position in chain1_carbon_positions:
        chain1_densities_carbon += grid._get_atomic_density_koes(position, carbon_vanderwaals_radius)

    # Check that the carbon densities are the same as in the original deeprank.
    assert np.all(np.abs(chain1_densities_carbon - target_chain1_densities_carbon) < error_margin)
