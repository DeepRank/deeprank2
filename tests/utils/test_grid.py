from typing import List

import h5py
import numpy as np

from deeprankcore.query import ProteinProteinInterfaceAtomicQuery, ProteinProteinInterfaceResidueQuery
import deeprankcore.features.contact
import deeprankcore.features.surfacearea
from deeprankcore.utils.grid import MapMethod, GridSettings, Grid
from deeprankcore.molstruct.atom import AtomicElement
from deeprankcore.domain.nodestorage import BSA as BSA_FEATURE


def _inflate(index: np.array, value: np.array, shape: List[int]):

    data = np.zeros(shape[0] * shape[1] * shape[2])

    data[index] = value[:,0]

    return data.reshape(shape)


def test_residue_grid_orientation():

    error_margin = 0.001

    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]

    # Extract data from original deeprank's preprocessed file.
    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", 'r') as data_file:
        grid_points_group = data_file["1AK4/grid_points"]

        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]

        target_center = grid_points_group["center"][()]

        bsa_chain1_group = data_file["1AK4/mapped_features/Feature_ind/bsa_chain1"]
        chain1_bsa_index = bsa_chain1_group["index"][()]
        chain1_bsa_value = bsa_chain1_group["value"][()]
        target_chain1_bsa = _inflate(chain1_bsa_index, chain1_bsa_value, points_counts)

        bsa_chain2_group = data_file["1AK4/mapped_features/Feature_ind/bsa_chain2"]
        chain2_bsa_index = bsa_chain2_group["index"][()]
        chain2_bsa_value = bsa_chain2_group["value"][()]
        target_chain2_bsa = _inflate(chain2_bsa_index, chain2_bsa_value, points_counts)

    target_bsa = target_chain1_bsa + target_chain2_bsa

    # Build the atomic graph, according to this repository's code.
    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"
    chain_id1 = "C"
    chain_id2 = "D"
    distance_cutoff = 8.5

    query = ProteinProteinInterfaceResidueQuery(pdb_path, chain_id1, chain_id2,
                                                distance_cutoff=distance_cutoff)

    graph = query.build([deeprankcore.features.surfacearea])

    # Make a grid from the graph.
    map_method = MapMethod.FAST_GAUSSIAN
    grid_settings = GridSettings(points_counts, grid_sizes)
    grid = Grid("test_grid", graph.center, grid_settings)
    graph.map_to_grid(grid, map_method)

    assert np.all(np.abs(target_center - grid.center) < error_margin), f"\n{grid.center} != \n{target_center}"

    # Orientation must be the same as in the original deeprank.
    # Check that the grid point coordinates are the same.
    assert grid.xs.shape == target_xs.shape
    assert np.all(np.abs(grid.xs - target_xs) < error_margin), f"\n{grid.xs} != \n{target_xs}"

    assert grid.ys.shape == target_ys.shape
    assert np.all(np.abs(grid.ys - target_ys) < error_margin), f"\n{grid.ys} != \n{target_ys}"

    assert grid.zs.shape == target_zs.shape
    assert np.all(np.abs(grid.zs - target_zs) < error_margin), f"\n{grid.zs} != \n{target_zs}"

    # Get the bsa from the grid
    bsa = grid.features[BSA_FEATURE]

    assert bsa.shape == target_bsa.shape, f"{bsa.shape} != {target_bsa.shape}"

    for i in range(bsa.shape[0]):
        assert np.all(np.abs(bsa[i] - target_bsa[i]) < error_margin), f"{i}:\n{bsa[i]} !=\n{target_bsa[i]}"


def test_atomic_grid_orientation():

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

    # Get atomic positions from the graph
    chain1_carbon_positions = [node.id.position  # the node id is actually the atom
                               for node in graph.nodes
                               if node.id.residue.chain.id == chain_id1 and
                                    node.id.element == AtomicElement.C]

    # Make a grid from the graph.
    map_method = MapMethod.FAST_GAUSSIAN
    grid_settings = GridSettings(points_counts, grid_sizes)
    grid = Grid("test_grid", graph.center, grid_settings)
    graph.map_to_grid(grid, map_method)

    assert np.all(np.abs(target_center - grid.center) < error_margin), f"\n{grid.center} != \n{target_center}"

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
