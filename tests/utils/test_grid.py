import h5py
import numpy as np

from deeprank2.query import VALID_RESOLUTIONS, ProteinProteinInterfaceQuery
from deeprank2.utils.grid import Grid, GridSettings, MapMethod


def test_grid_orientation() -> None:
    coord_error_margin = 1.0  # Angstrom
    points_counts = [10, 10, 10]
    grid_sizes = [30.0, 30.0, 30.0]

    # Extract data from original deeprank's preprocessed file.
    with h5py.File("tests/data/hdf5/original-deeprank-1ak4.hdf5", "r") as data_file:
        grid_points_group = data_file["1AK4/grid_points"]
        target_xs = grid_points_group["x"][()]
        target_ys = grid_points_group["y"][()]
        target_zs = grid_points_group["z"][()]
        target_center = grid_points_group["center"][()]

    for resolution in VALID_RESOLUTIONS:
        print(f"Testing for {resolution} level grids.")  # noqa:T201; in case pytest fails, this will be printed.
        query = ProteinProteinInterfaceQuery(
            pdb_path="tests/data/pdb/1ak4/1ak4.pdb",
            resolution=resolution,
            chain_ids=["C", "D"],
            influence_radius=8.5,
            max_edge_length=8.5,
        )
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
