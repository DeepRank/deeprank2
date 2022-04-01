from typing import Dict

import h5py
import numpy
import itertools
from scipy.signal import bspline

from deeprank_gnn.models.grid import Grid, MapMethod


def map_feature_gaussian(grid: Grid, position: numpy.ndarray, value: float):

    beta = 1.0

    fx, fy, fz = position
    distances = numpy.sqrt((grid.xgrid - fx) ** 2 + (grid.ygrid - fy) ** 2 + (grid.zgrid - fz) ** 2)

    return value * numpy.exp(-beta * distances)


def map_feature_fast_gaussian(grid: Grid, position: numpy.ndarray, value: float):

    beta = 1.0
    cutoff = 5.0 * beta

    fx, fy, fz = position
    distances = numpy.sqrt((grid.xgrid - fx) ** 2 + (grid.ygrid - fy) ** 2 + (grid.zgrid - fz) ** 2)

    data = numpy.zeros(distances.shape)

    data[distances < cutoff] = value * numpy.exp(-beta * distances[distances < cutoff])

    return data


def map_feature_bsp_line(grid: Grid, position: numpy.ndarray, value: float):

    order = 4

    fx, fy, fz = position
    bsp_data = (bspline((grid.xgrid - fx) / grid.resolution, order) *
                bspline((grid.ygrid - fy) / grid.resolution, order) *
                bspline((grid.zgrid - fz) / grid.resolution, order))

    return value * bsp_data


def map_feature_nearest_neighbour(grid: Grid, position: numpy.ndarray, value: float):

    fx, fy, fz = position
    distances_x = numpy.abs(grid.xs - fx)
    distances_y = numpy.abs(grid.ys - fx)
    distances_z = numpy.abs(grid.zs - fx)

    indices_x = numpy.argsort(distances_x)[:2]
    indices_y = numpy.argsort(distances_y)[:2]
    indices_z = numpy.argsort(distances_z)[:2]

    sorted_x = distances_x[indices_x]
    weights_x = sorted_x / numpy.sum(sorted_x)

    sorted_y = distances_y[indices_y]
    weights_y = sorted_y / numpy.sum(sorted_y)

    sorted_z = distances_z[indices_z]
    weights_z = sorted_z / numpy.sum(sorted_z)

    indices = [indices_x, indices_y, indices_z]
    points = list(itertools.product(*indices))

    weight_products = list(itertools.product(weights_x, weights_y, weights_z))
    weights = [numpy.sum(p) for p in weight_products]

    neighbour_data = numpy.zeros((grid.xs.shape[0], grid.ys.shape[0], grid.zs.shape[0]))

    for point_index, point in enumerate(points):
        weight = weights[point_index]

        neighbour_data[point] = weight * value

    return neighbour_data


def map_features(grid: Grid, position: numpy.ndarray, feature_name: str, feature_value: numpy.ndarray, method: MapMethod):

    for index, value in enumerate(feature_value):

        index_name = "{}_{:03d}".format(feature_name, index)

        if method == MapMethod.GAUSSIAN:
            grid_data = map_feature_gaussian(grid, position, value)

        elif method == MapMethod.FAST_GAUSSIAN:
            grid_data = map_feature_fast_gaussian(grid, position, value)

        elif method == MapMethod.BSP_LINE:
            grid_data = map_feature_bsp_line(grid, position, value)

        elif method == MapMethod.NEAREST_NEIGHBOUR:
            grid_data = map_feature_nearest_neighbour(grid, position, value)

        # set to grid
        grid.add_feature_values(index_name, grid_data)


def grid_to_hdf5(grid: Grid, hdf5_file: h5py.File):

    grid_group = hdf5_file.require_group(grid.id)

    # store grid points
    points_group = grid_group.require_group("grid_points")
    points_group.create_dataset("x", data=grid.xs)
    points_group.create_dataset("y", data=grid.ys)
    points_group.create_dataset("z", data=grid.zs)
    points_group.create_dataset("center", data=grid.center)

    # store grid features
    features_group = grid_group.require_group("mapped_features")
    for feature_name, feature_data in grid.features.items():
        feature_group = features_group.require_group(feature_name)
        feature_group.create_dataset("value", data=feature_data, compression="lzf", chunks=True)
