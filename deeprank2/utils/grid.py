"""This module holds the classes that are used when working with a 3D grid."""

from __future__ import annotations

import itertools
import logging
from enum import Enum
from typing import TYPE_CHECKING

import h5py
import numpy as np
from scipy.interpolate import BSpline

from deeprank2.domain import gridstorage

if TYPE_CHECKING:
    from numpy.typing import NDArray

_log = logging.getLogger(__name__)


class MapMethod(Enum):
    """This holds the value of either one of 4 grid mapping methods.

    A mapping method determines how feature point values are divided over the grid points.
    """

    GAUSSIAN = 1
    FAST_GAUSSIAN = 2
    BSP_LINE = 3
    NEAREST_NEIGHBOURS = 4


class Augmentation:
    """A rotation around an axis, to be applied to a feature before mapping it to a grid."""

    def __init__(self, axis: NDArray, angle: float):
        self._axis = axis
        self._angle = angle

    @property
    def axis(self) -> NDArray:
        return self._axis

    @property
    def angle(self) -> float:
        return self._angle


class GridSettings:
    """Objects of this class hold the settings to build a grid.

    The grid is basically a multi-divided 3D cube with
    the following properties:
    - points_counts: the number of points on the x, y, z edges of the cube
    - sizes: x, y, z sizes of the box in Å
    - resolutions: the size in Å of one x, y, z edge subdivision. Also the distance between two points on the edge.
    """

    def __init__(
        self,
        points_counts: list[int],
        sizes: list[float],
    ):
        if len(points_counts) != 3 or len(sizes) != 3:  # noqa:PLR2004
            msg = "Incorrect grid dimensions."
            raise ValueError(msg)

        self._points_counts = points_counts
        self._sizes = sizes

    @property
    def resolutions(self) -> list[float]:
        return [self._sizes[i] / self._points_counts[i] for i in range(3)]

    @property
    def sizes(self) -> list[float]:
        return self._sizes

    @property
    def points_counts(self) -> list[int]:
        return self._points_counts


class Grid:
    """A 3D (volumetric) representation of a `Graph`.

    A Grid contains the following information:

    - coordinates of points
    - names of features
    - feature values on each point.
    """

    def __init__(self, id_: str, center: list[float], settings: GridSettings):
        self.id = id_
        self._center = np.array(center)
        self._settings = settings
        self._set_mesh(self._center, settings)
        self._features = {}

    def _set_mesh(self, center: NDArray, settings: GridSettings) -> None:
        """Builds the grid points."""
        half_size_x = settings.sizes[0] / 2
        half_size_y = settings.sizes[1] / 2
        half_size_z = settings.sizes[2] / 2

        min_x = center[0] - half_size_x
        max_x = min_x + (settings.points_counts[0] - 1.0) * settings.resolutions[0]
        self._xs = np.linspace(min_x, max_x, num=settings.points_counts[0])

        min_y = center[1] - half_size_y
        max_y = min_y + (settings.points_counts[1] - 1.0) * settings.resolutions[1]
        self._ys = np.linspace(min_y, max_y, num=settings.points_counts[1])

        min_z = center[2] - half_size_z
        max_z = min_z + (settings.points_counts[2] - 1.0) * settings.resolutions[2]
        self._zs = np.linspace(min_z, max_z, num=settings.points_counts[2])

        self._ygrid, self._xgrid, self._zgrid = np.meshgrid(self._ys, self._xs, self._zs)

    @property
    def center(self) -> NDArray:
        return self._center

    @property
    def xs(self) -> NDArray:
        return self._xs

    @property
    def xgrid(self) -> NDArray:
        return self._xgrid

    @property
    def ys(self) -> NDArray:
        return self._ys

    @property
    def ygrid(self) -> NDArray:
        return self._ygrid

    @property
    def zs(self) -> NDArray:
        return self._zs

    @property
    def zgrid(self) -> NDArray:
        return self._zgrid

    @property
    def features(self) -> dict[str, NDArray]:
        return self._features

    def add_feature_values(self, feature_name: str, data: NDArray) -> None:
        """Makes sure feature values per grid point get stored.

        This method may be called repeatedly to add on to existing grid point values.
        """
        if feature_name not in self._features:
            self._features[feature_name] = data
        else:
            self._features[feature_name] += data

    def _get_mapped_feature_gaussian(
        self,
        position: NDArray,
        value: float,
    ) -> NDArray:
        beta = 1.0

        fx, fy, fz = position
        distances = np.sqrt((self.xgrid - fx) ** 2 + (self.ygrid - fy) ** 2 + (self.zgrid - fz) ** 2)

        return value * np.exp(-beta * distances)

    def _get_mapped_feature_fast_gaussian(self, position: NDArray, value: float) -> NDArray:
        beta = 1.0
        cutoff = 5.0 * beta

        fx, fy, fz = position
        distances = np.sqrt((self.xgrid - fx) ** 2 + (self.ygrid - fy) ** 2 + (self.zgrid - fz) ** 2)

        data = np.zeros(distances.shape)

        data[distances < cutoff] = value * np.exp(-beta * distances[distances < cutoff])

        return data

    def _get_mapped_feature_bsp_line(
        self,
        position: NDArray,
        value: float,
    ) -> NDArray:
        order = 4

        fx, fy, fz = position
        bsp_data = (
            BSpline((self.xgrid - fx) / self._settings.resolutions[0], order)
            * BSpline((self.ygrid - fy) / self._settings.resolutions[1], order)
            * BSpline((self.zgrid - fz) / self._settings.resolutions[2], order)
        )

        return value * bsp_data

    def _get_mapped_feature_nearest_neighbour(
        self,
        position: NDArray,
        value: float,
    ) -> NDArray:
        fx, _, _ = position
        distances_x = np.abs(self.xs - fx)
        distances_y = np.abs(self.ys - fx)
        distances_z = np.abs(self.zs - fx)

        indices_x = np.argsort(distances_x)[:2]
        indices_y = np.argsort(distances_y)[:2]
        indices_z = np.argsort(distances_z)[:2]

        sorted_x = distances_x[indices_x]
        weights_x = sorted_x / np.sum(sorted_x)

        sorted_y = distances_y[indices_y]
        weights_y = sorted_y / np.sum(sorted_y)

        sorted_z = distances_z[indices_z]
        weights_z = sorted_z / np.sum(sorted_z)

        indices = [indices_x, indices_y, indices_z]
        points = list(itertools.product(*indices))

        weight_products = list(itertools.product(weights_x, weights_y, weights_z))
        weights = [np.sum(p) for p in weight_products]

        neighbour_data = np.zeros((self.xs.shape[0], self.ys.shape[0], self.zs.shape[0]))

        for point_index, point in enumerate(points):
            weight = weights[point_index]

            neighbour_data[point] = weight * value

        return neighbour_data

    def _get_atomic_density_koes(
        self,
        position: NDArray,
        vanderwaals_radius: float,
    ) -> NDArray:
        """Function to map individual atomic density on the grid.

        The formula is equation (1) of the Koes paper
        Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1.

        Returns:
            NDArray: The mapped density.
        """
        distances = np.sqrt(np.square(self.xgrid - position[0]) + np.square(self.ygrid - position[1]) + np.square(self.zgrid - position[2]))

        density_data = np.zeros(distances.shape)

        indices_close = distances < vanderwaals_radius
        indices_far = (distances >= vanderwaals_radius) & (distances < 1.5 * vanderwaals_radius)

        density_data[indices_close] = np.exp(-2.0 * np.square(distances[indices_close]) / np.square(vanderwaals_radius))
        density_data[indices_far] = (
            4.0 / np.square(np.e) / np.square(vanderwaals_radius) * np.square(distances[indices_far])
            - 12.0 / np.square(np.e) / vanderwaals_radius * distances[indices_far]
            + 9.0 / np.square(np.e)
        )

        return density_data

    def map_feature(
        self,
        position: NDArray,
        feature_name: str,
        feature_value: NDArray | float,
        method: MapMethod,
    ) -> None:
        """Maps point feature data at a given position to the grid, using the given method.

        The feature_value should either be a single number or a one-dimensional array.
        """
        # determine whether we're dealing with a single number of multiple numbers:
        index_names_values = []
        if isinstance(feature_value, float):
            index_names_values = [(feature_name, feature_value)]

        elif isinstance(feature_value, int):
            index_names_values = [(feature_name, float(feature_value))]

        else:
            for index, value in enumerate(feature_value):
                index_name = f"{feature_name}_{index:03d}"
                index_names_values.append((index_name, value))

        # map the data to the grid
        for index_name, value in index_names_values:
            if method == MapMethod.GAUSSIAN:
                grid_data = self._get_mapped_feature_gaussian(position, value)

            elif method == MapMethod.FAST_GAUSSIAN:
                grid_data = self._get_mapped_feature_fast_gaussian(position, value)

            elif method == MapMethod.BSP_LINE:
                grid_data = self._get_mapped_feature_bsp_line(position, value)

            elif method == MapMethod.NEAREST_NEIGHBOURS:
                grid_data = self._get_mapped_feature_nearest_neighbour(position, value)

            # set to grid
            self.add_feature_values(index_name, grid_data)

    def to_hdf5(self, hdf5_path: str) -> None:
        """Write the grid data to hdf5, according to deeprank standards."""
        with h5py.File(hdf5_path, "a") as hdf5_file:
            # create a group to hold everything
            grid_group = hdf5_file.require_group(self.id)

            # store grid points
            points_group = grid_group.require_group("grid_points")
            points_group.create_dataset("x", data=self.xs)
            points_group.create_dataset("y", data=self.ys)
            points_group.create_dataset("z", data=self.zs)
            points_group.create_dataset("center", data=self.center)

            # store grid features
            features_group = grid_group.require_group(gridstorage.MAPPED_FEATURES)
            for feature_name, feature_data in self.features.items():
                features_group.create_dataset(
                    feature_name,
                    data=feature_data,
                    compression="lzf",
                    chunks=True,
                )
