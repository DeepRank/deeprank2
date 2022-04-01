from enum import Enum
from typing import Dict

import numpy


class GridSettings:
    def __init__(self, points_count: int, size: float):
        self._points_count = points_count
        self._size = size

    @property
    def resolution(self) -> float:
        return self._size / self._points_count

    @property
    def size(self) -> float:
        return self._size

    @property
    def points_count(self) -> int:
        return self._points_count


class Grid:
    def __init__(self, id_:str, settings: GridSettings, center: numpy.array):
        self.id = id_

        self._settings = settings
        self._center = center

        self._resize(settings, center)

        self._features = {}

    def _resize(self, settings: GridSettings, center: numpy.array):

        half_size = settings.size / 2

        min_x = center[0] - half_size
        max_x = center[0] + half_size
        self._xs = numpy.linspace(min_x, max_x, num=settings.points_count)

        min_y = center[1] - half_size
        max_y = center[1] + half_size
        self._ys = numpy.linspace(min_y, max_y, num=settings.points_count)

        min_z = center[2] - half_size
        max_z = center[2] + half_size
        self._zs = numpy.linspace(min_z, max_z, num=settings.points_count)

        self._ygrid, self._xgrid, self._zgrid = numpy.meshgrid(self._ys, self._xs, self._zs)

    @property
    def xs(self) -> numpy.array:
        return self._xs

    @property
    def xgrid(self) -> numpy.array:
        return self._xgrid

    @property
    def ys(self) -> numpy.array:
        return self._ys

    @property
    def ygrid(self) -> numpy.array:
        return self._ygrid

    @property
    def zs(self) -> numpy.array:
        return self._zs

    @property
    def zgrid(self) -> numpy.array:
        return self._zgrid

    @property
    def center(self) -> numpy.array:
        return self._center

    @property
    def features(self) -> Dict[str, numpy.array]:
        return self._features

    def add_feature_values(self, key: str, data: numpy.ndarray):
        """ Makes sure feature values per grid point get stored.

            This method may be called repeatedly to add on to existing grid point values.
        """

        if key not in self._features:
            self._features[key] = data
        else:
            self._features[key] += data


class MapMethod(Enum):
    GAUSSIAN = 1
    FAST_GAUSSIAN = 2
    BSP_LINE = 3
    NEAREST_NEIGHBOURS = 4
