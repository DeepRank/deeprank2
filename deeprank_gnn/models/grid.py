from enum import Enum

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
    def __init__(self, settings: GridSettings, center: numpy.array):
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

    @property
    def xs(self) -> numpy.array:
        return self._xs

    @property
    def ys(self) -> numpy.array:
        return self._ys

    @property
    def zs(self) -> numpy.array:
        return self._zs

    def add_feature(self, name: str, data: numpy.array):
        self._features[name] = data


class MapMethod(Enum):
    GAUSSIAN = 1
    FAST_GAUSSIAN = 2
    BSP_LINE = 3
    NEAREST_NEIGHBOURS = 4
