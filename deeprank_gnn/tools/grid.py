from typing import Union

import numpy

from deeprank_gnn.models.grid import Grid, MapMethod
from deeprank_gnn.models.graph import Edge, Node
from deeprank_gnn.models.feature import PointFeature


def map_feature_gaussian(grid: Grid, position: numpy.array, feature_name: str: feature_value: Union[float, numpy.array]):



def map_node_feature(grid: Grid, feature_name: str, feature: PointFeature, method: MapMethod):

    if method == MapMethod.GAUSSIAN:
        map_feature_gaussian(grid, feature.position, feature_name, feature.value)

    elif method == MapMethod.FAST_GAUSSIAN:
        map_feature_fast_gaussian(grid, feature.position, feature_name, feature.value)

    elif method == MapMethod.BSP_LINE:
        map_feature_bsp_line(grid, feature.position, feature_name, feature.value)

    elif method == MapMethod.NEAREST_NEIGHBOUR:
        map_feature_nearest_neighbour(grid, feature.position, feature_name, feature.value)


