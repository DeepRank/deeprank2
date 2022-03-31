from typing import Union

import numpy


class PointFeature:
    def __init__(self, position: numpy.array, value: Union[float, numpy.array]):
        self.position = position
        self.value = value
