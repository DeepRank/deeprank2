import numpy

from enum import Enum


class Polarity(Enum):
    APOLAR = 0
    POLAR = 1
    NEGATIVE_CHARGE = 2
    POSITIVE_CHARGE = 3

    @property
    def onehot(self):
        t = numpy.zeros(4)
        t[self.value] = 1.0

        return t
