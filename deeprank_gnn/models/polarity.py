import numpy

from enum import Enum


class Polarity(Enum):
    "a value to express a residue's polarity"

    APOLAR = 0
    POLAR = 1
    NEGATIVE_CHARGE = 2
    POSITIVE_CHARGE = 3

    @property
    def onehot(self):
        t = numpy.zeros(4)
        t[self.value] = 1.0

        return t
