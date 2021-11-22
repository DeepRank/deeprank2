import torch

from enum import Enum


class Polarity(Enum):
    APOLAR = 0
    POLAR = 1
    NEGATIVE_CHARGE = 2
    POSITIVE_CHARGE = 3

    def onehot(self):
        value = torch.zeros(4)
        value[self.value] = 1.0

        return value
