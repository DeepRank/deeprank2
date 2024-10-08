import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import relu

# ruff: noqa: ANN001, ANN201, ANN202

######################################################################
#
# Model automatically generated by modelGenerator
#
######################################################################

# ----------------------------------------------------------------------
# Network Structure
# ----------------------------------------------------------------------
# conv layer   0: conv | input -1  output  4  kernel  2  post relu
# conv layer   1: pool | kernel  2  post None
# conv layer   2: conv | input  4  output  5  kernel  2  post relu
# conv layer   3: pool | kernel  2  post None
# fc   layer   0: fc   | input -1  output  84  post relu
# fc   layer   1: fc   | input  84  output  1  post None
# ----------------------------------------------------------------------


class CnnRegression(nn.Module):
    """Convolutional Neural Network architecture for regression.

    This type of network is used to predict a single scalar value of a continuous variable.

    Args:
        num_features: Number of features in the input data.
        box_shape: Shape of the input data.
    """

    def __init__(self, num_features: int, box_shape: tuple[int]):
        super().__init__()

        self.convlayer_000 = nn.Conv3d(num_features, 4, kernel_size=2)
        self.convlayer_001 = nn.MaxPool3d((2, 2, 2))
        self.convlayer_002 = nn.Conv3d(4, 5, kernel_size=2)
        self.convlayer_003 = nn.MaxPool3d((2, 2, 2))

        size = self._get_conv_output(num_features, box_shape)

        self.fclayer_000 = nn.Linear(size, 84)
        self.fclayer_001 = nn.Linear(84, 1)

    def _get_conv_output(self, num_features: int, shape: tuple[int]):
        num_data_points = 2
        input_ = Variable(torch.rand(num_data_points, num_features, *shape))
        output = self._forward_features(input_)
        return output.data.view(num_data_points, -1).size(1)

    def _forward_features(self, x):
        x = relu(self.convlayer_000(x))
        x = self.convlayer_001(x)
        x = relu(self.convlayer_002(x))
        x = self.convlayer_003(x)
        return x  # noqa:RET504 (unnecessary-assign)

    def forward(self, data):
        x = self._forward_features(data.x)
        x = x.view(x.size(0), -1)
        x = relu(self.fclayer_000(x))
        x = self.fclayer_001(x)
        return x  # noqa:RET504 (unnecessary-assign)


######################################################################
#
# Model automatically generated by modelGenerator
#
######################################################################

# ----------------------------------------------------------------------
# Network Structure
# ----------------------------------------------------------------------
# conv layer   0: conv | input -1  output  4  kernel  2  post relu
# conv layer   1: pool | kernel  2  post None
# conv layer   2: conv | input  4  output  5  kernel  2  post relu
# conv layer   3: pool | kernel  2  post None
# fc   layer   0: fc   | input -1  output  84  post relu
# fc   layer   1: fc   | input  84  output  1  post None
# ----------------------------------------------------------------------


class CnnClassification(nn.Module):
    """Convolutional Neural Network architecture for binary classification.

    This type of network is used to predict the class of an input data point.

    Args:
        num_features: Number of features in the input data.
        box_shape: Shape of the input data.
    """

    def __init__(self, num_features, box_shape):
        super().__init__()

        self.convlayer_000 = nn.Conv3d(num_features, 4, kernel_size=2)
        self.convlayer_001 = nn.MaxPool3d((2, 2, 2))
        self.convlayer_002 = nn.Conv3d(4, 5, kernel_size=2)
        self.convlayer_003 = nn.MaxPool3d((2, 2, 2))

        size = self._get_conv_output(num_features, box_shape)

        self.fclayer_000 = nn.Linear(size, 84)
        self.fclayer_001 = nn.Linear(84, 2)

    def _get_conv_output(self, num_features, shape):
        inp = Variable(torch.rand(1, num_features, *shape))
        out = self._forward_features(inp)
        return out.data.view(1, -1).size(1)

    def _forward_features(self, x):
        x = relu(self.convlayer_000(x))
        x = self.convlayer_001(x)
        x = relu(self.convlayer_002(x))
        x = self.convlayer_003(x)
        return x  # noqa:RET504 (unnecessary-assign)

    def forward(self, data):
        x = self._forward_features(data.x)
        x = x.view(x.size(0), -1)
        x = relu(self.fclayer_000(x))
        x = self.fclayer_001(x)
        return x  # noqa:RET504 (unnecessary-assign)
