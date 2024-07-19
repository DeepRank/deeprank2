import torch
from torch import nn
from torch.nn.functional import dropout, leaky_relu, relu, softmax
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_mean, scatter_sum

# ruff: noqa: ANN001, ANN201


class GINetConvLayer(nn.Module):
    """GiNet convolutional layer for graph neural networks.

    Args:
        in_channels: Number of input features.
        out_channels: Number of output features.
        number_edge_features: Number of edge features. Defaults to 1.
        bias: If set to `False`, the layer will not learn an additive bias. Defaults to False.
    """

    def __init__(self, in_channels, out_channels, number_edge_features=1, bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.fc_edge_attr = nn.Linear(number_edge_features, number_edge_features, bias=bias)
        self.fc_attention = nn.Linear(2 * self.out_channels + number_edge_features, 1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        size = self.in_channels
        uniform(size, self.fc.weight)
        uniform(size, self.fc_attention.weight)
        uniform(size, self.fc_edge_attr.weight)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        num_node = len(x)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        xcol = self.fc(x[col])
        xrow = self.fc(x[row])

        ed = self.fc_edge_attr(edge_attr)
        # create edge feature by concatenating node feature
        alpha = torch.cat([xrow, xcol, ed], dim=1)
        alpha = self.fc_attention(alpha)
        alpha = leaky_relu(alpha)

        alpha = softmax(alpha, dim=1)
        h = alpha * xcol

        out = torch.zeros(num_node, self.out_channels).to(alpha.device)
        z = scatter_sum(h, row, dim=0, out=out)

        return z  # noqa:RET504 (unnecessary-assign)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class GINet(nn.Module):
    """Architecture based on the GiNet convolutional layer, suited for both regression and classification tasks.

    Args:
        input_shape: Number of input features.
        output_shape: Number of output value per graph. Defaults to 1.
        input_shape_edge: Number of edge input features. Defaults to 1.
    """

    def __init__(self, input_shape, output_shape=1, input_shape_edge=1):
        super().__init__()
        self.conv1 = GINetConvLayer(input_shape, 16, input_shape_edge)
        self.conv2 = GINetConvLayer(16, 32, input_shape_edge)

        self.conv1_ext = GINetConvLayer(input_shape, 16, input_shape_edge)
        self.conv2_ext = GINetConvLayer(16, 32, input_shape_edge)

        self.fc1 = nn.Linear(2 * 32, 128)
        self.fc2 = nn.Linear(128, output_shape)
        self.dropout = 0.4

    def forward(self, data):
        act = relu
        data_ext = data.clone()

        # EXTERNAL INTERACTION GRAPH
        # first conv block
        data.x = act(self.conv1(data.x, data.edge_index, data.edge_attr))

        # second conv block
        data.x = act(self.conv2(data.x, data.edge_index, data.edge_attr))

        # INTERNAL INTERACTION GRAPH
        # first conv block
        data_ext.x = act(self.conv1_ext(data_ext.x, data_ext.edge_index, data_ext.edge_attr))

        # second conv block
        data_ext.x = act(self.conv2_ext(data_ext.x, data_ext.edge_index, data_ext.edge_attr))

        # FC
        x = scatter_mean(data.x, data.batch, dim=0)
        x_ext = scatter_mean(data_ext.x, data_ext.batch, dim=0)

        x = torch.cat([x, x_ext], dim=1)
        x = act(self.fc1(x))
        x = dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)

        return x  # noqa:RET504 (unnecessary-assign)
