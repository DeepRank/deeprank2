import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric.nn import max_pool_x
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_mean

from deeprank2.utils.community_pooling import community_pooling, get_preloaded_cluster

# ruff: noqa: ANN001, ANN201


class SGraphAttentionLayer(nn.Module):
    """SGraphAttentionLayer.

    This is a new layer that is similar to the graph attention network but simpler
    z_i =  1 / Ni \\Sum_j a_ij * [x_i || x_j] * W + b_i
    || is the concatenation operator: [1,2,3] || [4,5,6] = [1,2,3,4,5,6]
    Ni is the number of neighbor of node i
    \\Sum_j runs over the neighbors of node i
    a_ij is the edge attribute between node i and j

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        bias: If set to `False`, the layer will not learn an additive bias. Defaults to True.
    """  # noqa: D301

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        undirected: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.undirected = undirected

        self.weight = nn.Parameter(torch.Tensor(2 * in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        size = 2 * self.in_channels
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        num_node = len(x)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        # create edge feature by concatenating node feature
        alpha = torch.cat([x[row], x[col]], dim=-1)

        # multiply the edge features with the fliter
        alpha = torch.mm(alpha, self.weight)

        # multiply each edge features with the corresponding dist
        alpha = edge_attr * alpha

        # scatter the resulting edge feature to get node features
        out = torch.zeros(num_node, self.out_channels).to(alpha.device)
        out = scatter_mean(alpha, row, dim=0, out=out)

        # if the graph is undirected and (i,j) and (j,i) are both in
        # the edge_index then we do not need to have that second line
        # or we count everything twice
        if not self.undirected:
            out = scatter_mean(alpha, col, dim=0, out=out)

        # add the bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class SGAT(nn.Module):
    """Simple graph attention network, suited for both regression and classification tasks.

    It uses two graph attention layers and a MLP to predict the output.

    Args:
        input_shape: Size of each input sample.
        output_shape: Size of each output sample. Defaults to 1.
        input_shape_edge: Size of each input edge. Defaults to None.
    """

    def __init__(
        self,
        input_shape,
        output_shape=1,
        input_shape_edge=None,  # noqa: ARG002
    ):
        super().__init__()

        self.conv1 = SGraphAttentionLayer(input_shape, 16)
        self.conv2 = SGraphAttentionLayer(16, 32)

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, output_shape)

        self.clustering = "mcl"

    def forward(self, data):
        act = nn.Tanhshrink()
        act = relu

        # first conv block
        data.x = act(self.conv1(data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster0, data.batch)
        data = community_pooling(cluster, data)

        # second conv block
        data.x = act(self.conv2(data.x, data.edge_index, data.edge_attr))
        cluster = get_preloaded_cluster(data.cluster1, data.batch)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        # FC
        x = scatter_mean(x, batch, dim=0)
        x = act(self.fc1(x))
        x = self.fc2(x)

        return x  # noqa:RET504 (unnecessary-assign)
