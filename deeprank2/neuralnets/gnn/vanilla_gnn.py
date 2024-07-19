# Example of network that doesn't require clusters.

import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_sum

# ruff: noqa: ANN001, ANN201


class VanillaConvolutionalLayer(nn.Module):
    """Vanilla convolutional layer for graph neural networks.

    Args:
        count_node_features: Number of node features.
        count_edge_features: Number of edge features.
    """

    def __init__(self, count_node_features, count_edge_features):
        super().__init__()
        message_size = 32
        edge_input_size = 2 * count_node_features + count_edge_features
        self._edge_mlp = nn.Sequential(nn.Linear(edge_input_size, message_size), nn.ReLU())
        node_input_size = count_node_features + message_size
        self._node_mlp = nn.Sequential(nn.Linear(node_input_size, count_node_features), nn.ReLU())

    def forward(self, node_features, edge_node_indices, edge_features):
        # generate messages over edges
        node0_indices, node1_indices = edge_node_indices
        node0_features = node_features[node0_indices]
        node1_features = node_features[node1_indices]
        message_input = torch.cat([node0_features, node1_features, edge_features], dim=1)
        messages_per_neighbour = self._edge_mlp(message_input)
        # aggregate messages
        out = torch.zeros(node_features.shape[0], messages_per_neighbour.shape[1]).to(node_features.device)
        message_sums_per_node = scatter_sum(messages_per_neighbour, node0_indices, dim=0, out=out)
        # update nodes
        node_input = torch.cat([node_features, message_sums_per_node], dim=1)
        return self._node_mlp(node_input)


class VanillaNetwork(nn.Module):
    """Vanilla graph neural network architecture suited for both regression and classification tasks.

    It uses two vanilla convolutional layers and a MLP to predict the output.

    Args:
        input_shape: Number of node input features.
        output_shape: Number of output value per graph.
        input_shape_edge: Number of edge input features.
    """

    def __init__(self, input_shape: int, output_shape: int, input_shape_edge: int):
        super().__init__()
        self._external1 = VanillaConvolutionalLayer(input_shape, input_shape_edge)
        self._external2 = VanillaConvolutionalLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = nn.Sequential(nn.Linear(input_shape, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_shape))

    def forward(self, data):
        external_updated1_node_features = self._external1(data.x, data.edge_index, data.edge_attr)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_index, data.edge_attr)
        means_per_graph_external = scatter_mean(external_updated2_node_features, data.batch, dim=0)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z  # noqa:RET504 (unnecessary-assign)
